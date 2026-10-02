"""
ClipIQ - Streamlit frontend.

Presentation/orchestration layer only. All AI work is delegated to the existing backend:

    utils.audio_processor.process_input(source)        -> audio chunks
    core.transcriber.transcribe_all(chunks)            -> transcript text
    core.summarize.generate_title / summarize          -> title / summary
    core.extractor.extract_*                           -> decisions / action items / questions
    core.rag_engine.build_rag_chain / ask_question     -> retrieval-augmented Q&A

The stages are called one by one, in the same order as main.run_pipeline(), so the UI can
report real stage boundaries and measured timings instead of a fake progress bar.

Run with:  streamlit run app.py      (requires streamlit >= 1.36)
"""

from __future__ import annotations

import inspect
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from html import escape as html_escape
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

import streamlit as st

# Make `core/` and `utils/` importable no matter where Streamlit is launched from.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

APP_NAME = "ClipIQ"
APP_VERSION = "1.0"
TAGLINE = "Turn long-form video into structured intelligence."

st.set_page_config(
    page_title=f"{APP_NAME} | Video Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ai_video_assistant")

SOURCE_YOUTUBE = "YouTube URL"
SOURCE_FILE = "Upload File"
SOURCE_KINDS = [SOURCE_YOUTUBE, SOURCE_FILE]

LANGUAGES: dict[str, Optional[str]] = {"English": "en", "Hindi": "hi", "Auto Detect": None}

ALLOWED_EXTENSIONS = ["mp4", "mkv", "mov", "avi", "webm", "mp3", "wav", "m4a", "flac", "ogg", "aac"]
YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com")

# (label, group, description). Stages in the same group run inside one backend call.
STAGES: list[tuple[str, str, str]] = [
    ("Fetching media", "fetch", "Resolving the source and retrieving the media"),
    ("Extracting audio", "fetch", "Converting and chunking audio for transcription"),
    ("Transcribing", "transcribe", "Converting speech to text"),
    ("Generating summary", "summary", "Writing the title and executive summary"),
    ("Extracting insights", "insights", "Finding decisions, action items and questions"),
    ("Building knowledge base", "rag", "Indexing the transcript for retrieval"),
    ("Finalizing results", "finalize", "Assembling your workspace"),
]
GROUPS = list(dict.fromkeys(g for _, g, _ in STAGES))
GROUP_LABELS = {
    "fetch": "Fetching media and extracting audio",
    "transcribe": "Transcribing",
    "summary": "Generating summary",
    "insights": "Extracting insights",
    "rag": "Building knowledge base",
    "finalize": "Finalizing results",
    "query": "Answering question",
    "startup": "Starting the AI engine",
}
LAST_IN_GROUP = {g: max(i for i, (_, gg, _) in enumerate(STAGES) if gg == g) for g in GROUPS}

SUGGESTED_QUESTIONS = [
    "What are the main ideas discussed?",
    "What decisions were made?",
    "What action items were mentioned?",
    "Explain the most important concept from this video.",
]

TRANSCRIPT_VIEW_HEIGHT = 480

# --------------------------------------------------------------------------------------
# Custom CSS
# --------------------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#0B0F14; --surface:#121821; --surface-2:#1A222D;
  --border:#1F2937; --border-strong:#334155;
  --text:#E5E7EB; --muted:#9CA3AF; --accent:#3B82F6; --accent-2:#60A5FA; --accent-soft:#93C5FD;
  --ok:#34d399; --err:#f87171; --radius:14px; color-scheme:dark;
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
}
html,body,.stApp,[data-testid="stApp"]{
  color:var(--text); font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
}
.stApp,[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(900px circle at 50% -10%, rgba(59,130,246,0.22), transparent 60%),
    linear-gradient(180deg, #08111F 0%, #0A1A30 50%, #0B1F3A 100%) !important;
  background-attachment:fixed;
  background-repeat:no-repeat;
}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none !important}
[data-testid="stHeader"]{background:transparent !important}
.block-container{max-width:1180px;padding-top:1.1rem;padding-bottom:5rem}
.stApp p,.stApp li,.stApp label,.stApp span{font-family:inherit}
.stApp [data-testid="stIconMaterial"],.stApp .material-symbols-rounded{
  font-family:"Material Symbols Rounded" !important;font-feature-settings:"liga"}
.stApp .material-icons{font-family:"Material Icons" !important;font-feature-settings:"liga"}
.stApp .material-symbols-outlined{font-family:"Material Symbols Outlined" !important;font-feature-settings:"liga"}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{color:var(--text);letter-spacing:-.01em}
::selection{background:rgba(59,130,246,.4)}

/* Sidebar */
[data-testid="stSidebar"]{background:#0E141B !important;border-right:1px solid var(--border)}
[data-testid="stSidebar"] .block-container,[data-testid="stSidebarContent"]{padding-top:1.2rem}
.sb-label{font-size:.78rem;font-weight:600;color:var(--muted);margin:1.3rem 0 .5rem}
.sb-brand{display:flex;align-items:center;gap:.65rem;font-weight:600}
.sb-text{color:var(--muted);font-size:.85rem;line-height:1.55}
.kv{display:flex;justify-content:space-between;gap:1rem;font-size:.85rem;padding:.4rem 0;border-bottom:1px solid var(--border)}
.kv span:first-child{color:var(--muted)} .kv span:last-child{text-align:right;max-width:62%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sb-foot{margin-top:2.2rem;padding-top:1rem;border-top:1px solid var(--border);color:var(--muted);font-size:.76rem}

/* Top bar */
.topbar{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.8rem 1.1rem;
  border:1px solid var(--border);background:var(--surface);backdrop-filter:blur(14px);border-radius:16px;margin-bottom:1.2rem}
.brand{display:flex;align-items:center;gap:.8rem}
.logo{width:36px;height:36px;border-radius:10px;display:grid;place-items:center;color:#fff;
  background:linear-gradient(135deg,#2563EB,#3B82F6)}
.brand-name{font-weight:650;font-size:1.02rem;line-height:1.2}
.brand-sub{color:var(--muted);font-size:.82rem}
.top-right{display:flex;align-items:center;gap:.7rem;flex-wrap:wrap;justify-content:flex-end}
.pill{display:inline-flex;align-items:center;gap:.45rem;font-size:.78rem;padding:.28rem .7rem;border-radius:999px;
  border:1px solid var(--border);color:var(--text);background:var(--surface)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 0 rgba(52,211,153,.5);animation:pulse 2.4s infinite}
.dot.off{background:var(--err);animation:none}
.tag{font-size:.72rem;letter-spacing:.06em;color:#CBD5E1;border:1px solid #334155;
  background:rgba(148,163,184,.08);padding:.28rem .7rem;border-radius:8px;font-weight:500}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.45)}70%{box-shadow:0 0 0 7px rgba(52,211,153,0)}100%{box-shadow:0 0 0 0 rgba(52,211,153,0)}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes blink{0%,80%,100%{opacity:.25}40%{opacity:1}}

/* Hero */
.hero{text-align:center;padding:2.4rem 0 1.4rem}
.hero h1{font-size:2.55rem;font-weight:700;letter-spacing:-.03em;line-height:1.12;margin:0 auto .8rem;max-width:760px}
.hero p{color:var(--muted);font-size:1.06rem;margin:0 auto;max-width:600px;line-height:1.6}

/* Cards */
.card{border:1px solid var(--border);background:var(--surface);border-radius:var(--radius);padding:1.15rem 1.25rem;
  transition:border-color .2s ease}
.card:hover{border-color:var(--border-strong)}
.card h3{font-size:1.05rem;font-weight:600;margin:0}
.section-title{font-size:1.15rem;font-weight:600;margin:1.8rem 0 .8rem}
.muted{color:var(--muted)}
.grid-5{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:12px}
.grid-3{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px}
.metric .k{font-size:.78rem;color:var(--muted);margin-bottom:.35rem}
.metric .v{font-size:1rem;font-weight:600;display:flex;align-items:center;gap:.5rem}
.title-card .t{font-size:1.55rem;font-weight:650;letter-spacing:-.02em;line-height:1.3;margin:.2rem 0 .35rem}
.title-card .s{color:var(--muted);font-size:.85rem;word-break:break-all}
.label{font-size:.8rem;color:var(--accent-soft);font-weight:500}

/* Summary */
.summary{line-height:1.75;font-size:1rem;color:var(--text);max-width:880px}
.summary p{margin:.2rem 0 .9rem} .summary ul{padding-left:1.2rem;margin:.2rem 0 .9rem} .summary li{margin:.3rem 0}
.summary h4{margin:1rem 0 .4rem;font-size:1rem}
.summary code,.itxt code{font-family:var(--mono);font-size:.85em;background:var(--surface-2);padding:1px 6px;border-radius:5px}
.summary-actions-spacer{height:1rem}
.card-head{display:flex;align-items:center;gap:.7rem;margin-bottom:.9rem}

/* Insight cards */
.ico{width:32px;height:32px;border-radius:9px;display:grid;place-items:center;background:rgba(59,130,246,.14);color:var(--accent-soft);flex:0 0 auto}
.count{margin-left:auto;font-size:.74rem;color:var(--muted);border:1px solid var(--border);border-radius:999px;padding:1px 9px}
.ins-list{list-style:none;padding:0;margin:.9rem 0 0;display:flex;flex-direction:column;gap:.5rem}
.ins-list li{display:flex;gap:.7rem;padding:.65rem .75rem;border-radius:10px;background:rgba(255,255,255,.025);
  border:1px solid transparent;font-size:.92rem;line-height:1.5;transition:border-color .15s ease,background .15s ease}
.ins-list li:hover{border-color:var(--border);background:rgba(255,255,255,.04)}
.bullet{flex:0 0 auto;width:6px;height:6px;border-radius:50%;background:#64748B;margin-top:.55rem}
.meta{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.4rem}
.chip{font-size:.74rem;color:var(--accent-soft);background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.25);border-radius:6px;padding:1px 8px}
.empty-note{color:var(--muted);font-size:.88rem;padding:.9rem .2rem 0}

/* Empty state */
.empty{text-align:center;padding:2.4rem 1rem;border:1px dashed var(--border-strong);border-radius:18px;margin:1.8rem 0 1.2rem;background:var(--surface)}
.empty .orb{width:64px;height:64px;margin:0 auto 1rem;border-radius:18px;display:grid;place-items:center;color:var(--accent-soft);
  background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.3)}
.empty h3{font-size:1.25rem;margin:0 0 .35rem;font-weight:600} .empty p{margin:0;color:var(--muted)}
.how h4{margin:.7rem 0 .3rem;font-size:.98rem;font-weight:600} .how p{margin:0;color:var(--muted);font-size:.88rem;line-height:1.55}

/* Processing */
.proc-head{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;margin-bottom:.9rem}
.proc-title{font-size:1.3rem;font-weight:650;letter-spacing:-.02em}
.proc-src{color:var(--muted);font-size:.84rem;word-break:break-all;margin-top:.15rem}
.proc-count{font-size:.8rem;color:var(--muted);white-space:nowrap}
.bar{height:4px;background:rgba(255,255,255,.07);border-radius:99px;overflow:hidden;margin-bottom:.8rem}
.bar>span{display:block;height:100%;background:linear-gradient(90deg,#2563EB,#3B82F6);transition:width .5s ease}
.stage{display:flex;align-items:center;gap:.85rem;padding:.55rem .6rem;border-radius:10px;border:1px solid transparent}
.stage .mark{width:22px;height:22px;border-radius:50%;flex:0 0 auto;display:grid;place-items:center;border:1px solid var(--border-strong);color:transparent}
.stage .name{font-size:.94rem;font-weight:500} .stage .desc{font-size:.8rem;color:var(--muted)}
.stage .dur{margin-left:auto;font-family:var(--mono);font-size:.76rem;color:var(--muted)}
.stage.pending .name{color:var(--muted)}
.stage.done .mark{background:rgba(52,211,153,.14);border-color:rgba(52,211,153,.4);color:var(--ok)}
.stage.active{background:rgba(59,130,246,.08);border-color:rgba(59,130,246,.25)}
.stage.active .mark{border:2px solid rgba(59,130,246,.25);border-top-color:#60A5FA;animation:spin .8s linear infinite}
.stage.failed{background:rgba(248,113,113,.07);border-color:rgba(248,113,113,.3)}
.stage.failed .mark{border-color:var(--err);color:var(--err);font-size:.75rem;font-weight:700}
.proc-note{margin-top:.8rem;color:var(--muted);font-size:.78rem}

/* Errors */
.err-card{border:1px solid rgba(248,113,113,.35);background:rgba(248,113,113,.06);border-radius:var(--radius);padding:1.05rem 1.2rem;margin:.4rem 0 1rem}
.err-card .h{display:flex;align-items:center;gap:.6rem;font-weight:600;color:#fecaca}
.err-card .b{margin-top:.5rem;color:#e5e7eb;font-size:.92rem;line-height:1.55}
.err-card .b b{color:#fff}
.inline-err{color:#fecaca;background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.3);border-radius:10px;padding:.6rem .8rem;font-size:.88rem;margin-top:.7rem}
.note{color:var(--muted);font-size:.78rem;margin-top:.6rem;text-align:center}
.formats{display:flex;gap:.4rem;flex-wrap:wrap;justify-content:center;margin-top:.9rem}

/* Transcript */
.tline{display:flex;gap:.9rem;padding:.75rem .4rem;border-bottom:1px solid rgba(255,255,255,.05);line-height:1.65;font-size:.94rem}
.tno{font-family:var(--mono);color:var(--muted);font-size:.74rem;min-width:2rem;padding-top:.25rem}
.ts{font-family:var(--mono);color:var(--accent-soft);background:rgba(59,130,246,.12);border-radius:6px;padding:0 7px;font-size:.76rem;height:fit-content;margin-top:.2rem}
mark{background:rgba(59,130,246,.4);color:#fff;border-radius:3px;padding:0 2px}
.stat{color:var(--muted);font-size:.82rem;margin:.5rem 0}

/* Chat */
.thinking{display:flex;align-items:center;gap:.35rem;color:var(--muted);font-size:.9rem}
.thinking i{width:6px;height:6px;border-radius:50%;background:var(--accent-soft);animation:blink 1.2s infinite;display:inline-block}
.thinking i:nth-child(2){animation-delay:.2s} .thinking i:nth-child(3){animation-delay:.4s}
[data-testid="stChatMessage"]{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:.9rem 1rem}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){background:var(--surface);border-color:var(--border)}
[data-testid="stChatMessageAvatarUser"],[data-testid="chatAvatarIcon-user"]{background:#334155 !important}
[data-testid="stChatMessageAvatarAssistant"],[data-testid="chatAvatarIcon-assistant"]{background:#1A222D !important;color:var(--accent-soft) !important}
[data-testid="stChatInput"]{border-radius:14px}
[data-testid="stChatInput"] textarea{color:var(--text)}

/* Widgets */
[data-testid="stForm"]{border:1px solid var(--border) !important;background:var(--surface);border-radius:18px;padding:1.3rem 1.4rem}
div[data-baseweb="input"],div[data-baseweb="base-input"],div[data-baseweb="select"]>div,[data-testid="stChatInput"]>div{
  background:#1A222D !important;border-color:var(--border) !important;border-radius:10px !important}
div[data-baseweb="input"]:focus-within,div[data-baseweb="select"]>div:focus-within{
  border-color:#64748B !important;box-shadow:0 0 0 3px rgba(148,163,184,.14) !important}
input,textarea{color:var(--text) !important}
[data-baseweb="popover"] ul,[data-baseweb="menu"]{background:#121821 !important}
[data-testid="stFileUploader"],[data-testid="stFileUploaderDropzone"]{box-sizing:border-box;width:100%;min-width:0;max-width:100%}
[data-testid="stFileUploaderDropzone"]{background:#121821;border:1.5px dashed #334155;border-radius:14px;padding:1.6rem;transition:border-color .2s,background .2s}
[data-testid="stFileUploaderDropzone"]:hover{border-color:#64748B;background:#1A222D}
.stButton>button,[data-testid="stDownloadButton"] button,[data-testid="stBaseButton-secondaryFormSubmit"]{
  display:flex;align-items:center;justify-content:center;width:100%;height:auto;min-height:2.5rem;padding:.55rem .65rem;
  background:var(--surface);color:var(--text);border:1px solid var(--border-strong);border-radius:10px;font-weight:500;
  line-height:1.25;text-align:center;white-space:normal;overflow-wrap:anywhere;transition:all .15s ease}
.stButton>button p,[data-testid="stDownloadButton"] button p,
[data-testid="stBaseButton-secondaryFormSubmit"] p{margin:0;line-height:inherit;white-space:normal;overflow-wrap:anywhere}
.stButton>button:hover,[data-testid="stDownloadButton"] button:hover{border-color:#64748B;background:#1A222D;color:#fff}
.stButton>button:active{transform:scale(.985)}
button[kind="primary"],button[kind="primaryFormSubmit"],[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"]{
  height:auto;padding:.65rem .75rem;white-space:normal;overflow-wrap:anywhere;text-align:center;line-height:1.25;
  background:linear-gradient(135deg,#2563EB,#3B82F6) !important;border:none !important;color:#fff !important;font-weight:600 !important;
  min-height:2.9rem;border-radius:11px !important;box-shadow:0 6px 18px rgba(148,163,184,.16);transition:filter .15s,box-shadow .15s,transform .1s}
button[kind="primary"]:hover,button[kind="primaryFormSubmit"]:hover,[data-testid="stBaseButton-primary"]:hover,[data-testid="stBaseButton-primaryFormSubmit"]:hover{
  filter:brightness(1.1);box-shadow:0 8px 24px rgba(148,163,184,.22)}
div[role="radiogroup"]{gap:.5rem;justify-content:center;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:.3rem;width:fit-content;margin:0 auto}
div[role="radiogroup"]>label{padding:.4rem 1.1rem;border-radius:9px;cursor:pointer;margin:0 !important;transition:background .15s}
div[role="radiogroup"]>label>div:first-child{display:none}
div[role="radiogroup"]>label:has(input:checked){background:rgba(99,102,241,.22)}
div[role="radiogroup"]>label:hover{background:rgba(255,255,255,.05)}
[data-testid="stExpander"]{border:1px solid var(--border);border-radius:12px;background:var(--surface)}
button[data-baseweb="tab"]{color:var(--muted);font-weight:500}
button[data-baseweb="tab"][aria-selected="true"]{color:#fff}
[data-baseweb="tab-highlight"]{background:#6366f1 !important}
[data-baseweb="tab-border"]{background:var(--border) !important}
@media (prefers-reduced-motion:reduce){*{animation:none !important;transition:none !important}}
@media (max-width:640px){.hero h1{font-size:1.8rem}.topbar{flex-direction:column;align-items:flex-start}}
</style>
"""

# --------------------------------------------------------------------------------------
# Icons (inline SVG, Lucide-style)
# --------------------------------------------------------------------------------------

_ICON_PATHS = {
    "logo": '<rect x="3" y="4" width="18" height="16" rx="3"/><path d="m10 9 5 3-5 3z"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "flag": '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><path d="M4 22v-7"/>',
    "tasks": '<path d="m3 17 2 2 4-4"/><path d="m3 7 2 2 4-4"/><path d="M13 6h8"/><path d="M13 12h8"/><path d="M13 18h8"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "file": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "wave": '<path d="M2 10v3"/><path d="M6 6v11"/><path d="M10 3v18"/><path d="M14 8v7"/><path d="M18 5v13"/><path d="M22 10v3"/>',
    "sparkles": '<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M19 3v4"/><path d="M21 5h-4"/>',
    "chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "copy": '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
}


def icon(name: str, size: int = 18) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>'
    )


# --------------------------------------------------------------------------------------
# Text / HTML helpers
# --------------------------------------------------------------------------------------

BULLET_RE = re.compile(r"^\s*(?:[-*•▪◦]|\d+[.)])\s+")
EMPTY_RE = re.compile(
    r"^(none|n/?a|nothing|no\b.{0,80}(found|mentioned|identified|discussed|detected|explicit|stated|made|raised|present)).*$",
    re.I,
)
ABSENT_VALUES = {"", "n/a", "na", "none", "null", "unknown", "unassigned", "not specified", "not mentioned", "tbd", "-"}
OWNER_RE = re.compile(r"(?:owner|assignee|assigned to|responsible)\s*[:\-–]\s*([^,;|()\n]+)", re.I)
DUE_RE = re.compile(r"(?:deadline|due(?:\s+date)?)\s*[:\-–]\s*([^,;|()\n]+)", re.I)
TS_RE = re.compile(r"^\s*[\[(]?(\d{1,2}:\d{2}(?::\d{2})?)[\])]?\s*[-–:]?\s*")


def esc(text: Any) -> str:
    """HTML-escape text (also neutralises `$`, which Streamlit's markdown treats as math)."""
    s = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    return s.replace("$", "&#36;")


def inline_md(text: Any) -> str:
    """Escape, then support **bold** and `code` from LLM output."""
    s = esc(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def md(markup: str) -> None:
    """Render an HTML snippet. Strips indentation/blank lines so Markdown never splits the block."""
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def coerce_text(obj: Any) -> str:
    """Backend outputs are usually strings; tolerate dict / message-like returns too."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("answer", "result", "output_text", "output", "text", "content"):
            if isinstance(obj.get(key), str):
                return obj[key]
    content = getattr(obj, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(obj, (list, tuple)):
        return "\n\n".join(coerce_text(o) for o in obj)
    return str(obj)


def render_rich_text(text: str) -> str:
    """Tiny Markdown subset (paragraphs, bullets, headings, bold) -> safe HTML."""
    blocks: list[str] = []
    para: list[str] = []
    bullets: list[str] = []

    def flush() -> None:
        if para:
            blocks.append(f"<p>{inline_md(' '.join(para))}</p>")
            para.clear()
        if bullets:
            blocks.append("<ul>" + "".join(f"<li>{inline_md(b)}</li>" for b in bullets) + "</ul>")
            bullets.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped.startswith("#"):
            flush()
            blocks.append(f"<h4>{inline_md(stripped.lstrip('#').strip())}</h4>")
        elif BULLET_RE.match(line):
            if para:
                flush()
            bullets.append(BULLET_RE.sub("", line, count=1).strip())
        else:
            if bullets:
                flush()
            para.append(stripped)
    flush()
    return "".join(blocks)


def to_items(value: Any) -> list[Any]:
    """Normalise extractor output (string / list / JSON / dict) into a list of items."""
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if raw[:1] in ("[", "{"):
            try:
                value = json.loads(raw)
            except ValueError:
                pass
        if isinstance(value, str):
            items: list[str] = []
            for line in raw.splitlines():
                t = line.strip()
                if not t or t.startswith("#"):
                    continue
                is_bullet = bool(BULLET_RE.match(t))
                t = BULLET_RE.sub("", t, count=1).replace("**", "").strip()
                if not is_bullet and t.endswith(":") and len(t) < 60:
                    continue  # section heading such as "Action Items:"
                if t:
                    items.append(t)
            if len(items) == 1 and EMPTY_RE.match(items[0]):
                return []
            return items
    if isinstance(value, dict):
        for key in ("items", "action_items", "decisions", "key_decisions", "questions", "open_questions"):
            if isinstance(value.get(key), (list, tuple)):
                return to_items(list(value[key]))
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for v in value:
            if isinstance(v, str):
                t = BULLET_RE.sub("", v.strip(), count=1).replace("**", "").strip()
                if t:
                    out.append(t)
            elif v is not None:
                out.append(v)
        return out
    return [str(value)]


def item_text(item: Any) -> str:
    if isinstance(item, dict):
        low = {str(k).lower(): v for k, v in item.items()}
        for key in ("decision", "question", "text", "task", "action", "description", "item", "title"):
            if low.get(key):
                return str(low[key])
        return ", ".join(str(v) for v in item.values() if v)
    return str(item)


def _clean_value(v: Any) -> Optional[str]:
    s = str(v).strip(" .") if v is not None else ""
    return None if s.lower() in ABSENT_VALUES else s


def parse_action_item(item: Any) -> dict[str, Optional[str]]:
    """Split an action item into task / owner / deadline - only using data actually present."""
    if isinstance(item, dict):
        low = {str(k).lower(): v for k, v in item.items()}

        def first(keys: tuple[str, ...]) -> Any:
            return next((low[k] for k in keys if low.get(k)), None)

        return {
            "task": _clean_value(first(("task", "action", "action_item", "description", "item", "title", "text"))) or item_text(item),
            "owner": _clean_value(first(("owner", "assignee", "assigned_to", "responsible", "who"))),
            "deadline": _clean_value(first(("deadline", "due", "due_date", "when", "by"))),
        }
    text = str(item)
    owner_m, due_m = OWNER_RE.search(text), DUE_RE.search(text)
    owner = _clean_value(owner_m.group(1)) if owner_m else None
    deadline = _clean_value(due_m.group(1)) if due_m else None
    task = DUE_RE.sub("", OWNER_RE.sub("", text))
    task = re.sub(r"^\s*task\s*[:\-–]\s*", "", task, flags=re.I)
    task = re.sub(r"\(\s*\)", "", task)
    task = re.sub(r"[\s,;|\-–(]+$", "", task).strip() or text
    return {"task": task, "owner": owner, "deadline": deadline}


@st.cache_data(show_spinner=False)
def split_transcript(text: str) -> list[tuple[Optional[str], str]]:
    """Break a raw transcript into readable segments, keeping leading timestamps if present."""
    text = text.strip()
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) <= 1:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) > 3:
            parts = lines
        else:
            sentences = re.split(r"(?<=[.!?।])\s+", text)
            parts = [" ".join(sentences[i : i + 4]) for i in range(0, len(sentences), 4)]
    segments: list[tuple[Optional[str], str]] = []
    for p in parts:
        p = " ".join(p.split())
        m = TS_RE.match(p)
        segments.append((m.group(1), p[m.end():]) if m else (None, p))
    return [s for s in segments if s[1]]


def slugify(text: str, default: str = "video-analysis") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or default


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


def columns_bottom(spec: Any) -> list[Any]:
    """Columns aligned to the bottom (falls back on older Streamlit versions)."""
    try:
        return st.columns(spec, vertical_alignment="bottom")
    except TypeError:
        return st.columns(spec)


def copy_button(text: str, label: str, height: int = 54, button_height: Optional[int] = None) -> None:
    """Clipboard button (Streamlit has no native clipboard API)."""
    payload = json.dumps(text).replace("<", "\\u003c")
    label_js = json.dumps(label)
    button_size = (
        f"height:{button_height}px;box-sizing:border-box;"
        if button_height is not None
        else "height:auto;"
    )
    html_doc = f"""<!doctype html><html><head><style>
    html,body{{margin:0;background:transparent;font-family:Inter,-apple-system,Segoe UI,Roboto,sans-serif}}
    button{{width:100%;{button_size}min-height:38px;padding:6px 8px;display:flex;align-items:center;justify-content:center;gap:8px;cursor:pointer;
      color:#e7e9ee;background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.16);border-radius:10px;
      font-size:14px;font-weight:500;line-height:1.2;text-align:center;white-space:normal;overflow-wrap:anywhere;transition:all .15s ease}}
    button svg{{flex:0 0 auto}}
    button:hover{{border-color:#64748B;background:#1A222D}}
    button.ok{{border-color:rgba(52,211,153,.6);color:#34d399}}
    </style></head><body>
    <button id="b"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
      stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS["copy"]}</svg><span id="t"></span></button>
    <script>
    const text={payload}; const label={label_js};
    const b=document.getElementById('b'), t=document.getElementById('t'); t.textContent=label;
    function fallback(){{const a=document.createElement('textarea');a.value=text;a.style.position='fixed';a.style.opacity='0';
      document.body.appendChild(a);a.select();try{{document.execCommand('copy');}}catch(e){{}}document.body.removeChild(a);}}
    b.addEventListener('click',async()=>{{
      try{{await navigator.clipboard.writeText(text);}}catch(e){{fallback();}}
      t.textContent='Copied';b.classList.add('ok');
      setTimeout(()=>{{t.textContent=label;b.classList.remove('ok');}},1600);
    }});
    </script></body></html>"""
    try:
        if hasattr(st, "iframe"):  # newer Streamlit
            st.iframe(html_doc, height=height)
        else:  # older Streamlit
            import streamlit.components.v1 as components

            components.html(html_doc, height=height)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Clipboard component unavailable")


# --------------------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------------------


def init_state() -> None:
    defaults: dict[str, Any] = {
        "result": None,
        "job": None,
        "error": None,
        "chat": [],
        "pending_q": None,
        "uploader_nonce": 0,
        "just_finished": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def cb_clear_results() -> None:
    st.session_state.result = None
    st.session_state.job = None
    st.session_state.error = None
    st.session_state.pending_q = None
    st.session_state.chat = []


def cb_new_analysis() -> None:
    cb_clear_results()
    st.session_state.uploader_nonce += 1
    for key in ("src_url", "src_lang", "src_kind", "t_query"):
        st.session_state.pop(key, None)


def cb_dismiss_error() -> None:
    st.session_state.error = None


def cb_ask(question: str) -> None:
    st.session_state.pending_q = question


def cb_clear_chat() -> None:
    st.session_state.chat = []


# --------------------------------------------------------------------------------------
# Backend (cached) + error handling
# --------------------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading AI engine...")
def load_backend() -> SimpleNamespace:
    """Import the existing backend once per server process (heavy model imports happen here)."""
    from dotenv import load_dotenv

    load_dotenv()
    from core.extractor import extract_action_items, extract_key_decisions, extract_questions
    from core.rag_engine import ask_question, build_rag_chain
    from core.summarize import generate_title, summarize
    from core.transcriber import transcribe_all
    from utils.audio_processor import process_input

    return SimpleNamespace(
        process_input=process_input,
        transcribe_all=transcribe_all,
        generate_title=generate_title,
        summarize=summarize,
        extract_action_items=extract_action_items,
        extract_key_decisions=extract_key_decisions,
        extract_questions=extract_questions,
        build_rag_chain=build_rag_chain,
        ask_question=ask_question,
    )


def get_backend() -> tuple[Optional[SimpleNamespace], Optional["ErrorInfo"]]:
    try:
        return load_backend(), None
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a friendly error
        logger.exception("Backend failed to load")
        return None, classify_error("startup", exc)


def language_kwargs(fn: Any, code: Optional[str]) -> dict[str, str]:
    """Forward the language only if the backend function explicitly accepts it."""
    if not code:
        return {}
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    for name in ("language", "lang"):
        if name in params:
            return {name: code}
    return {}


def backend_supports_language(backend: SimpleNamespace) -> bool:
    return bool(language_kwargs(backend.transcribe_all, "en") or language_kwargs(backend.process_input, "en"))


@dataclass
class ErrorInfo:
    title: str
    hint: str
    detail: str
    stage: str


class UserFacingError(Exception):
    """Raised for failures where we already know the friendly explanation."""

    def __init__(self, title: str, hint: str):
        super().__init__(title)
        self.title, self.hint = title, hint


class AnalysisError(Exception):
    def __init__(self, info: ErrorInfo):
        super().__init__(info.title)
        self.info = info


_ERROR_RULES: list[tuple[tuple[str, ...], str, str]] = [
    (("api key", "api_key", "apikey", "authentication", "unauthorized", "invalid_api", "401", "not set", "credentials"),
     "Missing or invalid API key",
     "Check that the required API keys are set in your .env file, then restart the app."),
    (("rate limit", "429", "quota", "too many requests"),
     "Rate limit reached", "The AI provider is throttling requests. Wait a moment and try again."),
    (("private video", "video unavailable", "sign in to confirm", "http error 403", "http error 404", "unable to download",
      "yt_dlp", "yt-dlp", "downloaderror"),
     "Download failed", "The video could not be downloaded. It may be private, region-locked, or removed."),
    (("ffmpeg", "ffprobe"),
     "Audio extraction failed", "FFmpeg is required to read media files. Make sure it is installed and on your PATH."),
    (("unsupported url", "unsupported format", "invalid data found", "moov atom", "could not find codec"),
     "Unsupported or corrupted media", "This source could not be decoded. Try a different file or a standard YouTube link."),
    (("connection", "timed out", "timeout", "network", "name resolution", "getaddrinfo", "ssl", "max retries", "urlopen"),
     "Network error", "A network request failed. Check your connection and try again."),
]
_STAGE_FALLBACK = {
    "fetch": ("Could not fetch or decode the media", "Verify the link or file, then try again."),
    "transcribe": ("Transcription failed", "The audio could not be converted to text. Try a clearer or shorter source."),
    "summary": ("Summary generation failed", "The language model call failed. Check your API key and connection."),
    "insights": ("Insight extraction failed", "The language model call failed. Check your API key and connection."),
    "rag": ("Knowledge base could not be built", "The transcript could not be indexed for Q&A."),
    "finalize": ("Could not assemble results", "Try running the analysis again."),
    "query": ("The question could not be answered", "Retrieval failed. Try rephrasing the question or ask again."),
    "startup": ("The AI engine could not start",
                "Check that all dependencies are installed (pip install -r Requirements.txt) and your .env is configured."),
}


def classify_error(stage: str, exc: BaseException) -> ErrorInfo:
    detail = f"{type(exc).__name__}: {str(exc)[:300]}"
    if isinstance(exc, UserFacingError):
        return ErrorInfo(exc.title, exc.hint, detail, stage)
    haystack = f"{type(exc).__name__} {exc}".lower()
    for keywords, title, hint in _ERROR_RULES:
        if any(k in haystack for k in keywords):
            return ErrorInfo(title, hint, detail, stage)
    title, hint = _STAGE_FALLBACK.get(stage, ("Unexpected error", "Try again."))
    return ErrorInfo(title, hint, detail, stage)


# --------------------------------------------------------------------------------------
# Pipeline orchestration (real stages, measured timings)
# --------------------------------------------------------------------------------------


class StageTracker:
    """Renders the live pipeline panel and records per-stage timings."""

    def __init__(self, placeholder: Any, source_label: str):
        self.placeholder = placeholder
        self.source_label = source_label
        self.status = {g: "pending" for g in GROUPS}
        self.durations: dict[str, float] = {}
        self.render()

    def render(self) -> None:
        rows, done = [], 0
        for idx, (label, group, desc) in enumerate(STAGES):
            state = self.status[group]
            done += state == "done"
            mark = icon("check", 13) if state == "done" else ("!" if state == "failed" else "")
            dur = ""
            if state == "done" and idx == LAST_IN_GROUP[group]:
                dur = f'<span class="dur">{self.durations[group]:.1f}s</span>'
            rows.append(
                f'<div class="stage {state}"><div class="mark">{mark}</div>'
                f'<div><div class="name">{label}</div><div class="desc">{desc}</div></div>{dur}</div>'
            )
        pct = done / len(STAGES) * 100
        with self.placeholder.container():
            md(
                f"""<div class="card">
                <div class="proc-head"><div><div class="proc-title">Analyzing your video</div>
                <div class="proc-src">{esc(self.source_label)}</div></div>
                <div class="proc-count">{done} of {len(STAGES)} stages complete</div></div>
                <div class="bar"><span style="width:{pct:.0f}%"></span></div>
                {''.join(rows)}
                <div class="proc-note">Progress reflects completed backend stages. Durations are measured, not estimated.</div>
                </div>"""
            )

    @contextmanager
    def stage(self, group: str) -> Iterator[None]:
        self.status[group] = "active"
        self.render()
        started = time.perf_counter()
        try:
            yield
        except AnalysisError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline stage '%s' failed", group)
            self.status[group] = "failed"
            self.render()
            raise AnalysisError(classify_error(group, exc)) from exc
        self.durations[group] = time.perf_counter() - started
        self.status[group] = "done"
        self.render()


def _save_upload(job: dict[str, Any]) -> str:
    suffix = Path(job["filename"]).suffix or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(job["bytes"])
        return tmp.name


def execute_pipeline(backend: SimpleNamespace, job: dict[str, Any], tracker: StageTracker) -> dict[str, Any]:
    """Same call sequence as main.run_pipeline(), split into reportable stages."""
    started = time.perf_counter()
    source, tmp_path = job["source"], None
    code = LANGUAGES.get(job["language"])
    try:
        if job["kind"] == SOURCE_FILE:
            tmp_path = source = _save_upload(job)

        with tracker.stage("fetch"):
            chunks = backend.process_input(source, **language_kwargs(backend.process_input, code))

        with tracker.stage("transcribe"):
            transcript = coerce_text(backend.transcribe_all(chunks, **language_kwargs(backend.transcribe_all, code)))
            if not transcript.strip():
                raise UserFacingError(
                    "No speech detected",
                    "The transcription came back empty. Try a source with clear spoken audio.",
                )

        with tracker.stage("summary"):
            title = coerce_text(backend.generate_title(transcript)).strip().strip('"').strip()
            summary = coerce_text(backend.summarize(transcript)).strip()

        with tracker.stage("insights"):
            action_items = backend.extract_action_items(transcript)
            decisions = backend.extract_key_decisions(transcript)
            questions = backend.extract_questions(transcript)

        with tracker.stage("rag"):
            rag_chain = backend.build_rag_chain(transcript)

        with tracker.stage("finalize"):
            result = {
                "title": title or "Untitled video",
                "summary": summary,
                "transcript": transcript,
                "action_items": to_items(action_items),
                "key_decisions": to_items(decisions),
                "open_questions": to_items(questions),
                "rag_chain": rag_chain,
                "source_label": job["filename"] if job["kind"] == SOURCE_FILE else job["source"],
                "source_kind": job["kind"],
                "language": job["language"],
                "language_applied": bool(
                    language_kwargs(backend.transcribe_all, code) or language_kwargs(backend.process_input, code)
                ),
                "words": len(transcript.split()),
                "elapsed": time.perf_counter() - started,
            }
        return result
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning("Could not remove temp file %s", tmp_path)


def answer_question(backend: SimpleNamespace, rag_chain: Any, question: str) -> str:
    try:
        answer = coerce_text(backend.ask_question(rag_chain, question)).strip()
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG query failed")
        raise AnalysisError(classify_error("query", exc)) from exc
    if not answer:
        raise AnalysisError(ErrorInfo("Empty answer", "The model returned no answer. Try rephrasing.", "empty response", "query"))
    return answer


def build_report_md(result: dict[str, Any]) -> str:
    def block(title: str, items: list[Any], action: bool = False) -> str:
        if not items:
            return f"## {title}\n\nNone identified.\n"
        lines = []
        for it in items:
            if action:
                p = parse_action_item(it)
                extra = "".join(f" ({k}: {v})" for k, v in (("Owner", p["owner"]), ("Deadline", p["deadline"])) if v)
                lines.append(f"- {p['task']}{extra}")
            else:
                lines.append(f"- {item_text(it)}")
        return f"## {title}\n\n" + "\n".join(lines) + "\n"

    return (
        f"# {result['title']}\n\nSource: {result['source_label']}\n\n## Executive Summary\n\n{result['summary']}\n\n"
        + block("Key Decisions", result["key_decisions"])
        + "\n" + block("Action Items", result["action_items"], action=True)
        + "\n" + block("Questions", result["open_questions"])
        + f"\n## Transcript\n\n{result['transcript']}\n"
    )


def build_export_markdown(result: dict[str, Any], content: str) -> str:
    sections = [f"# {result['title']}"]
    if content in ("Summary", "Both"):
        sections.append(f"## Summary\n\n{result['summary'] or 'No summary was generated.'}")
    if content in ("Transcript", "Both"):
        segments = split_transcript(result["transcript"])
        if any(timestamp for timestamp, _ in segments):
            transcript = "\n\n".join(
                f"**{timestamp}** {text}" if timestamp else text
                for timestamp, text in segments
            )
        else:
            transcript = result["transcript"]
        sections.append(f"## Transcript\n\n{transcript}")
    return "\n\n".join(sections) + "\n"


def _pdf_font_name() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = [
        fonts_dir / "Nirmala.ttf",
        fonts_dir / "segoeui.ttf",
        fonts_dir / "arial.ttf",
        Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    ]
    for font_path in candidates:
        if not font_path.is_file():
            continue
        try:
            if "ClipIQUnicode" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("ClipIQUnicode", str(font_path)))
            return "ClipIQUnicode"
        except Exception:
            logger.warning("Could not register PDF font %s", font_path, exc_info=True)
    logger.warning("No Unicode TrueType font found for PDF export; unsupported characters will be replaced")
    return "Helvetica"


def build_export_pdf(markdown: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    buffer = io.BytesIO()
    font_name = _pdf_font_name()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ClipIQTitle", parent=styles["Title"], fontName=font_name,
        fontSize=20, leading=25, alignment=TA_LEFT, textColor=colors.HexColor("#111827"),
        spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "ClipIQHeading", parent=styles["Heading2"], fontName=font_name,
        fontSize=14, leading=19, textColor=colors.HexColor("#1F2937"),
        spaceBefore=12, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ClipIQBody", parent=styles["BodyText"], fontName=font_name,
        fontSize=10, leading=15, textColor=colors.HexColor("#111827"),
        spaceAfter=7,
    )
    story = []
    for line in markdown.splitlines():
        cleaned = "".join(char for char in line if char in "\t" or ord(char) >= 32)
        if font_name == "Helvetica":
            cleaned = cleaned.encode("cp1252", errors="replace").decode("cp1252")
        text = html_escape(cleaned, quote=False)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"`(.+?)`", r'<font name="Courier">\1</font>', text)
        if cleaned.startswith("# "):
            story.append(Paragraph(text[2:], title_style))
        elif cleaned.startswith("## "):
            story.append(Paragraph(text[3:], heading_style))
        elif BULLET_RE.match(cleaned):
            story.append(Paragraph(f"- {BULLET_RE.sub('', text, count=1)}", body_style))
        elif cleaned.strip():
            story.append(Paragraph(text, body_style))
        else:
            story.append(Spacer(1, 0.08 * inch))
    SimpleDocTemplate(
        buffer, pagesize=letter, rightMargin=0.7 * inch, leftMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    ).build(story)
    return buffer.getvalue()


# --------------------------------------------------------------------------------------
# UI components
# --------------------------------------------------------------------------------------


def render_topbar(engine_ready: bool) -> None:
    status = "AI Engine Ready" if engine_ready else "AI Engine Unavailable"
    md(
        f"""<div class="topbar">
        <div class="brand"><div class="logo">{icon("logo", 20)}</div>
        <div><div class="brand-name">{APP_NAME}</div><div class="brand-sub">{TAGLINE}</div></div></div>
        <div class="top-right"><span class="pill"><span class="dot {'' if engine_ready else 'off'}"></span>{status}</span>
        <span class="tag">AI VIDEO INTELLIGENCE · v{APP_VERSION}</span></div></div>"""
    )


def render_sidebar(backend: Optional[SimpleNamespace]) -> None:
    result = st.session_state.result
    source = (result["source_kind"] if result else st.session_state.get("src_kind")) or "-"
    language = (result["language"] if result else st.session_state.get("src_lang")) or "-"
    if result and not result["language_applied"]:
        language = f"{language} (backend default)"
    with st.sidebar:
        md(f'<div class="sb-brand"><div class="logo" style="width:32px;height:32px">{icon("logo", 18)}</div>{APP_NAME}</div>')
        md('<div class="sb-label">Project</div><div class="sb-text">Video and audio analysis with transcription, structured extraction and retrieval-augmented Q&A.</div>')
        md(
            f"""<div class="sb-label">Processing</div>
            <div class="kv"><span>Source</span><span>{esc(source)}</span></div>
            <div class="kv"><span>Language</span><span>{esc(language)}</span></div>
            <div class="kv"><span>Pipeline</span><span>Transcribe, extract, RAG</span></div>
            <div class="kv"><span>Engine</span><span>{'Ready' if backend else 'Unavailable'}</span></div>"""
        )
        md('<div class="sb-label">Session</div>')
        st.button("New Analysis", on_click=cb_new_analysis, use_container_width=True, help="Reset everything and start over")
        st.button("Clear Results", on_click=cb_clear_results, use_container_width=True,
                  disabled=st.session_state.result is None, help="Remove the current results and chat history")
        md('<div class="sb-label">About</div><div class="sb-text">Paste a YouTube link or upload a recording. The app transcribes it, '
           'writes a summary, pulls out decisions, action items and questions, and lets you ask follow-up questions answered from the transcript.</div>')
        md('<div class="sb-foot">Built with Python • Streamlit • RAG • AI</div>')


def render_error(info: ErrorInfo, dismissible: bool = True) -> None:
    md(
        f"""<div class="err-card"><div class="h">{icon("alert", 18)}Something went wrong while processing the video.</div>
        <div class="b"><b>{esc(info.title)}.</b> {esc(info.hint)}<br>
        <span class="muted">Failed during: {esc(GROUP_LABELS.get(info.stage, info.stage))}</span></div></div>"""
    )
    with st.expander("Technical details"):
        st.code(info.detail, language="text")
        st.caption("The full traceback was written to the server log.")
    if dismissible:
        st.button("Dismiss", on_click=cb_dismiss_error)


def validate_input(kind: str, url: str, upload: Any) -> Optional[str]:
    if kind == SOURCE_YOUTUBE:
        url = (url or "").strip()
        if not url:
            return "Paste a YouTube URL to get started."
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower().split(":")[0]
        if parsed.scheme not in ("http", "https") or not any(host == h or host.endswith("." + h) for h in YOUTUBE_HOSTS):
            return "That doesn't look like a valid YouTube link. Use a URL such as https://www.youtube.com/watch?v=..."
        return None
    if upload is None:
        return "Upload a video or audio file to get started."
    ext = Path(upload.name).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        return f"Unsupported file type '.{ext or '?'}'. Supported: {', '.join(ALLOWED_EXTENSIONS)}."
    if upload.size == 0:
        return "The uploaded file is empty."
    return None


def render_hero(backend: SimpleNamespace) -> None:
    md(
        """<div class="hero"><h1>Transform any video into actionable intelligence.</h1>
        <p>Transcribe, summarize, extract insights, and ask questions about your content.</p></div>"""
    )
    _, mid, _ = st.columns([1, 5, 1])
    with mid:
        kind = st.radio("Source type", SOURCE_KINDS, horizontal=True, key="src_kind", label_visibility="collapsed")
        with st.form("analyze_form", clear_on_submit=False):
            url, upload = "", None
            if kind == SOURCE_YOUTUBE:
                url = st.text_input("YouTube URL", key="src_url", placeholder="https://www.youtube.com/watch?v=...",
                                    help="Public YouTube video link")
            else:
                upload = st.file_uploader("Video or audio file", type=ALLOWED_EXTENSIONS,
                                          key=f"src_file_{st.session_state.uploader_nonce}",
                                          label_visibility="collapsed",
                                          help="Drag and drop a recording, or browse your files")
            col_lang, col_btn = columns_bottom([1, 1.2])
            with col_lang:
                language = st.selectbox("Language", list(LANGUAGES), key="src_lang",
                                        help="Spoken language of the source")
            with col_btn:
                submitted = st.form_submit_button("Analyze Video", type="primary", use_container_width=True)

        if submitted:
            problem = validate_input(kind, url, upload)
            if problem:
                md(f'<div class="inline-err">{esc(problem)}</div>')
            else:
                st.session_state.error = None
                st.session_state.job = {
                    "kind": kind,
                    "source": url.strip() if kind == SOURCE_YOUTUBE else upload.name,
                    "filename": upload.name if upload else "",
                    "bytes": upload.getvalue() if upload else b"",
                    "language": language,
                }
                st.rerun()

        if backend_supports_language(backend):
            md('<div class="note">The selected language is passed to the transcriber.</div>')
        else:
            md('<div class="note">The current transcriber has no language option, so it uses its own default. Your selection is recorded for reference.</div>')
        md('<div class="formats">' + "".join(f'<span class="chip">{e}</span>' for e in ["YouTube"] + [x.upper() for x in ALLOWED_EXTENSIONS]) + "</div>")

    md(
        f"""<div class="empty"><div class="orb">{icon("logo", 28)}</div>
        <h3>Your video intelligence workspace</h3><p>Upload a video or paste a YouTube URL to begin.</p></div>"""
    )
    md(
        f"""<div class="grid-3 how">
        <div class="card"><div class="ico">{icon("wave")}</div><h4>Ingest and transcribe</h4>
        <p>Media is fetched, converted to audio, chunked and transcribed.</p></div>
        <div class="card"><div class="ico">{icon("sparkles")}</div><h4>Structured extraction</h4>
        <p>Language model passes produce a title, summary, decisions, action items and open questions.</p></div>
        <div class="card"><div class="ico">{icon("layers")}</div><h4>Grounded Q&amp;A with RAG</h4>
        <p>The transcript is indexed so answers are retrieved from the video itself, not guessed.</p></div></div>"""
    )


def render_processing(backend: SimpleNamespace, job: dict[str, Any]) -> None:
    label = job["filename"] if job["kind"] == SOURCE_FILE else job["source"]
    panel = st.empty()
    tracker = StageTracker(panel, label)
    result, error = None, None
    try:
        result = execute_pipeline(backend, job, tracker)
    except AnalysisError as exc:
        error = exc.info
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected pipeline failure")
        error = classify_error("finalize", exc)

    st.session_state.job = None
    if error:
        st.session_state.error = error
    else:
        st.session_state.result = result
        st.session_state.chat = []
        st.session_state.just_finished = True
    st.rerun()


def insight_card(icon_name: str, title: str, items_html: list[str], empty_msg: str) -> str:
    if items_html:
        body = '<ul class="ins-list">' + "".join(f'<li><span class="bullet"></span><div class="itxt">{h}</div></li>' for h in items_html) + "</ul>"
    else:
        body = f'<div class="empty-note">{empty_msg}</div>'
    return (
        f'<div class="card"><div class="card-head" style="margin-bottom:0"><div class="ico">{icon(icon_name)}</div>'
        f'<h3>{title}</h3><span class="count">{len(items_html)}</span></div>{body}</div>'
    )


def render_overview(result: dict[str, Any]) -> None:
    summary_html = render_rich_text(result["summary"]) or '<p class="muted">No summary was generated.</p>'
    md(
        f"""<div class="card"><div class="card-head"><div class="ico">{icon("file")}</div><h3>Executive Summary</h3></div>
        <div class="summary">{summary_html}</div></div>"""
    )
    md('<div class="summary-actions-spacer"></div>')
    c1, c2, _ = st.columns([1.3, 1.3, 4])
    with c1:
        copy_button(result["summary"], "Copy Summary")
    with c2:
        st.download_button("Download Summary", result["summary"], file_name=f"{slugify(result['title'])}-summary.txt",
                           mime="text/plain", use_container_width=True)

    md('<div class="section-title">Key insights</div>')
    decisions = [inline_md(item_text(i)) for i in result["key_decisions"]]
    questions = [inline_md(item_text(i)) for i in result["open_questions"]]
    actions = []
    for item in result["action_items"]:
        p = parse_action_item(item)
        chips = "".join(f'<span class="chip">{k}: {esc(v)}</span>' for k, v in (("Owner", p["owner"]), ("Deadline", p["deadline"])) if v)
        actions.append(inline_md(p["task"]) + (f'<div class="meta">{chips}</div>' if chips else ""))
    md(
        '<div class="grid-3">'
        + insight_card("flag", "Key Decisions", decisions, "No decisions were identified in this video.")
        + insight_card("tasks", "Action Items", actions, "No action items were identified in this video.")
        + insight_card("help", "Questions", questions, "No open questions were identified in this video.")
        + "</div>"
    )
    st.write("")
    st.download_button("Download full report (Markdown)", build_report_md(result),
                       file_name=f"{slugify(result['title'])}-report.md", mime="text/markdown")


def render_transcript(result: dict[str, Any]) -> None:
    transcript = result["transcript"]
    segments = split_transcript(transcript)
    c_search, c_copy, c_dl = columns_bottom([4, 1.2, 1.2])
    with c_search:
        query = st.text_input("Search transcript", key="t_query", placeholder="Search the transcript...",
                              label_visibility="collapsed")
    with c_copy:
        copy_button(transcript, "Copy Transcript", height=40, button_height=40)
    with c_dl:
        st.download_button("Download", transcript, file_name=f"{slugify(result['title'])}-transcript.txt",
                           mime="text/plain", use_container_width=True)

    q = query.strip()
    matches = [(i, ts, text) for i, (ts, text) in enumerate(segments, 1) if not q or q.lower() in text.lower()]
    stat = f"{len(segments)} segments · {result['words']:,} words"
    if q:
        stat = f"{len(matches)} of {len(segments)} segments match “{esc(q)}”"
    md(f'<div class="stat">{stat}</div>')

    with st.expander("Transcript", expanded=True):
        with st.container(height=TRANSCRIPT_VIEW_HEIGHT):
            if not matches:
                md('<div class="empty-note">No segments match your search.</div>')
            pattern = re.compile(f"({re.escape(esc(q))})", re.I) if q else None
            rows = []
            for i, ts, text in matches:
                body = esc(text)
                if pattern:
                    body = pattern.sub(r"<mark>\1</mark>", body)
                stamp = f'<span class="ts">{esc(ts)}</span>' if ts else ""
                rows.append(f'<div class="tline"><span class="tno">{i:02d}</span>{stamp}<div>{body}</div></div>')
            if rows:
                md("".join(rows))


def render_message(msg: dict[str, Any]) -> None:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("error"):
            md(f'<div class="inline-err" style="margin:0">{esc(msg["content"])}</div>')
        else:
            st.markdown(msg["content"].replace("$", "\\$"))


def render_chat(backend: SimpleNamespace, result: dict[str, Any]) -> None:
    head_l, head_r = st.columns([5, 1])
    with head_l:
        md(
            f"""<div class="card-head" style="margin-bottom:.2rem"><div class="ico">{icon("chat")}</div>
            <div><h3>Ask the Video</h3><div class="muted" style="font-size:.88rem">Ask questions and get answers grounded in the processed content.</div></div></div>"""
        )
    with head_r:
        if st.session_state.chat:
            st.button("Clear chat", on_click=cb_clear_chat, use_container_width=True)

    def chips() -> None:
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            with cols[i % 2]:
                st.button(q, key=f"sugg_{i}", on_click=cb_ask, args=(q,), use_container_width=True)

    if not st.session_state.chat:
        md('<div class="sb-label">Suggested questions</div>')
        chips()
    else:
        with st.expander("Suggested questions"):
            chips()

    pending = st.session_state.pending_q
    st.session_state.pending_q = None
    messages = st.container()
    with messages:
        for msg in st.session_state.chat:
            render_message(msg)
    typed = st.chat_input("Ask anything about this video...")
    question = (typed or pending or "").strip()
    if not question:
        return

    st.session_state.chat.append({"role": "user", "content": question})
    with messages:
        render_message(st.session_state.chat[-1])
        with st.chat_message("assistant", avatar="🤖"):
            slot = st.empty()
            slot.markdown('<div class="thinking"><i></i><i></i><i></i>&nbsp;Retrieving context and writing an answer</div>',
                          unsafe_allow_html=True)
            try:
                answer = answer_question(backend, result["rag_chain"], question)
                reply = {"role": "assistant", "content": answer}
                slot.markdown(answer.replace("$", "\\$"))
            except AnalysisError as exc:
                reply = {"role": "assistant", "error": True,
                         "content": f"{exc.info.title}. {exc.info.hint}"}
                slot.empty()
                md(f'<div class="inline-err" style="margin:0">{esc(reply["content"])}</div>')
    st.session_state.chat.append(reply)


def render_dashboard(backend: SimpleNamespace, result: dict[str, Any]) -> None:
    if st.session_state.just_finished:
        st.session_state.just_finished = False
        try:
            st.toast("Analysis complete", icon="✅")
        except Exception:  # noqa: BLE001
            pass

    language = result["language"] if result["language_applied"] else "Backend default"
    _, another_video_col = st.columns([3, 2])
    with another_video_col:
        st.button(
            "Analyze Another Video",
            on_click=cb_new_analysis,
            use_container_width=True,
        )

    md(
        f"""<div class="card title-card"><div class="label">Video Intelligence</div>
        <div class="t">{esc(result["title"])}</div><div class="s">{esc(result["source_label"])}</div></div>
        <div class="grid-5">
        <div class="card metric"><div class="k">Status</div><div class="v"><span class="dot"></span>Complete</div></div>
        <div class="card metric"><div class="k">Language</div><div class="v">{esc(language)}</div></div>
        <div class="card metric"><div class="k">Transcript</div><div class="v">Ready · {result["words"]:,} words</div></div>
        <div class="card metric"><div class="k">Knowledge base</div><div class="v">Ready for Q&amp;A</div></div>
        <div class="card metric"><div class="k">Processed in</div><div class="v">{fmt_duration(result["elapsed"])}</div></div></div>"""
    )
    st.write("")
    tab_overview, tab_chat, tab_transcript = st.tabs(["Overview", "Ask the Video", "Transcript"])
    with tab_overview:
        render_overview(result)
    with tab_chat:
        render_chat(backend, result)
    with tab_transcript:
        render_transcript(result)

    with st.expander("Export"):
        export_content_col, export_format_col = st.columns(2)
        with export_content_col:
            export_content = st.selectbox("Content", ["Summary", "Transcript", "Both"], key="export_content")
        with export_format_col:
            export_format = st.selectbox("Format", ["PDF", "Markdown"], key="export_format")

        export_markdown = build_export_markdown(result, export_content)
        export_name = {
            "Summary": "summary",
            "Transcript": "transcript",
            "Both": "summary_and_transcript",
        }[export_content]
        if export_format == "PDF":
            export_data = build_export_pdf(export_markdown)
            export_filename = f"clipiq_{export_name}.pdf"
            export_mime = "application/pdf"
        else:
            export_data = export_markdown
            export_filename = f"clipiq_{export_name}.md"
            export_mime = "text/markdown"
        st.download_button(
            "Download",
            export_data,
            file_name=export_filename,
            mime=export_mime,
            use_container_width=True,
        )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    init_state()
    md(CSS)
    backend, backend_error = get_backend()
    render_sidebar(backend)
    render_topbar(backend is not None)

    if backend is None:
        render_error(backend_error, dismissible=False)
        st.caption("Fix the issue above and reload the page.")
        return

    job = st.session_state.job
    if job:
        render_processing(backend, job)
        return

    if st.session_state.error:
        render_error(st.session_state.error)

    if st.session_state.result:
        render_dashboard(backend, st.session_state.result)
    else:
        render_hero(backend)


if __name__ == "__main__":
    main()