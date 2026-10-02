# ClipIQ

ClipIQ is an AI-powered assistant for processing video and audio content into meaningful, structured insights. It is designed to help users turn long recordings into concise summaries, action items, key decisions, and searchable knowledge using transcription, LLM-based analysis, and retrieval-augmented generation (RAG).

Whether the source is a YouTube video or a local audio/video file, ClipIQ analyzes the content, extracts the transcript, and makes the information easier to understand and query.

---

## Overview

This project combines several modern AI and data tools to create a practical workflow:

- Audio extraction and preprocessing
- Audio chunking for large files
- Speech-to-text transcription using Whisper
- AI-powered summarization and key insight extraction
- Vector-based retrieval for answering questions from transcript content
- Chat-style interaction with the processed meeting or video content

ClipIQ is especially useful for:
- Meeting recordings and interviews
- Conference talks and educational videos
- YouTube content analysis
- Capturing decisions, tasks, and open questions automatically

---

## Problem It Solves

Videos and long audio recordings often contain valuable information, but they are hard to search, review, and summarize manually. ClipIQ solves this by turning raw media into structured results that are easier to interpret and query.

Instead of watching a long recording from start to finish, users can:
- quickly get a summary
- extract action items
- identify important decisions
- ask direct questions about the content
- keep important information in an organized, searchable form

---

## Key Features

### 1. YouTube URL Support
ClipIQ can download audio from a YouTube URL and process it automatically.

### 2. Local File Support
Users can also provide a local audio or video file for analysis.

### 3. Audio Preprocessing
The project converts media into a usable format, standardizes audio quality, and splits long recordings into manageable chunks.

### 4. Speech-to-Text Transcription
Using Whisper, the system converts recorded speech into text.

### 5. AI Summary Generation
The transcript is summarized into a clear, readable overview of the conversation or presentation.

### 6. Action Item Extraction
The model identifies tasks, responsibilities, and follow-ups mentioned in the recording.

### 7. Key Decision Extraction
Important decisions made during the content are surfaced for quick review.

### 8. Open Question Detection
The system highlights unresolved questions or points requiring clarification.

### 9. RAG-Based Q&A
After building a vector-based knowledge store from the transcript, users can ask questions about the content and receive answers grounded in the actual recording.

---

## How It Works

The project follows a structured AI pipeline:

1. Input is received
   - YouTube URL or local file path

2. Audio is prepared
   - Download or convert to WAV
   - Split into chunks

3. Transcription
   - Each chunk is transcribed using Whisper

4. Content analysis
   - Summary generation
   - Action item extraction
   - Key decision extraction
   - Question detection

5. Knowledge indexing
   - Transcript is embedded into a vector database

6. Querying
   - User asks questions
   - Relevant transcript segments are retrieved
   - LLM answers using those retrieved sections

---

## Tech Stack

ClipIQ uses a Python-based AI pipeline with the following core tools:

- Python
- Whisper for transcription
- LangChain for orchestration
- Groq for LLM inference
- Chroma / vector storage for retrieval
- yt-dlp for YouTube audio extraction
- Pydub for audio processing
- Sentence Transformers for embeddings
- dotenv for environment configuration

---

## Project Structure

```text
ClipIQ/
│
├── core/
│   ├── extractor.py          # Extracts action items, decisions, and questions
│   ├── rag_engine.py         # Retrieval + LLM Q&A pipeline
│   ├── summarize.py          # Summary and title generation
│   ├── transcriber.py        # Whisper transcription logic
│   └── vector_store.py       # Vector database creation and retrieval logic
│
├── utils/
│   └── audio_processor.py    # Downloading, converting, and chunking audio
│
├── vector_db/               # Stores generated vector database files
│
├── .gitignore
├── .python-version
├── app.py                   # Application entry / UI-related script if used
├── main.py                  # Main CLI pipeline entry point
├── pyproject.toml           # Project configuration
├── requirements.txt         # Python dependencies
├── test.py                  # Example / testing script
├── uv.lock                  # Lockfile for uv
└── README.md
