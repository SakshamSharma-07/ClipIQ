import os 
import whisper

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None

def load_model():

    global _model

    if _model is None:
        print(f"loading model ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print(f"whisper model loaded successfully")

    return _model


def transcribe_chunk(chunk_path):
    model = load_model()

    result = model.transcribe(
        chunk_path,
        task="translate"
    )

    return result["text"]


def transcribe_all(chunks):
    full_transcript = ""

    for chunk in chunks:
        print(f"Transcribing: {os.path.basename(chunk)}")

        text = transcribe_chunk(chunk)
        full_transcript += text + " "

    return full_transcript.strip()


    