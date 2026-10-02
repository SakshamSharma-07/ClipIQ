from utils.audio_processor import process_input
from core.transcriber import transcribe_chunks

source = "https://www.youtube.com/watch?v=PxHxIJd2ZkI"

chunks = process_input(source)
print(transcribe_chunks(chunks))