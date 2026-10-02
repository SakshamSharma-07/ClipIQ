from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


COLLECTION_DIR = "vector_db"
COLLECTION_NAME = "transcript"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


def build_vector_store(transcript : str)->Chroma:
    print("Building vector Store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )

    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index":i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        embedding = embeddings,
        documents = docs,
        collection_name = COLLECTION_NAME,
        persist_directory = COLLECTION_DIR
    )

    return vector_store


def load_vector_store():
    embeddings = get_embeddings()

    vector_store = Chroma(
        embedding = embeddings,
        collection_name = COLLECTION_NAME,
        persist_directory = COLLECTION_DIR
    )

    return vector_store


def get_retriever(vector_store, k=4):
    return vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":k}
    )

