import chromadb
from langchain_chroma import Chroma

from .config import (
    CHROMA_HOST,
    CHROMA_PORT,
    CHROMA_SSL,
    COLLECTION_NAME,
)
from .factory import get_embeddings


def get_vectorstore() -> Chroma:
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, ssl=CHROMA_SSL)

    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )
