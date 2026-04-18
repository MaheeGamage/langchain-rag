import httpx
import chromadb
from langchain_chroma import Chroma

from .config import (
    CHROMA_HOST,
    CHROMA_PORT,
    CHROMA_SSL,
    COLLECTION_NAME,
    HTTP_KEEPALIVE_EXPIRY_S,
    HTTP_TIMEOUT_S,
)
from .factory import get_embeddings


def get_vectorstore() -> Chroma:
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, ssl=CHROMA_SSL)

    old = client._server._session
    client._server._session = httpx.Client(
        timeout=httpx.Timeout(HTTP_TIMEOUT_S),
        limits=httpx.Limits(
            max_connections=None,
            max_keepalive_connections=None,
            keepalive_expiry=HTTP_KEEPALIVE_EXPIRY_S,
        ),
        headers=dict(old.headers),
    )
    old.close()

    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )
