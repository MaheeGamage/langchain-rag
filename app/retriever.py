# app/retriever.py

from langchain_chroma import Chroma
from .config import CHROMA_PATH, COLLECTION_NAME
from .factory import get_embeddings


def get_retriever():
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})
