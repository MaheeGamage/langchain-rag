# app/retriever.py

from .vectorstore import get_vectorstore


def get_retriever():
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": 4})
