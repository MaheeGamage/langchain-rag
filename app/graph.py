# app/graph.py

import logging
import time
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from .retriever import get_retriever
from .config import LLM_PROVIDER, LLM_MODEL
from .factory import get_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [graph] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class RAGState(TypedDict):
    question: str
    context: str
    documents: List[Document]
    answer: str


retriever = get_retriever()
log.info("Using %s LLM: %s", LLM_PROVIDER, LLM_MODEL)
llm = get_llm() | StrOutputParser()


def retrieve(state: RAGState):
    t = time.perf_counter()
    docs = retriever.invoke(state["question"])
    return {"documents": docs}


def generate(state: RAGState):
    rag_context = "\n\n".join(doc.page_content for doc in state["documents"])
    user_context = state.get("context", "")

    user_context_section = f"\nUser-provided context:\n{user_context}\n" if user_context else ""

    prompt = f"""
Answer using the context below.
{user_context_section}
Retrieved context:
{rag_context}

Question:
{state["question"]}
"""
    t = time.perf_counter()
    answer = llm.invoke(prompt)
    return {"answer": answer}


def generate_without_retrieval(state: RAGState):
    prompt = f"""Answer the question below.
    Question: 
    {state["question"]}
    """
    t = time.perf_counter()
    answer = llm.invoke(prompt)
    return {"answer": answer}

def build_graph():

    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("generate_without_retrieval", generate_without_retrieval)  

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    # builder.set_entry_point("generate_without_retrieval")
    # builder.add_edge("generate_without_retrieval", END)

    return builder.compile()


graph = build_graph()
