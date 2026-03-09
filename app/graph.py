# app/graph.py

import logging
import sqlite3
import time
from typing import Annotated, TypedDict, List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from .retriever import get_retriever
from .config import LLM_PROVIDER, LLM_MODEL, CONVERSATIONS_DB, RAG_SYSTEM_PROMPT
from .factory import get_llm
from .models import ContextEntry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [graph] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Keep the connection alive for the lifetime of the process.
# SqliteSaver requires check_same_thread=False for use across async request handlers.
_db_conn = sqlite3.connect(CONVERSATIONS_DB, check_same_thread=False)
_checkpointer = SqliteSaver(_db_conn)

class RAGState(TypedDict):
    # --- from API request ---
    messages:  Annotated[list[BaseMessage], add_messages]  # full conversation (history + latest turn). Currently client store conversation history  
    context:   list[ContextEntry]                          # entries forwarded from the API request

    # --- internal graph state ---
    retrieved: list[ContextEntry]                          # chunks fetched by the retriever


retriever = get_retriever()
log.info("Using %s LLM: %s", LLM_PROVIDER, LLM_MODEL)
llm = get_llm() | StrOutputParser()


def retrieve(state: RAGState):
    # Use the latest HumanMessage as the retrieval query
    query = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )
    t = time.perf_counter()
    docs = retriever.invoke(query)
    log.info("Retrieved %d chunks in %.2fs", len(docs), time.perf_counter() - t)
    retrieved = [
        ContextEntry(
            type="snippet",
            name=doc.metadata.get("source"),
            content=doc.page_content,
            source="retriever",
            score=doc.metadata.get("score"),
        )
        for doc in docs
    ]
    return {"retrieved": retrieved}

def build_messages(state: RAGState) -> list[BaseMessage]:
    # Retrieved chunks → string block
    rag_context = "\n\n".join(
        e.content for e in state.get("retrieved", []) if e.content
    )

    # User-provided context entries → string block
    user_context_parts = []
    for entry in state.get("context", []):
        header = f"[{entry.type}]"
        if entry.name:
            header += f" {entry.name}"
        if entry.mimeType:
            header += f" ({entry.mimeType})"
        if entry.score is not None:
            header += f" score={entry.score:.2f}"
        if entry.content:
            user_context_parts.append(f"{header}\n{entry.content}")
    user_context = "\n\n".join(user_context_parts)

    system_parts = [RAG_SYSTEM_PROMPT]
    if user_context:
        system_parts.append(f"User-provided context:\n{user_context}")
    if rag_context:
        system_parts.append(f"Retrieved context:\n{rag_context}")

    system_content = "\n\n".join(system_parts)

    # Gemma models do not support SystemMessage — fold it into a human/ai pair instead
    if "gemma" in LLM_MODEL.lower():
        messages: list[BaseMessage] = [
            HumanMessage(content=system_content),
            AIMessage(content="Understood."),
        ]
    else:
        messages = [SystemMessage(content=system_content)]

    messages.extend(state["messages"])  # full history: past turns + latest HumanMessage
    return messages


def generate(state: RAGState):
    messages = build_messages(state)
    t = time.perf_counter()
    answer = llm.invoke(messages)
    log.info("Generated answer in %.2fs", time.perf_counter() - t)
    return {"messages": [AIMessage(content=answer)]}


def build_graph():
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=_checkpointer)


graph = build_graph()
