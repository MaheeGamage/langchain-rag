import logging
import time
from typing import Iterator

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.config import LLM_MODEL, LLM_PROVIDER
from app.factory import get_chat_llm
from app.graphs.common import (
    BASE_PROMPT,
    chunk_text_for_streaming,
    docs_to_context_entries,
    invoke_retriever_with_retry,
)
from app.models import ContextEntry
from app.retriever import (
    BM25Config,
    SemanticConfig,
    get_bm25_retriever,
    get_hybrid_retriever,
    get_semantic_retriever,
)

log = logging.getLogger(__name__)

retriever = get_hybrid_retriever([
    (get_semantic_retriever(SemanticConfig(k=5, score_threshold=0.55)), 0.5),
    (get_bm25_retriever(BM25Config(k=5)), 0.5),
])

AGENT_SYSTEM_PROMPT = (
    BASE_PROMPT
    + "\n\nYou have a `retrieve_context` tool that searches the knowledge base. "
    "Call it whenever the user's question requires factual information you do not "
    "already have. You may call it multiple times with refined queries if the first "
    "result is insufficient. Once you have enough context, answer directly without "
    "calling the tool again."
)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> tuple[str, list[Document]]:
    """Search the MLflow + quantum software experiment knowledge base.

    Args:
        query: A focused search query. Reformulate the user's question into
            keywords or a short phrase that captures what to look up.
    """
    t = time.perf_counter()
    docs = invoke_retriever_with_retry(retriever, query, log)
    log.info("rag_agent retrieve: %d chunks in %.2fs", len(docs), time.perf_counter() - t)

    serialised = "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )
    return serialised, list(docs)


def build_graph(*, checkpointer=None):
    log.info("Using %s LLM (rag_agent): %s", LLM_PROVIDER, LLM_MODEL)
    model = get_chat_llm()
    return create_agent(
        model,
        [retrieve_context],
        system_prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def _format_user_context(entries: list[ContextEntry]) -> str:
    parts: list[str] = []
    for entry in entries:
        header = f"[{entry.type}]"
        if entry.name:
            header += f" {entry.name}"
        if entry.mimeType:
            header += f" ({entry.mimeType})"
        if entry.content:
            parts.append(f"{header}\n{entry.content}")
    return "\n\n".join(parts)


def _extract_retrieved_docs(messages: list[BaseMessage]) -> list[ContextEntry]:
    docs: list[Document] = []
    for msg in messages:
        artifact = getattr(msg, "artifact", None)
        if isinstance(msg, ToolMessage) and isinstance(artifact, list):
            docs.extend(d for d in artifact if isinstance(d, Document))
    return docs_to_context_entries(docs)


def _message_text(msg: AIMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def stream_answer(
    *,
    messages: list[BaseMessage],
    context_entries: list[ContextEntry],
    graph_instance=None,
    config: dict | None = None,
) -> tuple[list[ContextEntry], Iterator[str]]:
    if graph_instance is None:
        graph_instance = build_graph()

    invoke_messages: list[BaseMessage] = []
    user_ctx = _format_user_context(context_entries)
    if user_ctx:
        invoke_messages.append(SystemMessage(content=f"User-provided context:\n{user_ctx}"))
    invoke_messages.extend(messages)

    result = graph_instance.invoke({"messages": invoke_messages}, config=config)
    result_messages = result.get("messages", [])

    answer = ""
    for msg in reversed(result_messages):
        if isinstance(msg, AIMessage):
            text = _message_text(msg)
            if text:
                answer = text
                break

    retrieved = _extract_retrieved_docs(result_messages)
    chunks = chunk_text_for_streaming(answer)

    def _iterator() -> Iterator[str]:
        for chunk in chunks:
            if chunk:
                yield chunk

    return retrieved, _iterator()
