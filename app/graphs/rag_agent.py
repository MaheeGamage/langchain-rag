import logging
import time
from typing import Iterator, Literal

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.config import LLM_MODEL, LLM_PROVIDER, RETRIEVER_PROFILE_OVERRIDE
from app.factory import get_chat_llm
from app.graphs.common import (
    chunk_text_for_streaming,
    docs_to_context_entries,
    invoke_retriever_with_retry,
)
from app.prompts.generate_prompt import SYSTEM_TEMPLATE_V1
from app.models import ContextEntry
from app.retriever import PROFILE_DESCRIPTIONS, get_profile_retriever

log = logging.getLogger(__name__)

RetrievalProfile = Literal["default", "acronym", "conceptual", "overview", "reasoning", "reranked"]

_PROFILE_HELP = "\n".join(f"  - \"{name}\": {desc}" for name, desc in PROFILE_DESCRIPTIONS.items())

AGENT_SYSTEM_PROMPT = (
    SYSTEM_TEMPLATE_V1
    + "\n\nYou have a `retrieve_context` tool that searches the knowledge base. "
    "Call it whenever the user's question requires factual information you do not "
    "already have. You may call it multiple times with refined queries if the first "
    "result is insufficient. Once you have enough context, answer directly without "
    "calling the tool again.\n\n"
    "When you call `retrieve_context`, pick the `profile` argument that best matches "
    "the question shape:\n" + _PROFILE_HELP
)


@tool(response_format="content_and_artifact")
def retrieve_context(
    query: str,
    profile: RetrievalProfile = "default",
) -> tuple[str, list[Document]]:
    """Search the MLflow + quantum software experiment knowledge base.

    Args:
        query: A focused search query. Reformulate the user's question into
            keywords or a short phrase that captures what to look up.
        profile: Retrieval strategy. Pick the best match for the question:
            - "default":    balanced 50/50 hybrid; use when unsure.
            - "acronym":    BM25-heavy, k=6; for acronyms (NISQ, VQE) or exact
                            API names (mlflow.log_param).
            - "conceptual": semantic-heavy, k=4; for definitions and explanations.
            - "overview":   k=10; for summary/taxonomy/listing questions that
                            need broad context.
            - "reasoning":  k=8; for multi-hop reasoning needing several
                            supporting facts.
            - "reranked":   k=10 each then cross-encoder rerank to top 4;
                            higher latency, best precision on hard queries.
    """
    effective_profile = RETRIEVER_PROFILE_OVERRIDE or profile
    t = time.perf_counter()
    print("RETRIEVER_PROFILE", RETRIEVER_PROFILE_OVERRIDE, profile)
    retriever = get_profile_retriever(effective_profile)
    docs = invoke_retriever_with_retry(retriever, query, log)
    log.info(
        "rag_agent retrieve[%s]: %d chunks in %.2fs",
        effective_profile, len(docs), time.perf_counter() - t,
    )

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
