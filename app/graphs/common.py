import logging
import time
from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.models import ContextEntry
from app.prompts.generate_prompt import GENERATOR_PROMPT


def latest_user_query(messages: list[BaseMessage]) -> str:
    return next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
        "",
    )


def invoke_retriever_with_retry(retriever, query: str, log: logging.Logger, max_attempts: int = 3):
    """Retry retriever calls for transient transport issues."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return retriever.invoke(query)
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            backoff_s = 0.25 * (2 ** (attempt - 1))
            log.warning(
                "Retriever invoke failed (attempt %d/%d): %s. Retrying in %.2fs",
                attempt,
                max_attempts,
                exc,
                backoff_s,
            )
            time.sleep(backoff_s)

    raise RuntimeError(f"Retriever invoke failed after {max_attempts} attempts: {last_exc}") from last_exc


def invoke_llm_with_retry(llm, prompt, log: logging.Logger, max_attempts: int = 3):
    """Retry LLM calls for transient transport failures (e.g. connection reset)."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            backoff_s = 0.25 * (2 ** (attempt - 1))
            log.warning(
                "LLM invoke failed (attempt %d/%d): %s. Retrying in %.2fs",
                attempt, max_attempts, exc, backoff_s,
            )
            time.sleep(backoff_s)
    raise RuntimeError(f"LLM invoke failed after {max_attempts} attempts: {last_exc}") from last_exc


def docs_to_context_entries(docs: Iterable) -> list[ContextEntry]:
    return [
        ContextEntry(
            type="snippet",
            name=doc.metadata.get("source"),
            content=doc.page_content,
            source="retriever",
            score=doc.metadata.get("score"),
        )
        for doc in docs
    ]


def context_entries_to_sources(entries: list[ContextEntry]) -> list[dict]:
    return [
        {
            "content": entry.content or "",
            "metadata": {
                "source": entry.name or "",
                **({"score": entry.score} if entry.score is not None else {}),
            },
        }
        for entry in entries
    ]


def build_messages(
    *,
    messages: list[BaseMessage],
    context_entries: list[ContextEntry],
    retrieved_entries: list[ContextEntry],
    llm_model: str,
) -> list[BaseMessage]:
    rag_context = "\n\n".join(e.content for e in retrieved_entries if e.content)

    user_context_parts = []
    for entry in context_entries:
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

    system_content = GENERATOR_PROMPT.substitute(
        user_context=user_context or "None",
        rag_context=rag_context or "None",
    )

    if "gemma" in llm_model.lower():
        full_messages: list[BaseMessage] = [
            HumanMessage(content=system_content),
            AIMessage(content="Understood."),
        ]
    else:
        full_messages = [SystemMessage(content=system_content)]

    full_messages.extend(messages)
    return full_messages


def token_to_text(token) -> str:
    if isinstance(token, str):
        return token

    content = getattr(token, "content", None)
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

    return str(token) if token is not None else ""


def chunk_text_for_streaming(text: str) -> list[str]:
    if not text:
        return []

    chunks: list[str] = []
    current: list[str] = []
    for word in text.split(" "):
        current.append(word)
        if len(current) >= 6:
            chunks.append(" ".join(current) + " ")
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks
