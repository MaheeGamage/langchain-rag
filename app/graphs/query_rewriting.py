import logging
import re
import time
from typing import Iterator

import mlflow
from langsmith import traceable
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from mlflow.entities import SpanType

from app.config import HELPER_LLM_MODEL, HELPER_LLM_PROVIDER, LLM_MODEL, LLM_PROVIDER
from app.factory import get_helper_llm, get_llm
from app.graphs.common import (
    build_messages,
    chunk_text_for_streaming,
    docs_to_context_entries,
    invoke_llm_with_retry,
    invoke_retriever_with_retry,
    latest_user_query,
)
from app.graphs.types import GraphState
from app.models import ContextEntry
from app.prompts.query_rewrite import REWRITE_PROMPT
from app.retriever import PROFILE_NAMES, get_profile_retriever

log = logging.getLogger(__name__)

# Pre-profile-era fixed retriever — kept as a comment for reference.  To restore
# exact previous behaviour (single fixed k=4, threshold=0.55, whitespace BM25),
# swap `get_profile_retriever(profile)` in retrieve() for this:
#   from app.retriever import BM25Config, HybridConfig, SemanticConfig, build_hybrid_retriever
#   _LEGACY_RETRIEVER_V1 = build_hybrid_retriever(HybridConfig(
#       semantic=SemanticConfig(k=4, score_threshold=0.55),
#       bm25=BM25Config(k=4),
#       semantic_weight=0.5, bm25_weight=0.5,
#   ))
log.info("Using %s LLM: %s", LLM_PROVIDER, LLM_MODEL)
log.info("Using %s helper LLM: %s", HELPER_LLM_PROVIDER, HELPER_LLM_MODEL)
llm = get_llm() | StrOutputParser()
helper_llm = get_helper_llm() | StrOutputParser()


class QueryRewritingState(GraphState, total=False):
    rewritten_query: str
    retrieval_profile: str


_QUERY_RE = re.compile(r"QUERY\s*:\s*(.+)", re.IGNORECASE)
_PROFILE_RE = re.compile(r"PROFILE\s*:\s*([A-Za-z_]+)", re.IGNORECASE)


def _parse_rewrite(raw: str, fallback_query: str) -> tuple[str, str]:
    """Parse the helper LLM's output into (query, profile).

    Tolerant of V2-style single-line output: if no ``QUERY:`` prefix is found,
    the whole stripped string is treated as the query and profile falls back
    to "default".  Unknown profile names also fall back to "default".
    """
    q_match = _QUERY_RE.search(raw)
    p_match = _PROFILE_RE.search(raw)

    if q_match:
        query = q_match.group(1).strip()
    else:
        # V2-style output (no QUERY: prefix) — treat the whole thing as query,
        # but drop any trailing PROFILE: line if present.
        query = _PROFILE_RE.sub("", raw).strip()
    if not query:
        query = fallback_query

    profile = p_match.group(1).strip().lower() if p_match else "default"
    if profile not in PROFILE_NAMES:
        log.warning("Helper LLM picked unknown profile %r; using 'default'", profile)
        profile = "default"

    return query, profile


def rewrite_query(state: QueryRewritingState, run_config: dict | None = None):
    question = latest_user_query(state["messages"])
    t = time.perf_counter()
    invoke_kwargs = {"config": run_config} if run_config else {}
    raw = helper_llm.invoke(REWRITE_PROMPT.substitute(question=question), **invoke_kwargs)
    query, profile = _parse_rewrite(raw, fallback_query=question)
    log.info(
        "Rewrote in %.2fs: %r -> query=%r profile=%s",
        time.perf_counter() - t, question, query, profile,
    )
    return {"rewritten_query": query, "retrieval_profile": profile}


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(state: QueryRewritingState):
    query = state.get("rewritten_query") or latest_user_query(state["messages"])
    profile = state.get("retrieval_profile", "default")
    retriever = get_profile_retriever(profile)
    t = time.perf_counter()
    docs = invoke_retriever_with_retry(retriever, query, log)
    log.info(
        "Retrieved[%s] %d chunks in %.2fs",
        profile, len(docs), time.perf_counter() - t,
    )

    retrieved = docs_to_context_entries(docs)
    return {"retrieved": retrieved}


def generate(state: QueryRewritingState):
    messages = build_messages(
        messages=state["messages"],
        context_entries=state.get("context", []),
        retrieved_entries=state.get("retrieved", []),
        llm_model=LLM_MODEL,
    )
    t = time.perf_counter()
    answer = llm.invoke(messages)
    log.info("Generated answer in %.2fs", time.perf_counter() - t)
    return {"messages": [AIMessage(content=answer)]}


def build_graph(*, checkpointer=None, profile_selection: bool = True):
    builder = StateGraph(QueryRewritingState)

    builder.add_node("rewrite_query", rewrite_query)

    if profile_selection:
        builder.add_node("retrieve", retrieve)
    else:
        def _retrieve_default_profile(state: QueryRewritingState):
            return retrieve({**state, "retrieval_profile": "default"})
        builder.add_node("retrieve", _retrieve_default_profile)

    builder.add_node("generate", generate)

    builder.set_entry_point("rewrite_query")
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


@traceable(name="query_rewriting/stream_answer")
def stream_answer(
    *,
    messages: list[BaseMessage],
    context_entries: list[ContextEntry],
    graph_instance=None,
    config: dict | None = None,
    profile_selection: bool = True,
) -> tuple[list[ContextEntry], Iterator[str]]:
    state: QueryRewritingState = {
        "messages": messages,
        "context": context_entries,
        "retrieved": [],
    }

    _cbs = (config or {}).get("callbacks", [])
    _invoke_cfg = {"callbacks": _cbs} if _cbs else None

    rewrite_result = rewrite_query(state, run_config=_invoke_cfg)
    state.update(rewrite_result)

    if not profile_selection:
        state["retrieval_profile"] = "default"

    retrieval_result = retrieve(state)
    state.update(retrieval_result)

    prompt_messages = build_messages(
        messages=messages,
        context_entries=context_entries,
        retrieved_entries=state.get("retrieved", []),
        llm_model=LLM_MODEL,
    )

    answer = invoke_llm_with_retry(get_llm(), prompt_messages, log, run_config=_invoke_cfg)
    chunks = chunk_text_for_streaming(answer if isinstance(answer, str) else answer.content)

    def _token_iterator() -> Iterator[str]:
        for chunk in chunks:
            if chunk:
                yield chunk

    return state.get("retrieved", []), _token_iterator()