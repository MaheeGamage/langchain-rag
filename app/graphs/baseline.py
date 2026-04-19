import logging
import time
from typing import Iterator

import mlflow
from langsmith import traceable
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from mlflow.entities import SpanType

from app.config import LLM_MODEL, LLM_PROVIDER
from app.factory import get_llm
from app.graphs.common import (
    build_messages,
    chunk_text_for_streaming,
    docs_to_context_entries,
    invoke_retriever_with_retry,
    latest_user_query,
)
from app.graphs.types import GraphState
from app.models import ContextEntry
from app.retriever import BM25Config, HybridConfig, SemanticConfig, build_hybrid_retriever

log = logging.getLogger(__name__)

retriever = build_hybrid_retriever(HybridConfig(
    semantic=SemanticConfig(k=4, score_threshold=0.55),
    bm25=BM25Config(k=4),
    semantic_weight=0.5,
    bm25_weight=0.5,
))
log.info("Using %s LLM: %s", LLM_PROVIDER, LLM_MODEL)
llm = get_llm() | StrOutputParser()


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(state: GraphState):
    query = latest_user_query(state["messages"])
    t = time.perf_counter()
    docs = invoke_retriever_with_retry(retriever, query, log)
    log.info("Retrieved %d chunks in %.2fs", len(docs), time.perf_counter() - t)

    retrieved = docs_to_context_entries(docs)
    return {"retrieved": retrieved}


def generate(state: GraphState):
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


def build_graph(*, checkpointer=None):
    builder = StateGraph(GraphState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


@traceable(name="baseline/stream_answer")
def stream_answer(
    *,
    messages: list[BaseMessage],
    context_entries: list[ContextEntry],
    graph_instance=None,
    config: dict | None = None,
) -> tuple[list[ContextEntry], Iterator[str]]:
    state: GraphState = {
        "messages": messages,
        "context": context_entries,
        "retrieved": [],
    }
    retrieval_result = retrieve(state)
    state.update(retrieval_result)

    prompt_messages = build_messages(
        messages=messages,
        context_entries=context_entries,
        retrieved_entries=state.get("retrieved", []),
        llm_model=LLM_MODEL,
    )

    answer = get_llm().invoke(prompt_messages)
    chunks = chunk_text_for_streaming(answer if isinstance(answer, str) else answer.content)

    def _token_iterator() -> Iterator[str]:
        for chunk in chunks:
            if chunk:
                yield chunk

    return state.get("retrieved", []), _token_iterator()
