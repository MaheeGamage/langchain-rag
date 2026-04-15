import logging
import re
import time
from typing import Iterator, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

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
from app.retriever import get_retriever

log = logging.getLogger(__name__)

MAX_RETRIEVAL_ROUNDS = 2
retriever = get_retriever()
planner_llm = get_llm() | StrOutputParser()
generator_llm = get_llm() | StrOutputParser()


class AgenticGraphState(GraphState, total=False):
    next_step: Literal["retrieve", "generate"]
    active_query: str
    retrieval_round: int


def _parse_decision(text: str, default_query: str) -> tuple[str, str]:
    action_match = re.search(r"ACTION\s*:\s*(retrieve|generate)", text, flags=re.IGNORECASE)
    query_match = re.search(r"QUERY\s*:\s*(.+)", text, flags=re.IGNORECASE)

    action = action_match.group(1).lower() if action_match else "retrieve"
    query = query_match.group(1).strip() if query_match else default_query
    if not query:
        query = default_query

    return action, query


def _dedupe_entries(entries: list[ContextEntry]) -> list[ContextEntry]:
    seen: set[tuple[str, str]] = set()
    deduped: list[ContextEntry] = []

    for entry in entries:
        key = (entry.name or "", entry.content or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)

    return deduped


def plan(state: AgenticGraphState):
    question = latest_user_query(state["messages"])
    retrieved = state.get("retrieved", [])

    if retrieved:
        return {"next_step": "generate"}

    context_count = len(state.get("context", []))
    planner_prompt = (
        "Decide whether to retrieve from the knowledge base before answering.\\n"
        "Return exactly two lines:\\n"
        "ACTION: retrieve or generate\\n"
        "QUERY: <query to run if retrieving; otherwise repeat user question>\\n\\n"
        f"User question: {question}\\n"
        f"Injected context items: {context_count}"
    )

    try:
        raw = planner_llm.invoke(planner_prompt)
        action, query = _parse_decision(raw, question)
    except Exception as exc:
        log.warning("Planner failed, defaulting to retrieve: %s", exc)
        action, query = "retrieve", question

    next_step = "retrieve" if action == "retrieve" else "generate"
    return {
        "next_step": next_step,
        "active_query": query,
        "retrieval_round": state.get("retrieval_round", 0),
    }


def retrieve(state: AgenticGraphState):
    query = state.get("active_query") or latest_user_query(state["messages"])
    t = time.perf_counter()
    docs = invoke_retriever_with_retry(retriever, query, log)
    log.info("Agentic retrieve: %d chunks in %.2fs", len(docs), time.perf_counter() - t)

    retrieved_now = docs_to_context_entries(docs)
    merged = _dedupe_entries([*state.get("retrieved", []), *retrieved_now])

    return {
        "retrieved": merged,
        "retrieval_round": state.get("retrieval_round", 0) + 1,
    }


def assess(state: AgenticGraphState):
    question = latest_user_query(state["messages"])
    round_count = state.get("retrieval_round", 0)

    if round_count >= MAX_RETRIEVAL_ROUNDS:
        return {"next_step": "generate"}

    snippets = "\\n\\n".join(
        (entry.content or "")[:800] for entry in state.get("retrieved", [])[:4]
    )

    judge_prompt = (
        "You are deciding whether another retrieval step is needed.\\n"
        "Return exactly two lines:\\n"
        "ACTION: retrieve or generate\\n"
        "QUERY: <refined query if ACTION is retrieve, otherwise repeat question>\\n\\n"
        f"User question: {question}\\n"
        f"Current retrieved context:\\n{snippets}"
    )

    try:
        raw = planner_llm.invoke(judge_prompt)
        action, query = _parse_decision(raw, question)
    except Exception as exc:
        log.warning("Assess failed, defaulting to generate: %s", exc)
        action, query = "generate", question

    next_step = "retrieve" if action == "retrieve" else "generate"
    return {"next_step": next_step, "active_query": query}


def generate(state: AgenticGraphState):
    messages = build_messages(
        messages=state["messages"],
        context_entries=state.get("context", []),
        retrieved_entries=state.get("retrieved", []),
        llm_model=LLM_MODEL,
    )
    t = time.perf_counter()
    answer = generator_llm.invoke(messages)
    log.info("Agentic generate in %.2fs (rounds=%d)", time.perf_counter() - t, state.get("retrieval_round", 0))
    return {"messages": [AIMessage(content=answer)]}


def _route_after_plan(state: AgenticGraphState):
    return state.get("next_step", "retrieve")


def _route_after_assess(state: AgenticGraphState):
    return state.get("next_step", "generate")


def build_graph(*, checkpointer=None):
    log.info("Using %s LLM (agentic): %s", LLM_PROVIDER, LLM_MODEL)

    builder = StateGraph(AgenticGraphState)
    builder.add_node("plan", plan)
    builder.add_node("retrieve", retrieve)
    builder.add_node("assess", assess)
    builder.add_node("generate", generate)

    builder.set_entry_point("plan")

    builder.add_conditional_edges(
        "plan",
        _route_after_plan,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )
    builder.add_edge("retrieve", "assess")
    builder.add_conditional_edges(
        "assess",
        _route_after_assess,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )
    builder.add_edge("generate", END)

    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


def stream_answer(
    *,
    messages: list[BaseMessage],
    context_entries: list[ContextEntry],
    graph_instance=None,
    config: dict | None = None,
) -> tuple[list[ContextEntry], Iterator[str]]:
    if graph_instance is None:
        graph_instance = build_graph()

    result = graph_instance.invoke(
        {
            "messages": messages,
            "context": context_entries,
            "retrieved": [],
        },
        config=config,
    )

    answer = ""
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage):
            answer = message.content
            break

    chunks = chunk_text_for_streaming(answer)

    def _iterator() -> Iterator[str]:
        for chunk in chunks:
            if chunk:
                yield chunk

    return result.get("retrieved", []), _iterator()
