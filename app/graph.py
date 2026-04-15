import importlib
import logging
import sqlite3
from collections.abc import Callable, Iterator

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import CONVERSATIONS_DB, RAG_GRAPH_IMPLEMENTATION
from app.graphs.common import context_entries_to_sources
from app.models import ContextEntry

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

_IMPLEMENTATIONS = {
    "baseline": {
        "module": "app.graphs.baseline",
        "build_graph": "build_graph",
        "stream_answer": "stream_answer",
    },
    "agentic": {
        "module": "app.graphs.agentic",
        "build_graph": "build_graph",
        "stream_answer": "stream_answer",
    },
}


def _load_callables(name: str) -> tuple[Callable, Callable]:
    spec = _IMPLEMENTATIONS[name]
    module = importlib.import_module(spec["module"])
    build_graph = getattr(module, spec["build_graph"])
    stream_answer = getattr(module, spec["stream_answer"])
    return build_graph, stream_answer


ACTIVE_GRAPH_IMPLEMENTATION = RAG_GRAPH_IMPLEMENTATION
_build_graph, _stream_answer = _load_callables(ACTIVE_GRAPH_IMPLEMENTATION)
graph = _build_graph(checkpointer=_checkpointer)

log.info("Active RAG graph implementation: %s", ACTIVE_GRAPH_IMPLEMENTATION)


def stream_query(
    *,
    messages: list[BaseMessage],
    context_entries: list[ContextEntry],
    thread_id: str,
) -> Iterator[dict]:
    config = {"configurable": {"thread_id": thread_id}}

    try:
        retrieved, token_iterator = _stream_answer(
            messages=messages,
            context_entries=context_entries,
            graph_instance=graph,
            config=config,
        )
    except Exception as exc:
        yield {
            "type": "metadata",
            "sources": [],
            "thread_id": thread_id,
        }
        yield {
            "type": "error",
            "content": f"Retrieval/generation failed: {exc}",
        }
        return

    yield {
        "type": "metadata",
        "sources": context_entries_to_sources(retrieved),
        "thread_id": thread_id,
    }

    for token in token_iterator:
        if token:
            yield {
                "type": "token",
                "content": token,
            }
