# app/api.py

import json
from app.schemas import QueryRequest, QueryResponse, SourceChunk
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from .graph import graph, build_messages, retrieve
from .models import ContextEntry
from .config import (
    CHROMA_TARGET,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    LLM_MODEL,
    LLM_PROVIDER,
    MLFLOW_ENABLED,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from .factory import get_llm

_TAGS = [
    {
        "name": "system",
        "description": "Health checks and runtime configuration.",
    },
    {
        "name": "rag",
        "description": "Retrieval-Augmented Generation endpoints. "
                       "Submit a question (with optional conversation history and "
                       "injected context) and receive an answer grounded in the "
                       "indexed documents.",
    },
]

# ── MLflow tracing setup ──────────────────────────────────────────────────────
if MLFLOW_ENABLED:
    import mlflow
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.langchain.autolog()

app = FastAPI(
    title="LangChain RAG API",
    description=(
        "A local Retrieval-Augmented Generation system built with "
        "LangChain, LangGraph, and ChromaDB.\n\n"
        "**Providers** for both LLM and embeddings are configured "
        "independently via `LLM_PROVIDER` / `EMBEDDING_PROVIDER` in `.env`. "
        "Supported values: `ollama`, `openai`, `gemini`.\n\n"
        "Interactive docs: **`/docs`** (Swagger UI) · **`/redoc`** (ReDoc)."
    ),
    version="1.0.0",
    openapi_tags=_TAGS,
    contact={
        "name": "Project repository",
        "url": "https://github.com/your-org/langchain-rag",
    },
    license_info={
        "name": "MIT",
    },
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["system"],
    summary="Health check",
    description="Returns `{\"status\": \"ok\"}` when the API process is running.",
    response_description="Service is healthy.",
)
async def health():
    return {"status": "ok"}


@app.get(
    "/config",
    tags=["system"],
    summary="Runtime configuration",
    description=(
        "Returns the active LLM and embedding provider/model names as "
        "resolved from environment variables at startup."
    ),
    response_description="Active provider and model names.",
)
async def config():
    return {
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "llm_provider": LLM_PROVIDER,
        "embedding_provider": EMBEDDING_PROVIDER,
        "chroma_target": CHROMA_TARGET,
    }


def _build_sources(entries: list[ContextEntry]) -> list[SourceChunk]:
    return [
        SourceChunk(
            content=entry.content or "",
            metadata={
                "source": entry.name or "",
                **({"score": entry.score} if entry.score is not None else {}),
            },
        )
        for entry in entries
    ]


def _format_sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["rag"],
    summary="Ask a question",
    description=(
        "Send a natural-language question to the RAG pipeline. "
        "Optionally include prior conversation turns (`conversation.history`) "
        "for multi-turn dialogue, or pre-retrieved context chunks (`context.entries`) "
        "to inject external documents directly into the prompt.\n\n"
        "The pipeline retrieves relevant chunks from ChromaDB, augments the prompt, "
        "and returns an answer together with the source chunks used."
    ),
    response_description="Generated answer and supporting source chunks.",
)
async def query(req: QueryRequest):
    thread_id = req.conversation.id if req.conversation and req.conversation.id else str(uuid.uuid4())

    messages = [HumanMessage(content=req.message)]
    config = {"configurable": {"thread_id": thread_id}}

    context_entries = req.context.entries if req.context else []

    result = graph.invoke(
        {"messages": messages, "context": context_entries, "retrieved": []},
        config=config,
    )

    # Extract the answer from the last AIMessage in the returned messages
    answer = ""
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            answer = m.content
            break

    sources = _build_sources(result.get("retrieved", []))
    return QueryResponse(thread_id=thread_id, answer=answer, sources=sources)


@app.post(
    "/query/stream",
    tags=["rag"],
    summary="Ask a question (streaming)",
    description=(
        "Streaming variant of `/query` that emits Server-Sent Events (SSE). "
        "Events: `token` (incremental text), `done` (final payload), `error`."
    ),
    response_description="SSE stream of tokens followed by a final JSON payload.",
)
async def query_stream(req: QueryRequest):
    thread_id = req.conversation.id if req.conversation and req.conversation.id else str(uuid.uuid4())
    context_entries = req.context.entries if req.context else []

    async def event_generator():
        try:
            state = {
                "messages": [HumanMessage(content=req.message)],
                "context": context_entries,
                "retrieved": [],
            }
            state.update(retrieve(state))
            messages = build_messages(state)
            sources = _build_sources(state.get("retrieved", []))

            llm = get_llm()
            answer_parts: list[str] = []
            for chunk in llm.stream(messages):
                text = getattr(chunk, "content", None)
                if text is None:
                    text = str(chunk)
                if text:
                    answer_parts.append(text)
                    yield _format_sse("token", {"text": text})
            yield _format_sse(
                "done",
                {
                    "thread_id": thread_id,
                    "answer": "".join(answer_parts),
                    "sources": [s.model_dump() for s in sources],
                },
            )
        except Exception as exc:
            yield _format_sse("error", {"thread_id": thread_id, "error": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
