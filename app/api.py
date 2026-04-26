# app/api.py

from app.schemas import QueryRequest, QueryResponse, SourceChunk
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
import uuid
import json
from .graph import ACTIVE_GRAPH_IMPLEMENTATION, graph, invoke_query, stream_query
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
        "rag_graph_implementation": ACTIVE_GRAPH_IMPLEMENTATION,
    }


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
    context_entries = req.context.entries if req.context else []

    retrieved, answer = invoke_query(
        messages=messages,
        context_entries=context_entries,
        thread_id=thread_id,
    )

    sources = [
        SourceChunk(
            content=entry.content or "",
            metadata={
                "source": entry.name or "",
                **({"score": entry.score} if entry.score is not None else {}),
            },
        )
        for entry in retrieved
    ]
    return QueryResponse(thread_id=thread_id, answer=answer, sources=sources)


@app.post(
    "/query/stream",
    tags=["rag"],
    summary="Ask a question (streaming)",
    description=(
        "Send a natural-language question to the RAG pipeline and receive "
        "the answer as a stream of tokens. Similar to /query but streams "
        "the generated answer token-by-token."
    ),
    response_description="Newline-delimited JSON stream with tokens and metadata.",
)
async def query_stream(req: QueryRequest):
    """Stream tokens as they are generated.
    
    Yields newline-delimited JSON events with types: metadata, token, error.
    """
    thread_id = req.conversation.id if req.conversation and req.conversation.id else str(uuid.uuid4())
    messages_list = [HumanMessage(content=req.message)]
    context_entries = req.context.entries if req.context else []

    def token_generator():
        """Generate newline-delimited JSON stream events from the active graph."""
        for event in stream_query(
            messages=messages_list,
            context_entries=context_entries,
            thread_id=thread_id,
        ):
            yield json.dumps(event) + "\n"
    
    return StreamingResponse(token_generator(), media_type="application/x-ndjson")
