# app/api.py

from app.schemas import QueryRequest, QueryResponse, SourceChunk
from fastapi import FastAPI
from langchain_core.messages import HumanMessage, AIMessage
from .graph import graph
from .models import ContextEntry
from .config import (
    CHROMA_TARGET,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    LLM_MODEL,
    LLM_PROVIDER,
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
    # Build LangChain messages from conversation history + current message
    messages = []
    if req.conversation and req.conversation.history:
        for turn in req.conversation.history:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            else:
                messages.append(AIMessage(content=turn.content))
    messages.append(HumanMessage(content=req.message))

    context_entries = req.context.entries if req.context else []

    result = graph.invoke({
        "messages": messages,
        "context": context_entries,
        "retrieved": [],
    })

    # Extract the answer from the last AIMessage in the returned messages
    answer = ""
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            answer = m.content
            break

    sources = [
        SourceChunk(
            content=entry.content or "",
            metadata={
                "source": entry.name or "",
                **({"score": entry.score} if entry.score is not None else {}),
            },
        )
        for entry in result.get("retrieved", [])
    ]
    return QueryResponse(answer=answer, sources=sources)
