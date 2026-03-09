# app/api.py

from app.schemas import QueryRequest, QueryResponse, SourceChunk, EvalRequest, EvalResponse
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
from .graph import graph
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
    {
        "name": "eval",
        "description": (
            "MLflow GenAI evaluation endpoints. Requires ``MLFLOW_ENABLED=true`` "
            "in the environment. Uses ``mlflow.genai.evaluate()`` with built-in "
            "LLM-as-a-Judge scorers to measure RAG pipeline quality."
        ),
    },
]

# ── MLflow tracing setup ──────────────────────────────────────────────────────
if MLFLOW_ENABLED:
    import mlflow
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.langchain.autolog()

# Thread pool for running blocking evaluation jobs without blocking the event loop
_eval_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlflow-eval")

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
    return QueryResponse(thread_id=thread_id, answer=answer, sources=sources)


@app.post(
    "/evaluate",
    response_model=EvalResponse,
    tags=["eval"],
    summary="Evaluate RAG pipeline quality with MLflow",
    description=(
        "Runs ``mlflow.genai.evaluate()`` against a provided dataset using "
        "LLM-as-a-Judge scorers. Each sample is passed through the full RAG "
        "pipeline (retrieval + generation) and scored by the selected judges.\n\n"
        "**Requires** ``MLFLOW_ENABLED=true`` and a reachable MLflow tracking "
        "server configured via ``MLFLOW_TRACKING_URI``.\n\n"
        "**Available scorers**: `relevance`, `correctness`, `fluency`, "
        "`groundedness`, `sufficiency`, `conciseness`, `domain_tone`.\n\n"
        "RAG-specific scorers (`groundedness`, `sufficiency`) inspect the "
        "retrieval trace and require no extra configuration.\n\n"
        "**Judge model** can be overridden per-request via `judge_model` "
        "(LiteLLM format: `openai:/gpt-4o-mini`, `ollama:/llama3.2`). "
        "Defaults to `MLFLOW_JUDGE_MODEL` env var or MLflow's built-in default."
    ),
    response_description="Evaluation run ID, aggregate metrics, and per-row scores.",
)
async def evaluate(req: EvalRequest):
    if not MLFLOW_ENABLED:
        raise HTTPException(
            status_code=503,
            detail=(
                "MLflow is disabled. Set MLFLOW_ENABLED=true (and optionally "
                "MLFLOW_TRACKING_URI) in the environment to use /evaluate."
            ),
        )

    from .evaluator import run_evaluation

    # Convert EvalSample objects to the dict format expected by mlflow.genai.evaluate
    mlflow_dataset = []
    for sample in req.dataset:
        entry: dict = {"inputs": {"question": sample.question}}
        expectations: dict = {}
        if sample.expected_response:
            expectations["expected_response"] = sample.expected_response
        if sample.expected_facts:
            expectations["expected_facts"] = sample.expected_facts
        if expectations:
            entry["expectations"] = expectations
        mlflow_dataset.append(entry)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _eval_executor,
            lambda: run_evaluation(
                dataset=mlflow_dataset,
                scorer_names=req.scorers,
                judge_model=req.judge_model,
                experiment_name=req.experiment_name,
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return EvalResponse(**result)
