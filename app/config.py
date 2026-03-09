# app/config.py

import os
from dotenv import load_dotenv

load_dotenv()

_VALID_PROVIDERS = ("ollama", "openai", "gemini")


def _parse_bool(name: str, default: str = "false") -> bool:
    raw = os.getenv(name, default).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {name}={raw!r}. Use true/false.")


def _parse_int(name: str, default: str) -> int:
    raw = os.getenv(name, default).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw!r}. Use an integer.") from exc

# ── Independent provider switches ─────────────────────────────────────────────
# Set these in .env.  They are fully independent — mix any combination.
# LLM_PROVIDER       controls which service answers questions.
# EMBEDDING_PROVIDER controls which service encodes text into vectors.
LLM_PROVIDER:       str = os.getenv("LLM_PROVIDER",       "ollama")
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "ollama")

for _p, _name in ((LLM_PROVIDER, "LLM_PROVIDER"), (EMBEDDING_PROVIDER, "EMBEDDING_PROVIDER")):
    if _p not in _VALID_PROVIDERS:
        raise ValueError(f"Unknown {_name}={_p!r}. Choose from: {list(_VALID_PROVIDERS)}")

# ── Chroma HTTP connection ───────────────────────────────────────────────────
# Defaults target a local-development host process (API runs on host, Chroma in Docker).
# Docker compose overrides these for container-to-container networking.
CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost").strip()
CHROMA_PORT: int = _parse_int("CHROMA_PORT", "8001")
CHROMA_SSL: bool = _parse_bool("CHROMA_SSL", "false")

# ── LLM defaults — keyed by LLM_PROVIDER ─────────────────────────────────────
_LLM_DEFAULTS: dict = {
    "ollama": {
        "model":   os.getenv("OLLAMA_LLM_MODEL",  "tinyllama"),
        "api_key": None,
        "base_url": os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434"),
    },
    "openai": {
        "model":   os.getenv("OPENAI_LLM_MODEL",  "gpt-4o-mini"),
        "api_key": os.getenv("OPENAI_API_KEY",    ""),
        "base_url": os.getenv("OPENAI_BASE_URL",  None),  # None = official endpoint
    },
    "gemini": {
        "model":   os.getenv("GEMINI_LLM_MODEL",  "gemini-2.5-flash"),
        "api_key": os.getenv("GEMINI_API_KEY",    ""),
        "base_url": None,
    },
}

# ── Embedding defaults — keyed by EMBEDDING_PROVIDER ─────────────────────────
_EMBEDDING_DEFAULTS: dict = {
    "ollama": {
        "model":   os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        "api_key": None,
        "base_url": os.getenv("OLLAMA_BASE_URL",       "http://localhost:11434"),
    },
    "openai": {
        "model":   os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "api_key": os.getenv("OPENAI_API_KEY",         ""),
        "base_url": os.getenv("OPENAI_BASE_URL",       None),
    },
    "gemini": {
        "model":   os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        "api_key": os.getenv("GEMINI_API_KEY",         ""),
        "base_url": None,
    },
}

# ── Resolved LLM values ───────────────────────────────────────────────────────
_llm = _LLM_DEFAULTS[LLM_PROVIDER]
LLM_MODEL:    str       = _llm["model"]
LLM_API_KEY:  str | None = _llm["api_key"]
LLM_BASE_URL: str | None = _llm["base_url"]

# ── Resolved Embedding values ─────────────────────────────────────────────────
_emb = _EMBEDDING_DEFAULTS[EMBEDDING_PROVIDER]
EMBEDDING_MODEL:    str       = _emb["model"]
EMBEDDING_API_KEY:  str | None = _emb["api_key"]
EMBEDDING_BASE_URL: str | None = _emb["base_url"]

# ChromaDB collection is named after the embedding model so that embeddings
# from different providers/models live in separate collections and are never
# mixed.  Switching EMBEDDING_PROVIDER automatically targets the right collection.
COLLECTION_NAME: str = (
    EMBEDDING_MODEL.replace("/", "_").replace("-", "_").replace(".", "_")
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH: str = "./data"
CONVERSATIONS_DB: str = os.getenv("CONVERSATIONS_DB", "./conversations.db")

scheme = "https" if CHROMA_SSL else "http"
CHROMA_TARGET: str = f"{scheme}://{CHROMA_HOST}:{CHROMA_PORT}"

# ── MLflow tracing ───────────────────────────────────────────────────────────
# Set MLFLOW_ENABLED=true in .env to activate automatic LangChain tracing.
# MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD are read natively by the
# mlflow library — no need to set them here.
MLFLOW_ENABLED:         bool = _parse_bool("MLFLOW_ENABLED", "false")
MLFLOW_TRACKING_URI:    str  = os.getenv("MLFLOW_TRACKING_URI",    "")
MLFLOW_EXPERIMENT_NAME: str  = os.getenv("MLFLOW_EXPERIMENT_NAME", "langchain-rag")

# ── Chunking ──────────────────────────────────────────────────────────────────
# ~500 tokens at ~4 chars/token with ~50-token overlap — standard middle ground.
# Finer → more precise retrieval but loses surrounding context.
# Coarser → richer context but lower retrieval precision.
CHUNK_SIZE:    int = 2000
CHUNK_OVERLAP: int = 200

# ── Ingestion ─────────────────────────────────────────────────────────────────
BATCH_SIZE: int = 25   # chunks per Chroma add_documents call
DATA_ROOT:   str = os.getenv("DATA_ROOT",   "./refined-content") # Ingestion data path
