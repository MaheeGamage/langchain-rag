# app/config.py

import os
from dotenv import load_dotenv

load_dotenv()

_VALID_PROVIDERS = ("ollama", "openai", "gemini")

# ── Independent provider switches ─────────────────────────────────────────────
# Set these in .env.  They are fully independent — mix any combination.
# LLM_PROVIDER       controls which service answers questions.
# EMBEDDING_PROVIDER controls which service encodes text into vectors.
LLM_PROVIDER:       str = os.getenv("LLM_PROVIDER",       "ollama")
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "ollama")

for _p, _name in ((LLM_PROVIDER, "LLM_PROVIDER"), (EMBEDDING_PROVIDER, "EMBEDDING_PROVIDER")):
    if _p not in _VALID_PROVIDERS:
        raise ValueError(f"Unknown {_name}={_p!r}. Choose from: {list(_VALID_PROVIDERS)}")

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
CHROMA_PATH: str = "./chroma_db"
DATA_PATH:   str = "./data"
