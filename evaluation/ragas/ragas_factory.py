# evaluation/ragas/ragas_factory.py
"""
Factory for Ragas-compatible judge LLM and embeddings clients.

This module provides async clients and Ragas instances for evaluation metrics.
It supports independent configuration of judge LLM and judge embeddings providers,
allowing flexible mixing (e.g., Ollama LLM + OpenAI embeddings).

Usage:
    from evaluation.ragas.ragas_factory import get_ragas_judge_llm, get_ragas_judge_embeddings
    
    llm = get_ragas_judge_llm()
    embeddings = get_ragas_judge_embeddings()
"""

from __future__ import annotations

import sys
import os

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

from app.config import (
    JUDGE_PROVIDER,
    JUDGE_LLM_MODEL,
    JUDGE_LLM_API_KEY,
    JUDGE_LLM_BASE_URL,
    JUDGE_EMBEDDING_PROVIDER,
    JUDGE_EMBEDDING_MODEL,
    JUDGE_EMBEDDING_API_KEY,
    JUDGE_EMBEDDING_BASE_URL,
)


# Gemini OpenAI-compatible endpoint for Ragas
GEMINI_OPENAI_COMPAT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def _normalize_base_url(base_url: str | None) -> str:
    """Remove trailing slashes from base URL."""
    return (base_url or "").rstrip("/")


def _ensure_v1_suffix(base_url: str) -> str:
    """Ensure base URL ends with /v1 for OpenAI-compatible endpoints."""
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def get_async_judge_llm_client() -> AsyncOpenAI:
    """Return an async OpenAI-compatible client for judge LLM.
    
    Returns:
        AsyncOpenAI: Async client configured for the judge provider.
        
    Raises:
        ValueError: If provider is unsupported or required API key is missing.
    """
    provider = JUDGE_PROVIDER.lower()
    
    if provider == "ollama":
        base_url = _normalize_base_url(JUDGE_LLM_BASE_URL or "http://localhost:11434")
        return AsyncOpenAI(
            api_key="ollama",  # Ollama doesn't require real API key
            base_url=_ensure_v1_suffix(base_url)
        )
    
    if provider == "openai":
        if not JUDGE_LLM_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when JUDGE_PROVIDER=openai")
        
        kwargs: dict[str, str] = {"api_key": JUDGE_LLM_API_KEY}
        base_url = _normalize_base_url(JUDGE_LLM_BASE_URL)
        if base_url:
            kwargs["base_url"] = base_url
        
        return AsyncOpenAI(**kwargs)
    
    if provider == "gemini":
        if not JUDGE_LLM_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when JUDGE_PROVIDER=gemini")
        
        # Use Gemini's OpenAI-compatible endpoint
        base_url = _normalize_base_url(JUDGE_LLM_BASE_URL) or GEMINI_OPENAI_COMPAT_BASE_URL
        return AsyncOpenAI(api_key=JUDGE_LLM_API_KEY, base_url=base_url)
    
    raise ValueError(f"Unsupported JUDGE_PROVIDER: {JUDGE_PROVIDER!r}")


def get_async_judge_embeddings_client() -> AsyncOpenAI:
    """Return an async OpenAI-compatible client for judge embeddings.
    
    Returns:
        AsyncOpenAI: Async client configured for the judge embedding provider.
        
    Raises:
        ValueError: If provider is unsupported or required API key is missing.
    """
    provider = JUDGE_EMBEDDING_PROVIDER.lower()
    
    if provider == "ollama":
        base_url = _normalize_base_url(JUDGE_EMBEDDING_BASE_URL or "http://localhost:11434")
        return AsyncOpenAI(
            api_key="ollama",
            base_url=_ensure_v1_suffix(base_url)
        )
    
    if provider == "openai":
        if not JUDGE_EMBEDDING_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when JUDGE_EMBEDDING_PROVIDER=openai")
        
        kwargs: dict[str, str] = {"api_key": JUDGE_EMBEDDING_API_KEY}
        base_url = _normalize_base_url(JUDGE_EMBEDDING_BASE_URL)
        if base_url:
            kwargs["base_url"] = base_url
        
        return AsyncOpenAI(**kwargs)
    
    if provider == "gemini":
        if not JUDGE_EMBEDDING_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when JUDGE_EMBEDDING_PROVIDER=gemini")
        
        base_url = _normalize_base_url(JUDGE_EMBEDDING_BASE_URL) or GEMINI_OPENAI_COMPAT_BASE_URL
        return AsyncOpenAI(api_key=JUDGE_EMBEDDING_API_KEY, base_url=base_url)
    
    raise ValueError(f"Unsupported JUDGE_EMBEDDING_PROVIDER: {JUDGE_EMBEDDING_PROVIDER!r}")


def _resolve_judge_llm_model() -> str:
    """Resolve the judge LLM model name with provider-specific defaults."""
    provider = JUDGE_PROVIDER.lower()
    model = (JUDGE_LLM_MODEL or "").strip()
    
    if model:
        return model
    
    # Provider-specific defaults when not configured
    defaults = {
        "ollama": "phi3.5",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.0-flash",
    }
    
    return defaults.get(provider, "gpt-4o-mini")


def _resolve_judge_embedding_model() -> str:
    """Resolve the judge embedding model name with provider-specific defaults."""
    provider = JUDGE_EMBEDDING_PROVIDER.lower()
    model = (JUDGE_EMBEDDING_MODEL or "").strip()
    
    if model:
        return model
    
    # Provider-specific defaults when not configured
    defaults = {
        "ollama": "nomic-embed-text",
        "openai": "text-embedding-3-small",
        "gemini": "text-embedding-004",
    }
    
    return defaults.get(provider, "text-embedding-3-small")


def get_ragas_judge_llm():
    """Return a Ragas LLM instance for evaluation metrics.
    
    Returns:
        Ragas LLM instance configured with the judge provider and model.
        
    Example:
        >>> llm = get_ragas_judge_llm()
        >>> # Use with Ragas metrics
        >>> faithfulness = Faithfulness(llm=llm)
    """
    client = get_async_judge_llm_client()
    model = _resolve_judge_llm_model()
    
    # Ragas llm_factory expects model as first positional arg, provider as keyword
    # All providers use OpenAI-compatible endpoints (Ollama and Gemini via compatibility layer)
    return llm_factory(model, provider="openai", client=client)


def get_ragas_judge_embeddings():
    """Return a Ragas embeddings instance for evaluation metrics.
    
    Returns:
        Ragas embeddings instance configured with the judge embedding provider and model.
        
    Example:
        >>> embeddings = get_ragas_judge_embeddings()
        >>> # Use with Ragas metrics
        >>> answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    """
    client = get_async_judge_embeddings_client()
    model = _resolve_judge_embedding_model()
    
    # Use OpenAIEmbeddings directly (works with OpenAI-compatible endpoints)
    return OpenAIEmbeddings(client=client, model=model)
