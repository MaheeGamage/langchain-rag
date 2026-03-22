from __future__ import annotations

import os

from openai import AsyncOpenAI, OpenAI

from app.config import (
    JUDGE_LLM_MODEL,
    JUDGE_LLM_API_KEY,
    JUDGE_LLM_BASE_URL,
    JUDGE_PROVIDER,
    LLM_BASE_URL,
)


GEMINI_OPENAI_COMPAT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_RAGAS_DEFAULT_MODEL = "gemini-2.0-flash"


def _normalize_base_url(base_url: str | None) -> str:
    return (base_url or "").rstrip("/")


def _ensure_v1_suffix(base_url: str) -> str:
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def resolve_judge_model(configured_model: str | None = None) -> str:
    """Return a judge model compatible with the active judge provider."""
    provider = JUDGE_PROVIDER.lower()
    model = (configured_model or JUDGE_LLM_MODEL or "").strip()

    if provider != "gemini":
        return model or "phi3.5"

    # Ragas Gemini integration expects Gemini models (gemini-*). Gemma models
    # can fail with errors like "Developer instruction is not enabled".
    if not model or model.lower().startswith("gemma"):
        return os.getenv("GEMINI_RAGAS_JUDGE_MODEL", GEMINI_RAGAS_DEFAULT_MODEL).strip()

    return model


def get_judge_client() -> tuple[OpenAI, str]:
    """Return a configured judge client and provider for ragas.llm_factory.

    Returns:
        tuple[OpenAI, str]:
            - OpenAI-compatible client
            - provider string for llm_factory
    """
    provider = JUDGE_PROVIDER.lower()

    if provider == "ollama":
        base_url = _normalize_base_url(JUDGE_LLM_BASE_URL or LLM_BASE_URL or "http://localhost:11434")
        client = OpenAI(api_key="ollama", base_url=_ensure_v1_suffix(base_url))
        return client, "openai"

    if provider == "openai":
        if not JUDGE_LLM_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when JUDGE_PROVIDER=openai")

        kwargs: dict[str, str] = {"api_key": JUDGE_LLM_API_KEY}
        base_url = _normalize_base_url(JUDGE_LLM_BASE_URL)
        if base_url:
            kwargs["base_url"] = base_url

        return OpenAI(**kwargs), "openai"

    if provider == "gemini":
        if not JUDGE_LLM_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when JUDGE_PROVIDER=gemini")

        # Use Gemini's OpenAI-compatible endpoint so ragas can use a single client interface.
        base_url = _normalize_base_url(JUDGE_LLM_BASE_URL) or GEMINI_OPENAI_COMPAT_BASE_URL
        return OpenAI(api_key=JUDGE_LLM_API_KEY, base_url=base_url), "openai"

    raise ValueError(f"Unsupported JUDGE_PROVIDER: {JUDGE_PROVIDER!r}")


def get_ragas_async_judge_setup(configured_model: str | None = None) -> tuple[AsyncOpenAI, str, str]:
    """Return async client, provider, and model ready for ragas.llm_factory.

    This helper is additive and does not change existing judge-client APIs.
    """
    client, provider = get_judge_client(async_client=True)
    model = resolve_judge_model(configured_model)

    # Keep a strict async return type for callers that use await-based metrics.
    if not isinstance(client, AsyncOpenAI):
        raise TypeError("Expected AsyncOpenAI client when async_client=True")

    return client, provider, model
