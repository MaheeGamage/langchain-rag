# app/factory.py
"""
Central factory for LLM and Embeddings instances.

Import get_llm() / get_embeddings() everywhere instead of instantiating
provider-specific classes directly.  Adding a new provider only requires
changes here and in config.py.

LLM and Embedding providers are fully independent — any combination works.
"""

import httpx

from .config import (
    JUDGE_LLM_MODEL, JUDGE_PROVIDER,
    HELPER_LLM_PROVIDER, HELPER_LLM_MODEL, HELPER_LLM_API_KEY, HELPER_LLM_BASE_URL,
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL,
    EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_BASE_URL,
    HTTP_KEEPALIVE_EXPIRY_S, HTTP_TIMEOUT_S,
)

_OLLAMA_HTTPX_LIMITS = httpx.Limits(
    max_connections=None,
    max_keepalive_connections=None,
    keepalive_expiry=HTTP_KEEPALIVE_EXPIRY_S,
)


def get_llm():
    """Return an LLM instance for the configured LLM_PROVIDER."""

    if LLM_PROVIDER == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            sync_client_kwargs={"limits": _OLLAMA_HTTPX_LIMITS},
        )

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        kwargs: dict = {"model": LLM_MODEL, "api_key": LLM_API_KEY}
        if LLM_BASE_URL:
            kwargs["base_url"] = LLM_BASE_URL
        return ChatOpenAI(**kwargs)

    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=LLM_API_KEY)

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER!r}")


def get_helper_llm():
    """Return a lightweight LLM for query rewriting, routing, and classification.

    Uses HELPER_LLM_PROVIDER / HELPER_LLM_MODEL, falling back to the main LLM when
    no dedicated helper model is configured.
    """
    if HELPER_LLM_PROVIDER == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(
            model=HELPER_LLM_MODEL,
            base_url=HELPER_LLM_BASE_URL,
            sync_client_kwargs={"limits": _OLLAMA_HTTPX_LIMITS},
        )

    if HELPER_LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        kwargs: dict = {"model": HELPER_LLM_MODEL, "api_key": HELPER_LLM_API_KEY}
        if HELPER_LLM_BASE_URL:
            kwargs["base_url"] = HELPER_LLM_BASE_URL
        return ChatOpenAI(**kwargs)

    if HELPER_LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=HELPER_LLM_MODEL, google_api_key=HELPER_LLM_API_KEY)

    raise ValueError(f"Unsupported HELPER_LLM_PROVIDER: {HELPER_LLM_PROVIDER!r}")


def get_chat_llm():
    """Return a tool-calling chat model for the configured LLM_PROVIDER.

    Used by graph implementations that rely on native tool calling (e.g. the
    `rag_agent` graph). For Ollama this returns `ChatOllama` instead of the
    completion-style `OllamaLLM` returned by `get_llm()`.
    """
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            client_kwargs={"limits": _OLLAMA_HTTPX_LIMITS},
        )

    return get_llm()


def get_embeddings():
    """Return an Embeddings instance for the configured EMBEDDING_PROVIDER."""

    if EMBEDDING_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=EMBEDDING_BASE_URL,
            sync_client_kwargs={"limits": _OLLAMA_HTTPX_LIMITS},
        )

    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        kwargs: dict = {"model": EMBEDDING_MODEL, "api_key": EMBEDDING_API_KEY}
        if EMBEDDING_BASE_URL:
            kwargs["base_url"] = EMBEDDING_BASE_URL
        return OpenAIEmbeddings(**kwargs)

    if EMBEDDING_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, google_api_key=EMBEDDING_API_KEY
        )

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER!r}")


def get_judge_model_uri() -> str:
    """Return an MLflow judge model URI string for use with built-in scorers.

    MLflow scorers (e.g. Correctness) accept a ``model`` parameter in the form
    ``"<provider>:/<model-name>"``:
    - ``openai:/gpt-4o-mini``  — routed via MLflow's native OpenAI adapter
    - ``gemini:/gemini-2.5-flash``  — routed via LiteLLM (requires ``pip install litellm``)
    - ``ollama:/phi3.5``  — routed via LiteLLM (requires ``pip install litellm``)
    """
    judge_model = JUDGE_LLM_MODEL
    judge_model_uri = f"{JUDGE_PROVIDER}:/{judge_model}"
    print(f"Using judge model: {judge_model_uri}")
    return judge_model_uri