# app/factory.py
"""
Central factory for LLM and Embeddings instances.

Import get_llm() / get_embeddings() everywhere instead of instantiating
provider-specific classes directly.  Adding a new provider only requires
changes here and in config.py.

LLM and Embedding providers are fully independent — any combination works.
"""

from .config import (
    JUDGE_LLM_MODEL, LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL,
    EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_BASE_URL, JUDGE_LLM_MODEL
)


def get_llm():
    """Return an LLM instance for the configured LLM_PROVIDER."""

    if LLM_PROVIDER == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=LLM_MODEL, base_url=LLM_BASE_URL)

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


def get_embeddings():
    """Return an Embeddings instance for the configured EMBEDDING_PROVIDER."""

    if EMBEDDING_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=EMBEDDING_BASE_URL)

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

def get_judge_llm():
    """Return an LLM instance for evaluation, separate from the main LLM used in the app."""
    # For simplicity, we'll use the same provider as the main LLM, but with a different model.
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=JUDGE_LLM_MODEL, base_url=LLM_BASE_URL)

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        kwargs: dict = {"model": JUDGE_LLM_MODEL, "api_key": LLM_API_KEY}
        if LLM_BASE_URL:
            kwargs["base_url"] = LLM_BASE_URL
        return ChatOpenAI(**kwargs)

    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=JUDGE_LLM_MODEL, google_api_key=LLM_API_KEY)

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER!r}")