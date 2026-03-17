# app/factory.py
"""
Central factory for LLM and Embeddings instances.

Import get_llm() / get_embeddings() everywhere instead of instantiating
provider-specific classes directly.  Adding a new provider only requires
changes here and in config.py.

LLM and Embedding providers are fully independent — any combination works.
"""

from .config import (
    JUDGE_LLM_MODEL, JUDGE_PROVIDER, LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL,
    EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_BASE_URL
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