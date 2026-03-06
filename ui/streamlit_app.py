# streamlit_app.py
#
# Pure presentation layer — all RAG work happens via the FastAPI backend.
# Run with:  streamlit run ui/streamlit_app.py

import os
import streamlit as st
import requests

# ── API base URL ───────────────────────────────────────────────────────────────
# In Docker: API_URL=http://api:8000  (service-to-service).
# Locally:   defaults to http://localhost:8000.
API_URL = os.getenv("API_URL", "http://localhost:8000")


# ── API helpers ────────────────────────────────────────────────────────────────
def _get_config() -> dict:
    """Fetch model info from the API. Returns empty dict on failure."""
    try:
        resp = requests.get(f"{API_URL}/config", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def _query_api(question: str, thread_id: str | None = None) -> dict:
    """Send a question to the /query endpoint and return the JSON response.

    ``thread_id`` is passed to the API so the server-side checkpointer can
    retrieve and persist history. Omit on the first turn — the API will
    generate and return a new thread_id.
    """
    payload = {
        "message": question,
        "conversation": {
            "id": thread_id,
        },
        "context": {
            "entries": [],
        },
        "meta": {
            "clientId": "streamlit",
        },
    }
    resp = requests.post(
        f"{API_URL}/query",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat",
    page_icon="🔍",
    layout="centered",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
api_config = _get_config()

with st.sidebar:
    st.title("RAG Chat")
    st.caption("Ask questions about your ingested documents.")

    st.divider()

    if api_config:
        st.markdown("**Models**")
        st.markdown(f"- LLM: `{api_config.get('llm_model', '?')}`")
        st.markdown(f"- Embeddings: `{api_config.get('embedding_model', '?')}`")
    else:
        st.warning("API not reachable")

    st.divider()

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = None
        st.rerun()

# ── Session state ──────────────────────────────────────────────────────────────
# Each message: {"role": "user"|"assistant", "content": str, "sources": list}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None


# ── Helper ─────────────────────────────────────────────────────────────────────
def _render_sources(sources: list[dict]) -> None:
    """Render retrieved source documents inside an expander."""
    with st.expander(f"📄 Sources ({len(sources)} chunks)", expanded=False):
        for i, src in enumerate(sources, start=1):
            meta = src.get("metadata", {})
            source = meta.get("source", "Unknown")
            page = meta.get("page")
            label = f"**[{i}]** {source}"
            if page is not None:
                label += f" — page {int(page) + 1}"
            st.markdown(label)
            st.caption(src.get("content", "")[:400])
            if i < len(sources):
                st.divider()


# ── Page header ────────────────────────────────────────────────────────────────
st.header("🔍 RAG Chat", divider="gray")

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show source documents for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_sources(msg["sources"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents…"):

    # 1. Store & display user message
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call the API and display the answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            try:
                result = _query_api(prompt, thread_id=st.session_state.thread_id)
                answer = result.get("answer", "No answer returned.")
                sources = result.get("sources", [])
                st.session_state.thread_id = result.get("thread_id", st.session_state.thread_id)
            except requests.exceptions.ConnectionError:
                answer = "⚠️ Could not connect to the API. Is the server running?"
                sources = []
            except Exception as e:
                answer = f"⚠️ API error: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            _render_sources(sources)

    # 3. Persist assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
