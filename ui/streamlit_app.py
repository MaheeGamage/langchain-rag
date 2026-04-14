# streamlit_app.py
#
# Pure presentation layer — all RAG work happens via the FastAPI backend.
# Run with:  streamlit run ui/streamlit_app.py

import os
import json
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


def _stream_query(question: str, thread_id: str | None = None):
    """Stream tokens from the /query/stream endpoint.
    
    Yields tuples of (content, sources, thread_id, error).
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
    
    sources = []
    new_thread_id = thread_id
    error = None
    
    try:
        resp = requests.post(
            f"{API_URL}/query/stream",
            json=payload,
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
        
        # Parse newline-delimited JSON
        for line in resp.iter_lines():
            if line:
                try:
                    event = json.loads(line)
                    event_type = event.get("type")
                    
                    if event_type == "metadata":
                        sources = event.get("sources", [])
                        new_thread_id = event.get("thread_id", thread_id)
                    elif event_type == "token":
                        content = event.get("content", "")
                        yield content, None, None, None  # Yield only token
                    elif event_type == "error":
                        error = event.get("content", "Unknown error")
                        yield "", sources, new_thread_id, error
                        break
                except json.JSONDecodeError:
                    continue
        
        # Final yield with complete metadata
        yield "", sources, new_thread_id, error
        
    except requests.exceptions.ConnectionError:
        error = "⚠️ Could not connect to the API. Is the server running?"
        yield "", [], thread_id, error
    except Exception as e:
        error = f"⚠️ API error: {e}"
        yield "", [], thread_id, error


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
        message_placeholder = st.empty()
        sources_placeholder = st.empty()
        
        # Show loading status while waiting for retrieval
        with message_placeholder.container():
            st.markdown("⏳ Retrieving documents and generating response...")
        
        full_answer = ""
        sources = []
        final_thread_id = st.session_state.thread_id
        error = None
        
        try:
            for token, chunk_sources, chunk_thread_id, chunk_error in _stream_query(prompt, thread_id=st.session_state.thread_id):
                if chunk_sources is not None:
                    sources = chunk_sources
                    # Clear the loading message once we have metadata
                    message_placeholder.empty()
                
                if token:
                    full_answer += token
                    message_placeholder.markdown(full_answer + "▌")  # Add cursor while streaming
                
                if chunk_thread_id is not None:
                    final_thread_id = chunk_thread_id
                
                if chunk_error:
                    error = chunk_error
            
            # Remove cursor and display final answer
            if full_answer:
                message_placeholder.markdown(full_answer)
            
            # Update thread ID
            st.session_state.thread_id = final_thread_id
            
            # Display error if any
            if error:
                st.error(error)
            
            # Display sources
            if sources:
                with sources_placeholder:
                    _render_sources(sources)
        
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            full_answer = f"⚠️ Error: {e}"
            sources = []

    # 3. Persist assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources,
    })
