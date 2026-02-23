# streamlit_app.py
#
# Entry point for the Streamlit UI.
# Run with:  streamlit run streamlit_app.py

import streamlit as st
from app.graph import graph
from app.config import LLM_MODEL, EMBEDDING_MODEL

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat",
    page_icon="🔍",
    layout="centered",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("RAG Chat")
    st.caption("Ask questions about your ingested documents.")

    st.divider()

    st.markdown("**Models**")
    st.markdown(f"- LLM: `{LLM_MODEL}`")
    st.markdown(f"- Embeddings: `{EMBEDDING_MODEL}`")

    st.divider()

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Session state ──────────────────────────────────────────────────────────────
# Each message: {"role": "user"|"assistant", "content": str, "sources": list}
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Helper ─────────────────────────────────────────────────────────────────────
def _render_sources(docs: list) -> None:
    """Render retrieved source documents inside an expander."""
    with st.expander(f"📄 Sources ({len(docs)} chunks)", expanded=False):
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page")
            label = f"**[{i}]** {source}"
            if page is not None:
                label += f" — page {int(page) + 1}"
            st.markdown(label)
            st.caption(doc.page_content[:400])
            if i < len(docs):
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

    # 2. Run the RAG graph and display the answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            result = graph.invoke({"question": prompt})

        answer = result.get("answer", "No answer returned.")
        sources = result.get("documents", [])

        st.markdown(answer)
        if sources:
            _render_sources(sources)

    # 3. Persist assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
