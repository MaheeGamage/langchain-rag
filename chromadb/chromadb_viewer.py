import streamlit as st
import pandas as pd
import numpy as np
from chromadb import PersistentClient, HttpClient
from langchain_chroma import Chroma

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CHROMA_MODE     = "http"            # "local" or "http"
CHROMA_PATH     = "./chroma_db"     # local mode only
CHROMA_HOST     = "localhost"       # http mode only
CHROMA_PORT     = 8001              # http mode only
COLLECTION_NAME = "nomic_embed_text"

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="ChromaDB Viewer", page_icon="🗄️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a2f3e; }
    .metric-card {
        background: #161b27; border: 1px solid #2a2f3e;
        border-radius: 10px; padding: 1.2rem 1.5rem; text-align: center;
    }
    .metric-card .label { color: #6b7280; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.3rem; }
    .metric-card .value { color: #e2e8f0; font-size: 2rem; font-weight: 700; }
    .metric-card .sub   { color: #4ade80; font-size: 0.78rem; margin-top: 0.2rem; }
    .chunk-card {
        background: #161b27; border: 1px solid #2a2f3e;
        border-left: 3px solid #6366f1; border-radius: 8px;
        padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    }
    .hit-card {
        background: #161b27; border: 1px solid #2a2f3e;
        border-left: 3px solid #4ade80; border-radius: 8px;
        padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    }
    .chunk-id   { color: #6b7280; font-size: 0.72rem; font-family: monospace; }
    .chunk-meta { color: #818cf8; font-size: 0.8rem; margin: 0.4rem 0; }
    .chunk-text { color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; }
    .score-badge {
        display: inline-block; background: #14532d; color: #4ade80;
        border-radius: 4px; padding: 0.1rem 0.5rem;
        font-size: 0.75rem; font-weight: 600; font-family: monospace;
    }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab"] { color: #6b7280; }
    .stTabs [aria-selected="true"] { color: #818cf8 !important; }
    div[data-testid="stDataFrameContainer"] { border: 1px solid #2a2f3e; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def safe_emb_stats(embeddings):
    """
    Returns (missing_count, embedding_dim) safely for both:
    - Python list of lists  (local PersistentClient)
    - 2D numpy array        (HTTP client)
    - list of numpy arrays  (mixed)
    """
    if embeddings is None:
        return 0, 0
    arr = np.array(embeddings)          # always convert → avoids bool ambiguity
    if arr.ndim == 2 and arr.shape[0] > 0:
        return 0, int(arr.shape[1])     # 2D: (n_docs, dim) — all present
    if arr.ndim == 1:
        # list of None / variable-length entries
        missing = sum(1 for e in embeddings if e is None or len(e) == 0)
        dim = next((len(e) for e in embeddings if e is not None and len(e) > 0), 0)
        return missing, dim
    return 0, 0

# ─── DB CONNECTION ────────────────────────────────────────────────────────────
@st.cache_resource
def load_collection():
    if CHROMA_MODE == "http":
        client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    else:
        client = PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)

@st.cache_data(ttl=30)
def fetch_all(_collection):
    if _collection.count() == 0:
        return None
    return _collection.get(include=["documents", "metadatas", "embeddings"])

# ─── INIT ─────────────────────────────────────────────────────────────────────
sidebar_ok      = False
data            = None
collection      = None
selected_source = "All sources"

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗄️ ChromaDB Viewer")
    mode_label = f"`{CHROMA_HOST}:{CHROMA_PORT}`" if CHROMA_MODE == "http" else f"`{CHROMA_PATH}`"
    st.caption(f"{mode_label} › `{COLLECTION_NAME}`")
    st.divider()

    try:
        collection = load_collection()
        data       = fetch_all(collection)
        sidebar_ok = True
    except Exception as e:
        st.error(f"Failed to connect:\n\n{e}")

    if sidebar_ok and data:
        sources = list({m.get("source", "unknown") for m in data["metadatas"]})
        selected_source = st.selectbox(
            "Filter by source", ["All sources"] + sorted(sources)
        )
        st.divider()
        st.caption("📊 Summary · 📄 Browse · 🔍 Query · 📈 Sources")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown("# 🗄️ ChromaDB Inspector")

if not sidebar_ok or data is None:
    st.warning("No data found or connection failed. Check your config at the top of the file.")
    if CHROMA_MODE == "http":
        st.code(f"# Make sure Chroma server is running:\nchroma run --path ./your_db --port {CHROMA_PORT}")
    st.stop()

# ─── FILTER ──────────────────────────────────────────────────────────────────
if selected_source != "All sources":
    indices  = [i for i, m in enumerate(data["metadatas"]) if m.get("source") == selected_source]
    filtered = {k: [data[k][i] for i in indices] for k in ["ids", "documents", "metadatas"]}
    # embeddings may be numpy array — index with list is fine
    raw_emb = data["embeddings"]
    filtered["embeddings"] = [raw_emb[i] for i in indices] if raw_emb is not None else []
else:
    filtered = data

total       = len(filtered["ids"])
sources_map = {}
for m in filtered["metadatas"]:
    s = m.get("source", "unknown")
    sources_map[s] = sources_map.get(s, 0) + 1

missing_emb, emb_dim = safe_emb_stats(filtered["embeddings"])
has_fp = sum(1 for m in filtered["metadatas"] if "fingerprint" in m)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📄 Browse Chunks", "🔍 Similarity Query", "📈 Sources Breakdown"])

# ── TAB 1: SUMMARY ───────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Collection Overview")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in [
        (c1, "Total Chunks",     total,            "documents stored"),
        (c2, "Unique Sources",   len(sources_map), "files / origins"),
        (c3, "Embedding Dim",    emb_dim,          "vector dimensions"),
        (c4, "With Fingerprint", has_fp,           f"of {total} chunks"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Metadata Keys Found")
        all_keys = set()
        for m in filtered["metadatas"]:
            all_keys.update(m.keys())
        key_counts = {k: sum(1 for m in filtered["metadatas"] if k in m) for k in sorted(all_keys)}
        df_keys = pd.DataFrame(key_counts.items(), columns=["Key", "Docs with key"])
        df_keys["Coverage"] = (df_keys["Docs with key"] / total * 100).round(1).astype(str) + "%"
        st.dataframe(df_keys, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### Embedding Health")
        st.dataframe(pd.DataFrame({
            "Check":  ["Total chunks", "Valid embeddings", "Missing embeddings", "Embedding dim"],
            "Value":  [total, total - missing_emb, missing_emb, emb_dim],
            "Status": ["ℹ️",
                       "✅" if missing_emb == 0 else "⚠️",
                       "✅" if missing_emb == 0 else "❌",
                       "✅" if emb_dim > 0 else "❌"],
        }), use_container_width=True, hide_index=True)

# ── TAB 2: BROWSE ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Browse Chunks")
    col1, col2 = st.columns([3, 1])
    with col1:
        search_text = st.text_input("🔎 Filter by content (substring)", placeholder="keyword...")
    with col2:
        page_size = st.selectbox("Per page", [5, 10, 25], index=1)

    display_indices = list(range(total))
    if search_text:
        display_indices = [i for i in display_indices
                           if search_text.lower() in filtered["documents"][i].lower()]

    total_pages = max(1, (len(display_indices) + page_size - 1) // page_size)
    page       = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1
    page_slice = display_indices[page * page_size:(page + 1) * page_size]

    st.caption(f"Showing {len(page_slice)} of {len(display_indices)} chunks")
    for i in page_slice:
        meta     = filtered["metadatas"][i]
        text     = filtered["documents"][i]
        meta_str = " · ".join(f"<b>{k}</b>: {v}" for k, v in meta.items() if k != "fingerprint")
        fp_str   = (f"<br><span style='color:#374151;font-size:0.7rem;font-family:monospace'>"
                    f"fp: {meta['fingerprint']}</span>") if "fingerprint" in meta else ""
        st.markdown(f"""
        <div class="chunk-card">
            <div class="chunk-id">ID: {filtered['ids'][i]}</div>
            <div class="chunk-meta">{meta_str}{fp_str}</div>
            <div class="chunk-text">{text[:500]}{"..." if len(text) > 500 else ""}</div>
        </div>""", unsafe_allow_html=True)

# ── TAB 3: SIMILARITY QUERY ──────────────────────────────────────────────────
with tab3:
    st.markdown("### Similarity Search")
    st.info("Configure your embeddings in `get_vectorstore()` inside the file.", icon="ℹ️")

    @st.cache_resource
    def get_vectorstore():
        # ── Uncomment your embeddings ─────────────────────────────────────────
        # from langchain_ollama import OllamaEmbeddings
        # embeddings = OllamaEmbeddings(model="nomic-embed-text")
        #
        # from langchain_openai import OpenAIEmbeddings
        # embeddings = OpenAIEmbeddings(api_key="sk-...")
        # ──────────────────────────────────────────────────────────────────────
        raise NotImplementedError("Configure your embeddings in get_vectorstore()")

        if CHROMA_MODE == "http":
            return Chroma(
                client=HttpClient(host=CHROMA_HOST, port=CHROMA_PORT),
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
            )
        return Chroma(
            persist_directory=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )

    query_text = st.text_area("Enter your query", placeholder="What is...", height=80)
    k          = st.slider("Top K results", min_value=1, max_value=20, value=5)

    if st.button("🔍 Search", type="primary") and query_text:
        try:
            results = get_vectorstore().similarity_search_with_score(query_text, k=k)
            st.markdown(f"#### Top {len(results)} Results")
            for rank, (doc, score) in enumerate(results, 1):
                meta_str = " · ".join(f"<b>{mk}</b>: {mv}"
                                      for mk, mv in doc.metadata.items() if mk != "fingerprint")
                st.markdown(f"""
                <div class="hit-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
                        <span style="color:#6b7280;font-size:0.8rem">#{rank}</span>
                        <span class="score-badge">score: {score:.4f}</span>
                    </div>
                    <div class="chunk-meta">{meta_str}</div>
                    <div class="chunk-text">{doc.page_content[:600]}{"..." if len(doc.page_content) > 600 else ""}</div>
                </div>""", unsafe_allow_html=True)
        except NotImplementedError:
            st.error("Please configure your embeddings in `get_vectorstore()`.")
        except Exception as e:
            st.error(f"Query failed: {e}")

# ── TAB 4: SOURCES BREAKDOWN ─────────────────────────────────────────────────
with tab4:
    st.markdown("### Sources Breakdown")
    df_sources = pd.DataFrame(
        [(s, c, round(c / total * 100, 1)) for s, c in sorted(sources_map.items(), key=lambda x: -x[1])],
        columns=["Source", "Chunks", "Share (%)"]
    )
    st.dataframe(df_sources, use_container_width=True, hide_index=True)
    st.markdown("#### Chunk Distribution")
    st.bar_chart(df_sources.set_index("Source")["Chunks"])

    st.markdown("#### Inspect a Source")
    pick   = st.selectbox("Select source", sorted(sources_map.keys()), key="source_pick")
    sample = [(filtered["ids"][i], filtered["metadatas"][i], filtered["documents"][i])
              for i, m in enumerate(filtered["metadatas"]) if m.get("source") == pick]
    st.caption(f"{len(sample)} chunks from this source")
    for doc_id, meta, text in sample[:5]:
        with st.expander(f"Chunk `{doc_id[:32]}...`"):
            st.json(meta)
            st.text(text[:800])