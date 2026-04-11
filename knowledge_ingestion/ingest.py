# ingest_pipeline/ingest.py
"""
Main ingestion entry point.

Pipeline summary
────────────────
  Stage 1 — Walk:    Discover every file under DATA_ROOT.
  Stage 2 — Route:   Dispatch each file to the correct parser by extension.
                     Unsupported / metadata-only files are silently skipped.
  Stage 3 — Chunk:   Split each parsed Document into retrieval-ready chunks
                     with heading breadcrumbs and size enforcement.
  Stage 4 — Embed:   Add chunks to ChromaDB in batches.
                     ChromaDB persists to a local SQLite file (chroma.sqlite3)
                     inside CHROMA_PATH — no external service needed.

Usage
─────
  # From the workspace root (with the venv active):
  python -m ingest_pipeline.ingest

  # Or specify a different content root:
  DATA_ROOT=./refined-content python -m ingest_pipeline.ingest
"""

import hashlib
import logging
import time
from collections import Counter
from pathlib import Path

from app.vectorstore import get_vectorstore
from langchain_chroma import Chroma
from tqdm import tqdm

from app.ingest_pipeline.chunker import chunk_documents
from app.config  import (
    BATCH_SIZE, CHROMA_TARGET, CHUNK_OVERLAP, CHUNK_SIZE,
    COLLECTION_NAME, DATA_ROOT, EMBEDDING_MODEL, EMBEDDING_PROVIDER,
)
from app.factory import get_embeddings
from app.ingest_pipeline.router  import route_file, walk_data_root

LOG_FILE = "ingest_pipeline.log"


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("ingest_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    return logger

# Other 

def generate_doc_id(content: str) -> str:
    """Generate a stable ID based on document content."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()
    
# ── Main ──────────────────────────────────────────────────────────────────────

def ingest() -> None:
    log = _setup_logger()
    t_start = time.time()

    data_root = Path(DATA_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(
            f"DATA_ROOT '{data_root}' does not exist.  "
            "Set the DATA_ROOT environment variable or update config.py."
        )

    # ── Stage 1: Walk ─────────────────────────────────────────────────────────
    all_files = walk_data_root(data_root)
    print(f"\n[1/4] Discovered {len(all_files):,} files under '{data_root}'")
    log.info(f"DATA_ROOT={data_root!r}  total_files={len(all_files)}")

    # ── Stage 2: Parse ────────────────────────────────────────────────────────
    print("[2/4] Parsing files (routing by extension) ...")
    raw_docs = []
    skipped_count = 0
    format_counter: Counter = Counter()

    for file_path in tqdm(all_files, desc="  Parsing", unit="file", ncols=80):
        docs = route_file(file_path)
        if not docs:
            skipped_count += 1
            continue
        for doc in docs:
            format_counter[doc.metadata.get("format", "?")] += 1
        raw_docs.extend(docs)

    print(f"    → {len(raw_docs):,} raw documents from {len(all_files) - skipped_count:,} files "
          f"({skipped_count:,} skipped)")
    print(f"    → Breakdown by format: {dict(format_counter)}")
    log.info(f"Parsed {len(raw_docs)} raw documents.  Skipped: {skipped_count}.  "
             f"Format counts: {dict(format_counter)}")

    if not raw_docs:
        print("    ⚠  No documents to ingest.  Check DATA_ROOT and file types.")
        return

    # ── Stage 3: Chunk ────────────────────────────────────────────────────────
    print(f"[3/4] Chunking (size={CHUNK_SIZE} chars, overlap={CHUNK_OVERLAP} chars) ...")
    chunks = chunk_documents(raw_docs)

    content_type_counter: Counter = Counter(
        c.metadata.get("content_type", "?") for c in chunks
    )
    corpus_counter: Counter = Counter(
        c.metadata.get("source_corpus", "?") for c in chunks
    )

    print(f"    → {len(chunks):,} chunks total")
    print(f"    → By content type: {dict(content_type_counter)}")
    print(f"    → By corpus:       {dict(corpus_counter)}")
    log.info(f"Produced {len(chunks)} chunks.  "
             f"content_type={dict(content_type_counter)}  corpus={dict(corpus_counter)}")

    # ── Stage 4: Embed & store ────────────────────────────────────────────────
    # ── Vector store: ChromaDB (local SQLite backend) ─────────────────────────
    # All document types — narrative prose (MDX, MD, PDF) and code (notebooks,
    # Python) — land in the SAME collection.
    #
    # FUTURE EXPANSION TO MULTIPLE COLLECTIONS:
    #   Replace the single `vectorstore` below with a factory that returns a
    #   different Chroma collection keyed by (corpus, content_type):
    #
    #       stores = {
    #           ("qiskit",  "narrative"): Chroma(..., collection_name="qiskit_narrative"),
    #           ("qiskit",  "code"):      Chroma(..., collection_name="qiskit_code"),
    #           ("mlflow",  "narrative"): Chroma(..., collection_name="mlflow_narrative"),
    #           ...
    #       }
    #
    #   Then during retrieval, route the user query to the relevant collection(s)
    #   first (e.g. "show me code" → code collections only) before doing a merged
    #   re-rank.  Benefits:
    #     • Sharper ANN search — code and prose live in separate embedding spaces.
    #     • Per-corpus chatbots skip irrelevant domains entirely.
    #     • Each collection can use the best embedding model for its content type
    #       (e.g. nomic-embed-code for code, nomic-embed-text for prose).

    print(f"[4/4] Embedding chunks (this is the slow part) ...")
    log.info(
        f"Starting embedding: provider={EMBEDDING_PROVIDER!r} model={EMBEDDING_MODEL!r} "
        f"collection={COLLECTION_NAME!r} chroma_target={CHROMA_TARGET!r}"
    )
    print(
        f"    Provider: {EMBEDDING_PROVIDER} | Model: {EMBEDDING_MODEL} | "
        f"Collection: {COLLECTION_NAME} | Chroma: {CHROMA_TARGET}"
    )

    # embeddings = get_embeddings()
    vectorstore = get_vectorstore()

    batches = [chunks[i : i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
    total_batches = len(batches)
    seen_ids: set[str] = set()
    skipped_dupes = 0
    failed_docs = 0

    with tqdm(
        total=len(chunks),
        unit="chunk",
        desc="  Embedding",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        for i, batch in enumerate(batches):
            t_batch = time.time()
            # Assign deterministic IDs and drop duplicates within this run
            ids: list[str] = []
            docs_to_add = []
            for doc in batch:
                doc_id = generate_doc_id(doc.page_content)
                if doc_id in seen_ids:
                    skipped_dupes += 1
                    continue
                seen_ids.add(doc_id)
                ids.append(doc_id)
                docs_to_add.append(doc)

            if not docs_to_add:
                log.info(f"Batch {i+1}/{total_batches}: all {len(batch)} chunks skipped (duplicate IDs)")
                pbar.update(len(batch))
                continue

            try:
                vectorstore.add_documents(docs_to_add, ids=ids)
            except Exception as exc:
                # Fall back to per-document ingest to isolate failures
                log.exception(
                    f"Batch {i+1}/{total_batches}: batch add failed, retrying per-doc. "
                    f"error={exc!r}"
                )
                for doc, doc_id in zip(docs_to_add, ids):
                    try:
                        vectorstore.add_documents([doc], ids=[doc_id])
                    except Exception as doc_exc:
                        failed_docs += 1
                        log.exception(
                            "Doc ingest failed; skipping. "
                            f"id={doc_id} source={doc.metadata.get('source_file','?')} "
                            f"error={doc_exc!r}"
                        )
            elapsed = time.time() - t_batch
            log.info(
                f"Batch {i+1}/{total_batches}: {len(docs_to_add)} added, "
                f"{len(batch) - len(docs_to_add)} skipped in {elapsed:.1f}s "
                f"(avg {elapsed / max(len(docs_to_add), 1):.2f}s/chunk)"
            )
            pbar.update(len(batch))

    total_time = time.time() - t_start
    summary = (
        f"Ingestion complete: {len(chunks):,} chunks from "
        f"{len(all_files) - skipped_count:,} files in {total_time:.1f}s "
        f"(skipped {skipped_dupes:,} duplicate chunks, "
        f"failed {failed_docs:,} chunks)"
    )
    print(f"\n    ✓ {summary}")
    print(f"    Log saved to: {LOG_FILE}")
    log.info(summary)


if __name__ == "__main__":
    ingest()
