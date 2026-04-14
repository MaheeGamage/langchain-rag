# pipeline/stages/embedder.py
"""
Stage 4 — Embed chunks and write them to ChromaDB.

Swap this stage to change the vector store backend (e.g. Pinecone, pgvector)
or add a second store (e.g. a BM25 index alongside Chroma).

Deduplication
─────────────
Each chunk gets a deterministic MD5 ID from its content.  Chunks whose ID
was already seen in this run are skipped before any network call is made.
This is a within-run guard; it does not check what is already in Chroma.

Error handling
──────────────
If a batch upsert fails, the stage retries per-document and skips only the
failing chunks, logging their IDs.  The run continues.
"""

import hashlib
import logging
import time
from collections import Counter

from langchain_core.documents import Document
from tqdm import tqdm

from app.vectorstore import get_vectorstore
from app.config import BATCH_SIZE, COLLECTION_NAME, CHROMA_TARGET, EMBEDDING_MODEL, EMBEDDING_PROVIDER

log = logging.getLogger("ingest_pipeline")


def _doc_id(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


class EmbedderStage:
    """
    Batch-embed Documents into a Chroma collection.

    Parameters
    ----------
    batch_size:
        Number of chunks per add_documents call.  Lower if the embedding
        provider OOMs; raise to speed up ingest.
    vectorstore_factory:
        Callable that returns a LangChain VectorStore.  Defaults to
        get_vectorstore() from app.vectorstore.  Override for testing or
        to target a different backend.
    """

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        vectorstore_factory=None,
    ) -> None:
        self.batch_size = batch_size
        self._vectorstore_factory = vectorstore_factory or get_vectorstore

    def run(self, chunks: list[Document]) -> dict:
        """
        Embed and store all chunks.

        Returns a summary dict with counts for reporting.
        """
        log.info(
            f"Starting embedding: provider={EMBEDDING_PROVIDER!r} "
            f"model={EMBEDDING_MODEL!r} collection={COLLECTION_NAME!r} "
            f"chroma={CHROMA_TARGET!r}"
        )
        print(
            f"    Provider: {EMBEDDING_PROVIDER} | Model: {EMBEDDING_MODEL} | "
            f"Collection: {COLLECTION_NAME} | Chroma: {CHROMA_TARGET}"
        )

        vectorstore = self._vectorstore_factory()
        batches = [chunks[i : i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
        total_batches = len(batches)

        seen_ids: set[str] = set()
        skipped_dupes = 0
        failed_docs = 0
        added_docs = 0

        with tqdm(
            total=len(chunks),
            unit="chunk",
            desc="  Embedding",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for i, batch in enumerate(batches):
                t_batch = time.time()

                ids: list[str] = []
                docs_to_add: list[Document] = []
                for doc in batch:
                    doc_id = _doc_id(doc.page_content)
                    if doc_id in seen_ids:
                        skipped_dupes += 1
                        continue
                    seen_ids.add(doc_id)
                    ids.append(doc_id)
                    docs_to_add.append(doc)

                if not docs_to_add:
                    log.info(f"Batch {i+1}/{total_batches}: all skipped (duplicates)")
                    pbar.update(len(batch))
                    continue

                batch_added, batch_failed = self._upsert_batch(
                    vectorstore, docs_to_add, ids, i + 1, total_batches
                )
                added_docs  += batch_added
                failed_docs += batch_failed

                elapsed = time.time() - t_batch
                log.info(
                    f"Batch {i+1}/{total_batches}: {batch_added} added, "
                    f"{batch_failed} failed in {elapsed:.1f}s"
                )
                pbar.update(len(batch))

        return {
            "added":   added_docs,
            "skipped": skipped_dupes,
            "failed":  failed_docs,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _upsert_batch(
        vectorstore,
        docs: list[Document],
        ids: list[str],
        batch_num: int,
        total_batches: int,
    ) -> tuple[int, int]:
        """Try batch upsert; fall back to per-doc on failure."""
        try:
            vectorstore.add_documents(docs, ids=ids)
            return len(docs), 0
        except Exception as exc:
            log.exception(
                f"Batch {batch_num}/{total_batches}: batch add failed, "
                f"retrying per-doc. error={exc!r}"
            )

        added = failed = 0
        for doc, doc_id in zip(docs, ids):
            try:
                vectorstore.add_documents([doc], ids=[doc_id])
                added += 1
            except Exception as doc_exc:
                failed += 1
                log.exception(
                    f"Doc ingest failed; skipping. "
                    f"id={doc_id} source={doc.metadata.get('source_file', '?')} "
                    f"error={doc_exc!r}"
                )
        return added, failed
