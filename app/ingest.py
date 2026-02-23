# app/ingest.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from tqdm import tqdm
import os
import logging
import time
from .config import EMBEDDING_MODEL, CHROMA_PATH, DATA_PATH, COLLECTION_NAME, EMBEDDING_PROVIDER
from .factory import get_embeddings

BATCH_SIZE = 25  # chunks per embedding call
LOG_FILE = "ingest.log"


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("ingest")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def ingest():
    log = _setup_logger()
    t_start = time.time()

    # ── 1. Load PDFs ────────────────────────────────────────────────────────
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    print(f"[1/4] Loading {len(pdf_files)} PDF file(s) from '{DATA_PATH}' ...")
    log.info(f"Found PDFs: {pdf_files}")

    documents = []
    for file in pdf_files:
        path = os.path.join(DATA_PATH, file)
        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)
        log.info(f"  Loaded '{file}': {len(docs)} pages")

    print(f"    → {len(documents)} pages loaded")
    log.info(f"Total pages loaded: {len(documents)}")

    # ── 2. Split ─────────────────────────────────────────────────────────────
    print("[2/4] Splitting into chunks ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"    → {len(chunks)} chunks (batch size: {BATCH_SIZE})")
    log.info(f"Split into {len(chunks)} chunks")

    # ── 3. Embed & store in batches ──────────────────────────────────────────
    print("[3/4] Embedding chunks (this is the slow part) ...")
    log.info(f"Starting embedding: provider={EMBEDDING_PROVIDER!r} model={EMBEDDING_MODEL!r} collection={COLLECTION_NAME!r}")
    print(f"    Provider: {EMBEDDING_PROVIDER} | Model: {EMBEDDING_MODEL} | Collection: {COLLECTION_NAME}")

    embeddings = get_embeddings()
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )

    batches = [chunks[i : i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
    total_batches = len(batches)

    with tqdm(
        total=len(chunks),
        unit="chunk",
        desc="  Embedding",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        for i, batch in enumerate(batches):
            t_batch = time.time()
            vectorstore.add_documents(batch)
            elapsed = time.time() - t_batch
            log.info(
                f"Batch {i+1}/{total_batches}: {len(batch)} chunks in {elapsed:.1f}s"
                f" (avg {elapsed/len(batch):.2f}s/chunk)"
            )
            pbar.update(len(batch))

    # ── 4. Done (Chroma >= 0.4 auto-persists) ────────────────────────────────
    print("[4/4] Finalizing ...")

    total_time = time.time() - t_start
    msg = (
        f"Ingestion complete: {len(chunks)} chunks from {len(pdf_files)} file(s)"
        f" in {total_time:.1f}s"
    )
    print(f"    ✓ {msg}")
    print(f"    Log saved to: {LOG_FILE}")
    log.info(msg)


if __name__ == "__main__":
    ingest()

