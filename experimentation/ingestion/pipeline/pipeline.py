# pipeline/pipeline.py
"""
IngestPipeline — wires the four stages together and runs them in sequence.

    Walker → Parser → Chunker → Embedder

Each stage is a plain object with a run() method.  Swap any stage by passing
a replacement to IngestPipeline's constructor — nothing else changes.

Usage
─────
    # Default pipeline (reads DATA_ROOT from config/env):
    pipeline = build_default_pipeline()
    pipeline.run(data_root=Path("./knowledge_ingestion/content"))

    # Custom chunking strategy for one doc type:
    from pipeline.strategies.chunking import PaperChunkingStrategy
    from pipeline.stages import ChunkingStage

    pipeline = build_default_pipeline(
        chunker=ChunkingStage(strategy_map={"paper": MyCustomPaperStrategy()})
    )
    pipeline.run(data_root=Path("./knowledge_ingestion/content"))
"""

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .stages.walker   import WalkerStage
from .stages.parser   import ParserStage
from .stages.chunker  import ChunkingStage
from .stages.embedder import EmbedderStage

LOG_FILE = "ingest_pipeline.log"


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("ingest_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    return logger


@dataclass
class IngestPipeline:
    """
    Composable four-stage ingestion pipeline.

    All stages have sensible defaults.  Override any stage to change its
    behaviour without touching the others.
    """
    walker:   WalkerStage   = field(default_factory=WalkerStage)
    parser:   ParserStage   = field(default_factory=ParserStage)
    chunker:  ChunkingStage = field(default_factory=ChunkingStage)
    embedder: EmbedderStage = field(default_factory=EmbedderStage)

    def run(self, data_root: Path) -> None:
        log = _setup_logger()
        t_start = time.time()

        if not data_root.exists():
            raise FileNotFoundError(
                f"data_root '{data_root}' does not exist. "
                "Set DATA_ROOT in your environment or pass a path explicitly."
            )

        # ── Stage 1: Walk ─────────────────────────────────────────────────────
        all_files = self.walker.run(data_root)
        print(f"\n[1/4] Discovered {len(all_files):,} files under '{data_root}'")
        log.info(f"data_root={data_root!r}  total_files={len(all_files)}")

        # ── Stage 2: Parse ────────────────────────────────────────────────────
        print("[2/4] Parsing files ...")
        raw_docs, skipped_files = self.parser.run(all_files)

        format_counter: Counter = Counter(
            d.metadata.get("format", "?") for d in raw_docs
        )
        corpus_counter: Counter = Counter(
            d.metadata.get("source_corpus", "?") for d in raw_docs
        )
        print(
            f"    → {len(raw_docs):,} raw documents "
            f"({skipped_files:,} files skipped — no parser)"
        )
        print(f"    → By format: {dict(format_counter)}")
        print(f"    → By corpus: {dict(corpus_counter)}")
        log.info(
            f"Parsed {len(raw_docs)} docs.  Skipped: {skipped_files}.  "
            f"Formats: {dict(format_counter)}  Corpora: {dict(corpus_counter)}"
        )

        if not raw_docs:
            print("    ⚠  No documents to ingest.  Check DATA_ROOT and file types.")
            return

        # ── Stage 3: Chunk ────────────────────────────────────────────────────
        print("[3/4] Chunking ...")
        chunks = self.chunker.run(raw_docs)

        content_type_counter: Counter = Counter(
            c.metadata.get("content_type", "?") for c in chunks
        )
        strategy_counter: Counter = Counter(
            c.metadata.get("source_corpus", "?") for c in chunks
        )
        print(f"    → {len(chunks):,} chunks total")
        print(f"    → By content type: {dict(content_type_counter)}")
        print(f"    → By corpus:       {dict(strategy_counter)}")
        log.info(
            f"Produced {len(chunks)} chunks.  "
            f"content_type={dict(content_type_counter)}"
        )

        # ── Stage 4: Embed ────────────────────────────────────────────────────
        print("[4/4] Embedding chunks ...")
        result = self.embedder.run(chunks)

        total_time = time.time() - t_start
        summary = (
            f"Ingestion complete: {result['added']:,} chunks added, "
            f"{result['skipped']:,} duplicates skipped, "
            f"{result['failed']:,} failed — "
            f"{len(all_files):,} files in {total_time:.1f}s"
        )
        print(f"\n    ✓ {summary}")
        print(f"    Log: {LOG_FILE}")
        log.info(summary)


def build_default_pipeline(
    walker:   WalkerStage   | None = None,
    parser:   ParserStage   | None = None,
    chunker:  ChunkingStage | None = None,
    embedder: EmbedderStage | None = None,
) -> IngestPipeline:
    """
    Construct a pipeline with defaults, optionally overriding specific stages.

    Examples
    ────────
    # Swap only the chunker:
    pipeline = build_default_pipeline(chunker=MyChunkingStage())

    # Swap only the embedder (e.g. for a dry-run that skips Chroma):
    pipeline = build_default_pipeline(embedder=DryRunEmbedderStage())
    """
    return IngestPipeline(
        walker   = walker   or WalkerStage(),
        parser   = parser   or ParserStage(),
        chunker  = chunker  or ChunkingStage(),
        embedder = embedder or EmbedderStage(),
    )
