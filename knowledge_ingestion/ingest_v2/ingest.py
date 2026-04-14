# experimentation/ingestion/ingest.py
"""
Entry point for the experimental modular ingestion pipeline.

Run from the repo root:
    poetry run python -m experimentation.ingestion.ingest

Override the data root:
    DATA_ROOT=./knowledge_ingestion/content poetry run python -m experimentation.ingestion.ingest

Customise a stage without editing this file — import build_default_pipeline
and pass replacement stages:

    from experimentation.ingestion.pipeline import build_default_pipeline
    from experimentation.ingestion.pipeline.stages import ChunkingStage
    from experimentation.ingestion.pipeline.strategies import PaperChunkingStrategy

    pipeline = build_default_pipeline(
        chunker=ChunkingStage(strategy_map={"paper": PaperChunkingStrategy()})
    )
    pipeline.run(data_root=Path("./knowledge_ingestion/content"))
"""

from pathlib import Path

from app.config import DATA_ROOT
from knowledge_ingestion.ingest_v2.pipeline import build_default_pipeline


def main() -> None:
    pipeline = build_default_pipeline()
    pipeline.run(
        data_root=Path(DATA_ROOT),
        # debug_output_dir=Path("./debug_output")
    )


if __name__ == "__main__":
    main()
