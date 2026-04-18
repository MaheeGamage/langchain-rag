import asyncio
import sys
from pathlib import Path

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import FactualCorrectness

try:
    from evaluation.ragas.ragas_factory import (
        get_ragas_judge_embeddings,
        get_ragas_judge_llm,
    )
except ModuleNotFoundError:
    # VS Code "Run Python File" executes by absolute file path, so repo root
    # may be missing from sys.path. Add it to support package imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evaluation.ragas.ragas_factory import (
        get_ragas_judge_embeddings,
        get_ragas_judge_llm,
    )


async def main() -> None:
    # Setup LLM
    llm = get_ragas_judge_llm()
    embeddings = get_ragas_judge_embeddings()

    # Create metric
    scorer = FactualCorrectness(llm=llm, mode="recall")

    # Evaluate
    result = await scorer.ascore(
        response="The open-source platform originally used in ML/AI that is proposed as the foundation for the quantum experiment tracking system is MLflow.",
        reference="MLflow",
    )
    print(f"Factual Correctness Score: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())