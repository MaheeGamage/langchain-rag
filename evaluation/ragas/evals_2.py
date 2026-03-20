import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI, OpenAI
from langchain_core.messages import AIMessage

from ragas import EvaluationDataset, SingleTurnSample, experiment
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)


# Add project root to path to import app modules.
# File: evaluation/ragas/evals.py -> repo root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from app.graph import graph
from app.config import EMBEDDING_MODEL, LLM_BASE_URL, JUDGE_LLM_BASE_URL, JUDGE_LLM_MODEL


DEFAULT_QUESTIONS_FILE = (
    PROJECT_ROOT / "evaluation" / "ragas" / "qprov_eval_questions.json"
)


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _list_available_models(client: OpenAI) -> list[str]:
    try:
        return [model.id for model in client.models.list().data]
    except Exception:
        return []


def _resolve_judge_model(configured_model: str | None, available_models: list[str]) -> str:
    # Small models like tinyllama often fail strict JSON schema generation in Ragas metrics,
    # so prefer stronger local models when available.
    preferred_non_tiny = ["mistral:latest", "phi3.5:latest", "mistral", "phi3.5"]
    available = set(available_models)

    if available:
        if configured_model and configured_model.startswith("tinyllama"):
            for model in preferred_non_tiny:
                if model in available:
                    return model

        candidates: list[str] = []
        if configured_model:
            candidates.append(configured_model)
            if ":" not in configured_model:
                candidates.append(f"{configured_model}:latest")

        candidates.extend(preferred_non_tiny)
        candidates.extend(["tinyllama:latest", "tinyllama"])

        for model in candidates:
            if model in available:
                return model

        return available_models[0]

    if configured_model and not configured_model.startswith("tinyllama"):
        return configured_model

    return "mistral:latest"


def _resolve_embedding_model(configured_model: str | None, available_models: list[str]) -> str:
    available = set(available_models)

    candidates: list[str] = []
    if configured_model:
        candidates.append(configured_model)
        if ":" not in configured_model:
            candidates.append(f"{configured_model}:latest")

    candidates.extend(["nomic-embed-text:latest", "nomic-embed-text"])

    if available:
        for model in candidates:
            if model in available:
                return model

    return configured_model or "nomic-embed-text:latest"

# Create OpenAI-compatible client for Ollama (judge LLM)
# `EVAL_OLLAMA_BASE_URL` is script-specific and can override global settings.
eval_ollama_base_url = _normalize_base_url(
    os.getenv("EVAL_OLLAMA_BASE_URL")
    or os.getenv("OLLAMA_BASE_URL")
    or JUDGE_LLM_BASE_URL
    or LLM_BASE_URL
    or "http://localhost:11435"
)

model_probe_client = OpenAI(
    api_key="ollama",  # Ollama doesn't require a real key
    base_url=f"{eval_ollama_base_url}/v1",
)

ollama_client = AsyncOpenAI(
    api_key="ollama",  # Ollama doesn't require a real key
    base_url=f"{eval_ollama_base_url}/v1",
)

# Resolve a robust judge model from available local Ollama models.
available_judge_models = _list_available_models(model_probe_client)
judge_model = _resolve_judge_model(JUDGE_LLM_MODEL, available_judge_models)
judge_embedding_model = _resolve_embedding_model(EMBEDDING_MODEL, available_judge_models)

if judge_model != (JUDGE_LLM_MODEL or ""):
    print(f"Using judge model '{judge_model}' (configured='{JUDGE_LLM_MODEL}')")

judge_llm = llm_factory(judge_model, client=ollama_client)
judge_embeddings = OpenAIEmbeddings(client=ollama_client, model=judge_embedding_model)


def _evaluate_with_collections_metrics(dataset: EvaluationDataset) -> tuple[dict[str, float], pd.DataFrame]:
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    rows = [sample.model_dump(exclude_none=True) for sample in dataset.samples]

    for metric in metrics:
        required_args = [
            name for name in signature(metric.ascore).parameters.keys() if name != "self"
        ]

        try:
            metric_inputs = [
                {arg: row[arg] for arg in required_args}
                for row in rows
            ]
        except KeyError as exc:
            missing = str(exc).strip("'\"")
            for row in rows:
                row[metric.name] = float("nan")
                row[f"{metric.name}_error"] = f"Missing required field '{missing}'"
            continue

        try:
            metric_results = metric.batch_score(metric_inputs)
            for row, metric_result in zip(rows, metric_results):
                row[metric.name] = metric_result.value
        except Exception as exc:
            for row in rows:
                row[metric.name] = float("nan")
                row[f"{metric.name}_error"] = f"{type(exc).__name__}: {exc}"

    summary: dict[str, float] = {}
    for metric in metrics:
        values = [
            row[metric.name]
            for row in rows
            if isinstance(row.get(metric.name), (int, float)) and row[metric.name] == row[metric.name]
        ]
        summary[metric.name] = sum(values) / len(values) if values else float("nan")

    return summary, pd.DataFrame(rows)


def rag_agent(question: str) -> str:
    """Run the RAG graph and return only the final assistant answer."""
    answer, _ = rag_agent_with_context(question)
    return answer


def rag_agent_with_context(question: str) -> tuple[str, list[str]]:
    """Run the RAG graph and return answer plus retrieved context chunks."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke(
        {"messages": question, "context": [], "retrieved": []},
        config=config,
    )

    answer = ""
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            answer = m.content
            break

    retrieved_contexts: list[str] = []
    for entry in result.get("retrieved", []):
        chunk = getattr(entry, "content", None) or getattr(entry, "page_content", None)
        if chunk:
            retrieved_contexts.append(chunk)

    return answer, retrieved_contexts


def _load_question_rows(
    questions_file: Path = DEFAULT_QUESTIONS_FILE,
    num_questions: int | None = None,
) -> list[dict[str, Any]]:
    if not questions_file.exists():
        raise FileNotFoundError(
            f"Questions file not found: {questions_file}. "
            "Create it or pass --questions-file."
        )

    with questions_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {questions_file}.")

    if num_questions is not None:
        if num_questions <= 0:
            raise ValueError("--num-questions must be greater than 0.")
        rows = rows[:num_questions]

    return rows


def load_dataset(
    questions_file: Path = DEFAULT_QUESTIONS_FILE,
    num_questions: int | None = None,
) -> EvaluationDataset:
    question_rows = _load_question_rows(
        questions_file=questions_file,
        num_questions=num_questions,
    )

    samples: list[SingleTurnSample] = []
    total = len(question_rows)

    for idx, row in enumerate(question_rows, start=1):
        question = str(row.get("question", "")).strip()
        if not question:
            continue

        response, retrieved_contexts = rag_agent_with_context(question)

        reference = str(row.get("grading_notes", "")).strip()
        if not reference:
            reference_contexts = row.get("reference_contexts", [])
            if isinstance(reference_contexts, list):
                reference = "\n".join(str(item) for item in reference_contexts if item)
        if not reference:
            reference = "No reference provided."

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=retrieved_contexts,
                response=response,
                reference=reference,
            )
        )
        print(f"Built sample {idx}/{total}")

    if not samples:
        raise ValueError("No valid questions found to build evaluation samples.")

    return EvaluationDataset(samples=samples)


my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)


@experiment()
async def run_experiment(row):
    response = rag_agent(row["question"])

    score = my_metric.score(
        llm=judge_llm,
        response=response,
        grading_notes=row["grading_notes"],
    )

    experiment_view = {
        **row,
        "response": response,
        "score": score.value,
    }
    return experiment_view


def main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS collections-metric evaluation on question prompts."
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Limit evaluation to the first N questions.",
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default=str(DEFAULT_QUESTIONS_FILE),
        help="Path to a JSON file with evaluation questions.",
    )
    args = parser.parse_args()

    dataset = load_dataset(
        questions_file=Path(args.questions_file),
        num_questions=args.num_questions,
    )
    print("dataset loaded successfully", dataset)

    experiment_summary, results_df = _evaluate_with_collections_metrics(dataset)
    print("Experiment completed successfully!")
    print("Experiment results:", experiment_summary)

    # Save experiment results to CSV
    output_dir = PROJECT_ROOT / "evaluation" / "ragas" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"evals_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nExperiment results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
