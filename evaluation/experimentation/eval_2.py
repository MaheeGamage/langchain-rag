"""
RAG Evaluation Pipeline for AI-Assisted Experiment Tracking in Quantum Software Development
============================================================================================
Thesis: Mahee Hewa Gamage, University of Jyväskylä
Supervisor: Vlad Stirbu -- EM4QS Project

Evaluation framework: RAGAS (Es et al., 2023) + MLflow
Judge model: Phi-3.5 Mini via Ollama (CPU-only, Intel i5-8365U)
Use case evaluated: UC1 -- Intelligent Advisory Chatbot

Dependencies:
    pip install -U ragas mlflow datasets pandas litellm sentence-transformers
    ollama pull phi3.5
    ollama serve

Run:
    python eval_2.py
    mlflow ui
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from datasets import Dataset

# ── RAGAS ─────────────────────────────────────────────────────────────────────
# Use ragas.metrics (legacy path) + LangchainLLMWrapper.
# ragas.metrics.collections + llm_factory/LangchainLLMWrapper are incompatible
# in this installed version. Legacy path works correctly with instantiated objects.
# Deprecation warnings are harmless -- they indicate future removal, not breakage.
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaLLM, OllamaEmbeddings


# ── Configuration ──────────────────────────────────────────────────────────────

DATASET_PATH = Path(__file__).parent / "eval_dataset_test.json"
MLFLOW_EXPERIMENT_NAME = "RAG_UC1_Advisory_Evaluation"
OLLAMA_BASE_URL = "http://localhost:11434"
JUDGE_MODEL = "phi3.5"
EMBED_MODEL = "nomic-embed-text"

os.environ['MLFLOW_TRACKING_USERNAME'] = 'mahee-admin'  # Replace with your username
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'j81A8k4QMib6'  # Replace with your password

mlflow.set_tracking_uri("https://mlflow.mahee.em4qs.qubernetes.dev/")


# ── RAG pipeline stub ──────────────────────────────────────────────────────────
# Replace query_rag_stub with your actual pipeline function.
# Return format must be:
#   {"answer": "...", "contexts": ["passage 1", "passage 2", ...]}

def query_rag_stub(question: str) -> dict:
    """Placeholder -- replace with your real RAG pipeline."""
    return {
        "answer": f"[STUB] Answer for: {question}",
        "contexts": [
            "[STUB] Retrieved context passage 1",
            "[STUB] Retrieved context passage 2",
        ]
    }


def load_rag_pipeline():
    """
    Swap the stub below for your real RAG import, e.g.:
        from graph import query_rag
        return query_rag
    """
    print("[WARNING] Using stub RAG pipeline. Replace load_rag_pipeline() with your real import.")
    return query_rag_stub


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_evaluation_dataset(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found: {path.resolve()}\n"
            f"Update DATASET_PATH to the correct location."
        )
    raw = path.read_text(encoding="utf-8").strip().lstrip("\ufeff")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"[ERROR] Invalid JSON: {e}\nFile: {path.resolve()}") from e
    if not isinstance(data, list):
        raise ValueError(f"[ERROR] Dataset must be a JSON list. Got: {type(data).__name__}")
    print(f"[OK] Loaded {len(data)} evaluation questions from: {path.resolve()}")
    return data


# ── RAGAS judge setup ──────────────────────────────────────────────────────────

def build_ragas_judge():
    """
    Build RAGAS judge using LangchainLLMWrapper + Ollama.

    Uses ragas.metrics (legacy) which accepts LangchainLLMWrapper.
    Deprecation warnings are harmless -- they mean "will break in a future
    RAGAS version", not the current installed one.

    Thesis note: Phi-3.5 Mini (3.8B) local judge instead of GPT-4 due to
    CPU-only constraints. Known limitation -- smaller models produce less
    calibrated RAGAS scores. See evaluation chapter limitations section.
    """
    llm = OllamaLLM(
        model=JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    run_config = RunConfig(
        timeout=600,     # phi3.5 on CPU can take 5-8 min per call
        max_retries=1,
        max_wait=10,
    )
    return ragas_llm, ragas_embeddings, run_config


def run_evaluation(dataset: list[dict], query_fn) -> tuple[dict, pd.DataFrame]:
    """Run all questions through the RAG pipeline, collect answers and contexts."""
    questions, answers, contexts, ground_truths = [], [], [], []
    question_ids, question_types, latencies = [], [], []

    print(f"\nRunning {len(dataset)} questions through RAG pipeline...")
    for item in dataset:
        print(f"  [{item['id']}] {item['question'][:70]}...")
        start = time.time()
        result = query_fn(item["question"])
        latency = round(time.time() - start, 3)

        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(item["ground_truth"])
        question_ids.append(item["id"])
        question_types.append(item["question_type"])
        latencies.append(latency)

    ragas_dataset = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    raw_df = pd.DataFrame({
        "id": question_ids,
        "question_type": question_types,
        "question": questions,
        "answer": answers,
        "ground_truth": ground_truths,
        "latency_s": latencies,
    })
    return ragas_dataset, raw_df


def compute_ragas_scores(
    ragas_dataset: dict,
    ragas_llm,
    ragas_embeddings,
    run_config,
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate with RAGAS legacy metrics (ragas.metrics, not collections).
    Metrics are instantiated as objects without llm -- the llm and embeddings
    are passed to evaluate() instead, which is correct for the legacy path.
    """
    print("\nRunning RAGAS evaluation (may take several minutes with local LLM)...")

    hf_dataset = Dataset.from_dict(ragas_dataset)

    # Instantiate without llm -- legacy metrics receive llm via evaluate()
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    result = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    def _safe_float(val) -> float:
        """Return NaN-safe float. Timed-out metrics return a list of NaNs."""
        import math
        try:
            v = float(val)
            return v if not math.isnan(v) else float("nan")
        except (TypeError, ValueError):
            # val is a list (e.g. [nan, nan]) -- take mean of non-nan values
            try:
                nums = [float(x) for x in val if x is not None]
                valid = [x for x in nums if not math.isnan(x)]
                return sum(valid) / len(valid) if valid else float("nan")
            except Exception:
                return float("nan")

    aggregate_scores = {
        "faithfulness":      _safe_float(result["faithfulness"]),
        "answer_relevancy":  _safe_float(result["answer_relevancy"]),
        "context_precision": _safe_float(result["context_precision"]),
        "context_recall":    _safe_float(result["context_recall"]),
    }

    # Warn about any metrics that timed out
    import math
    timed_out = [k for k, v in aggregate_scores.items() if math.isnan(v)]
    if timed_out:
        print(f"[WARNING] These metrics timed out and scored NaN: {timed_out}")
        print("[WARNING] Increase RunConfig timeout or switch to a faster model.")
    per_question_df = result.to_pandas()
    return aggregate_scores, per_question_df


def log_to_mlflow(
    aggregate_scores: dict,
    per_question_df: pd.DataFrame,
    raw_results_df: pd.DataFrame,
    dataset: list[dict],
    pipeline_config: dict,
) -> str:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        # Tags
        mlflow.set_tag("use_case",             "UC1_Advisory_Chatbot")
        mlflow.set_tag("evaluation_framework", "RAGAS")
        mlflow.set_tag("judge_model",          JUDGE_MODEL)
        mlflow.set_tag("judge_provider",       "Ollama_local")
        mlflow.set_tag("dataset_version",      "v1.0")
        mlflow.set_tag("num_questions",        len(dataset))
        mlflow.set_tag("thesis_project",       "AI_Assisted_Experiment_Tracking_QSD")

        # Params
        mlflow.log_param("judge_model",       JUDGE_MODEL)
        mlflow.log_param("judge_temperature", 0)
        mlflow.log_param("embed_model",       EMBED_MODEL)
        mlflow.log_param("num_questions",     len(dataset))
        mlflow.log_param("ragas_version",     _get_package_version("ragas"))
        mlflow.log_param("mlflow_version",    _get_package_version("mlflow"))
        for key, val in pipeline_config.items():
            mlflow.log_param(key, val)

        # Aggregate RAGAS metrics (skip NaN -- timed out metrics)
        import math
        valid_scores = {k: v for k, v in aggregate_scores.items() if not math.isnan(v)}
        for name, score in aggregate_scores.items():
            if not math.isnan(score):
                mlflow.log_metric(name, score)
            else:
                mlflow.set_tag(f"{name}_status", "timeout")
        composite = sum(valid_scores.values()) / len(valid_scores) if valid_scores else float("nan")
        if not math.isnan(composite):
            mlflow.log_metric("composite_ragas_score", composite)

        # Per question-type breakdown
        id_to_type = {item["id"]: item["question_type"] for item in dataset}
        if "id" not in per_question_df.columns:
            per_question_df.insert(0, "id", [item["id"] for item in dataset])
        if "question_type" not in per_question_df.columns:
            per_question_df.insert(1, "question_type",
                                   [id_to_type.get(i, "?") for i in per_question_df["id"]])
        for q_type in per_question_df["question_type"].unique():
            subset = per_question_df[per_question_df["question_type"] == q_type]
            for metric in ["faithfulness", "answer_relevancy",
                           "context_precision", "context_recall"]:
                if metric in subset.columns:
                    mlflow.log_metric(f"{metric}_{q_type}", float(subset[metric].mean()))

        # Latency
        mlflow.log_metric("avg_latency_s",      float(raw_results_df["latency_s"].mean()))
        mlflow.log_metric("max_latency_s",      float(raw_results_df["latency_s"].max()))
        mlflow.log_metric("total_query_time_s", float(raw_results_df["latency_s"].sum()))

        # Artifacts
        per_q_path = "ragas_per_question_scores.csv"
        per_question_df.to_csv(per_q_path, index=False)
        mlflow.log_artifact(per_q_path)

        raw_path = "raw_answers.csv"
        raw_results_df.to_csv(raw_path, index=False)
        mlflow.log_artifact(raw_path)

        mlflow.log_artifact(str(DATASET_PATH))

        # Summary
        run_id = mlflow.active_run().info.run_id
        print(f"\n{'='*60}")
        print(f"MLflow Run ID : {run_id}")
        print(f"Experiment    : {MLFLOW_EXPERIMENT_NAME}")
        print(f"\nAggregate RAGAS Scores:")
        for k, v in aggregate_scores.items():
            print(f"  {k:<28} {v:.4f}")
        print(f"  {'composite_ragas_score':<28} {composite:.4f}")
        print(f"\nRun: mlflow ui")
        print(f"{'='*60}\n")
        return run_id


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_package_version(package: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(package)
    except Exception:
        return "unknown"


def print_per_question_table(per_question_df: pd.DataFrame, dataset: list[dict]):
    id_to_type = {item["id"]: item["question_type"] for item in dataset}
    if "id" not in per_question_df.columns:
        per_question_df.insert(0, "id", [item["id"] for item in dataset])
    if "question_type" not in per_question_df.columns:
        per_question_df.insert(1, "question_type",
                               [id_to_type.get(i, "?") for i in per_question_df["id"]])
    cols = ["id", "question_type", "faithfulness", "answer_relevancy",
            "context_precision", "context_recall"]
    available = [c for c in cols if c in per_question_df.columns]
    print("\nPer-Question RAGAS Scores:")
    print(per_question_df[available].to_string(index=False, float_format="{:.3f}".format))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("RAG Evaluation -- AI-Assisted Quantum Experiment Tracking")
    print("UC1: Intelligent Advisory Chatbot")
    print("=" * 60)

    dataset = load_evaluation_dataset(DATASET_PATH)

    print(f"\nInitializing RAGAS judge ({JUDGE_MODEL} via Ollama)...")
    ragas_llm, ragas_embeddings, ragas_run_config = build_ragas_judge()

    query_fn = load_rag_pipeline()

    pipeline_config = {
        "rag_retriever":       "ChromaDB",
        "rag_embed_model":     "nomic-embed-text",
        "rag_top_k":           3,
        "rag_chunk_size":      400,
        "rag_chunk_overlap":   50,
        "rag_generator_model": "gemma-3-27b-it",
        "knowledge_base_docs": "mlflow_docs+qprov_spec+qiskit_bridge",
    }

    ragas_dataset, raw_results_df = run_evaluation(dataset, query_fn)

    aggregate_scores, per_question_df = compute_ragas_scores(
        ragas_dataset, ragas_llm, ragas_embeddings, ragas_run_config
    )

    print_per_question_table(per_question_df, dataset)

    log_to_mlflow(
        aggregate_scores=aggregate_scores,
        per_question_df=per_question_df,
        raw_results_df=raw_results_df,
        dataset=dataset,
        pipeline_config=pipeline_config,
    )


if __name__ == "__main__":
    main()