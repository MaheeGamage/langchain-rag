import os
import sys
import time
import uuid
import asyncio
import json
import mlflow
import pandas as pd
import tiktoken
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
    FactualCorrectness,
)
from langchain_core.messages import AIMessage, HumanMessage

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.graph import graph, _stream_answer
from app.graphs.token_counter import TokenCountCallback
from app.config import (
    JUDGE_LLM_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    JUDGE_PROVIDER,
    JUDGE_EMBEDDING_PROVIDER,
    RAG_GRAPH_IMPLEMENTATION,
    RETRIEVER_PROFILE_OVERRIDE,
)
from evaluation.ragas.ragas_factory import (
    get_ragas_judge_llm,
    get_ragas_judge_embeddings,
)


# Path to the eval dataset JSON. Accepts either:
#   * an absolute path (just paste it in), or
#   * a path relative to the `evaluation/` directory (e.g. "question-sets/qset-v3.json").
# Examples:
#   EVAL_DATASET_PATH = "/home/mahee/.../questions-with-response/exp_qsd-xxx.json"
#   EVAL_DATASET_PATH = "question-sets/qset-v3.json"
EVAL_DATASET_PATH = "/home/mahee/Work/Thesis/Repos/langchain-rag/evaluation/question-sets/qset-v3.json"
# EVAL_DATASET_PATH = "/home/mahee/Work/Thesis/Repos/langchain-rag/evaluation/question-sets/questions-with-response/exp_qsd-53fdfeda71b644028d483906ed9cb406.json"

_EVAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DATASET_PATH = (
    EVAL_DATASET_PATH
    if os.path.isabs(EVAL_DATASET_PATH)
    else os.path.abspath(os.path.join(_EVAL_ROOT, EVAL_DATASET_PATH))
)

MAX_Q_RAW = 1
Q_INDICES_RAW = None  # e.g. "0,3,7"
FILTER_SUBDOMAIN = None  # "qprov_provenance_taxonomy"
FILTER_Q_CLASS = None

# When True, skip running the RAG pipeline and read `precomputed.response`
# and `precomputed.retrieved_contexts` from the eval dataset JSON instead.
# Items missing either field are kept in results with an error recorded in
# `metric_errors` for every enabled metric, and no metrics are computed.
USE_PRECOMPUTED_ANSWERS = False

ENABLED_RAGAS_METRICS = [
    # "faithfulness",
    # "context_precision",
    # "context_recall",
    # "answer_relevance",
    # "factual_correctness",
    # "factual_correctness_recall",
]


def load_eval_dataset() -> tuple[list[dict[str, str]], dict]:
    """Load eval dataset and normalize to {'user_input', 'reference'} shape.

    Accepts two top-level shapes:
      1. A flat list of question items (legacy).
      2. An object {"config": {...}, "questions": [...]} — the `config` block
         captures the RAG settings used to generate any precomputed answers
         (llm_model, embedding_model, graph, etc.) so that re-evaluating a
         precomputed dataset logs the correct producer info to MLflow.
    Returns (normalized_items, dataset_config).
    """
    max_q = int(MAX_Q_RAW) if MAX_Q_RAW else None
    q_indices = (
        [int(i.strip()) for i in Q_INDICES_RAW.split(",") if i.strip()]
        if Q_INDICES_RAW
        else None
    )

    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        dataset_config = raw.get("config") or {}
        raw_items = raw.get("questions") or []
    else:
        dataset_config = {}
        raw_items = raw

    normalized = []
    for item in raw_items:
        question = item.get("inputs", {}).get("question")
        reference = item.get("expectations", {}).get("expected_response")
        subdomain = item.get("subdomain")
        question_class = item.get("q_class")
        precomputed = item.get("precomputed") or {}

        if question and reference:
            normalized.append(
                {
                    "user_input": question,
                    "reference": reference,
                    "subdomain": subdomain,
                    "question_class": question_class,
                    "precomputed_response": precomputed.get("response"),
                    "precomputed_retrieved_contexts": precomputed.get(
                        "retrieved_contexts"
                    ),
                }
            )

    if FILTER_SUBDOMAIN:
        normalized = [q for q in normalized if q["subdomain"] == FILTER_SUBDOMAIN]
    if FILTER_Q_CLASS:
        normalized = [q for q in normalized if q["question_class"] == FILTER_Q_CLASS]
    if q_indices is not None:
        normalized = [normalized[i] for i in q_indices if i < len(normalized)]
    elif max_q is not None:
        normalized = normalized[:max_q]

    return normalized, dataset_config


_tokenizer = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text)) if text else 0


def run_rag(question: str) -> dict:
    """Run the RAG pipeline using the actual graph from app/graph.py"""
    token_cb = TokenCountCallback()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "callbacks": [token_cb]}

    # Use _stream_answer so retrieved docs are extracted correctly for every
    # graph implementation. For rag_agent the documents live in ToolMessage
    # artifacts, not in a top-level "retrieved" state key.
    retrieved, token_iter = _stream_answer(
        messages=[HumanMessage(content=question)],
        context_entries=[],
        graph_instance=graph,
        config=config,
    )
    answer = "".join(token_iter)

    retrieved_contexts = [
        entry.content for entry in retrieved if getattr(entry, "content", None)
    ]

    return {
        "response": answer,
        "retrieved_contexts": retrieved_contexts,
        "llm_calls": token_cb.llm_calls,
        "llm_input_tokens": token_cb.input_tokens,
        "llm_output_tokens": token_cb.output_tokens,
    }


total_start = time.perf_counter()

# Start the MLflow run up-front so its built-in start/end timestamps (shown as
# "Duration" in the experiment run list) cover the full script execution —
# dataset load, RAG, and RAGAS evaluation — not just the final logging step.
mlflow.start_run()

# Define evaluation dataset with questions and reference answers
eval_dataset, dataset_config = load_eval_dataset()


# Resolve producer-side settings: in precomputed mode prefer values embedded
# in the dataset (they describe the pipeline that generated the answers); in
# live-RAG mode use the currently loaded app.config.
def _resolve(key: str, live_value):
    if USE_PRECOMPUTED_ANSWERS and dataset_config.get(key) is not None:
        return dataset_config.get(key)
    return live_value


producer_llm_provider = _resolve("llm_provider", LLM_PROVIDER)
producer_llm_model = _resolve("llm_model", LLM_MODEL)
producer_embed_provider = _resolve("embedding_provider", EMBEDDING_PROVIDER)
producer_embed_model = _resolve("embedding_model", EMBEDDING_MODEL)
producer_graph_type = _resolve("langchain_graph_type", RAG_GRAPH_IMPLEMENTATION)

print(f"Using {producer_llm_provider} LLM: {producer_llm_model}")
print(f"Using {producer_embed_provider} embeddings: {producer_embed_model}")
print(f"Using {JUDGE_PROVIDER} judge LLM for evaluation")
print(f"Using {JUDGE_EMBEDDING_PROVIDER} judge embeddings for evaluation")
if RETRIEVER_PROFILE_OVERRIDE:
    print(f"Retriever profile override: {RETRIEVER_PROFILE_OVERRIDE}")
print(f"Loaded {len(eval_dataset)} evaluation questions")
if FILTER_SUBDOMAIN:
    print(f"  Filter: subdomain={FILTER_SUBDOMAIN}")
if FILTER_Q_CLASS:
    print(f"  Filter: q_class={FILTER_Q_CLASS}")
if Q_INDICES_RAW:
    print(f"  Filter: Q_INDICES={Q_INDICES_RAW}")
elif MAX_Q_RAW:
    print(f"  Filter: MAX_Q={MAX_Q_RAW}")
else:
    print(
        "  Tip: filter with MAX_Q=N, Q_INDICES=0,3,7, FILTER_SUBDOMAIN=x, FILTER_Q_CLASS=x"
    )
if USE_PRECOMPUTED_ANSWERS:
    print("\nUsing precomputed answers from dataset (skipping RAG)...")
    if dataset_config:
        print(f"  Dataset config: {dataset_config}")
    else:
        print(
            "  [warn] Dataset has no top-level `config` block — producer "
            "settings will fall back to the currently loaded app.config."
        )
else:
    print("\nRunning RAG on evaluation questions...")

# Build samples by running RAG for each question
sample_records = []
rag_phase_start = time.perf_counter()
for item in eval_dataset:
    question = item["user_input"]
    reference = item["reference"]
    subdomain = item["subdomain"]
    question_class = item["question_class"]

    load_error = None
    rag_runtime_s = None
    llm_calls = None
    llm_input_tokens = None
    llm_output_tokens = None
    if USE_PRECOMPUTED_ANSWERS:
        response = item.get("precomputed_response")
        retrieved_contexts = item.get("precomputed_retrieved_contexts")
        missing = []
        if not response:
            missing.append("precomputed.response")
        if retrieved_contexts is None:
            missing.append("precomputed.retrieved_contexts")
        if missing:
            load_error = "Missing required field(s): " + ", ".join(missing)
            print(f"  [skip] Q: {question[:80]} -> {load_error}")
            response = response or ""
            retrieved_contexts = retrieved_contexts or []
        else:
            print(f"  Q: {question}")
            print(f"  A: {response}")
            print(f"  Retrieved {len(retrieved_contexts)} contexts (precomputed)\n")
    else:
        rag_start = time.perf_counter()
        result = run_rag(question)
        rag_runtime_s = time.perf_counter() - rag_start
        response = result["response"]
        retrieved_contexts = result["retrieved_contexts"]
        llm_calls = result["llm_calls"]
        llm_input_tokens = result["llm_input_tokens"]
        llm_output_tokens = result["llm_output_tokens"]
        print(f"  Q: {question}")
        print(f"  A: {response}")
        print(
            f"  Retrieved {len(retrieved_contexts)} contexts in {rag_runtime_s:.2f}s"
            f" | LLM calls={llm_calls} in={llm_input_tokens} out={llm_output_tokens} tokens\n"
        )

    sample_records.append(
        {
            "sample": SingleTurnSample(
                user_input=question,
                response=response,
                retrieved_contexts=retrieved_contexts,
                reference=reference,
            ),
            "subdomain": subdomain,
            "question_class": question_class,
            "load_error": load_error,
            "rag_runtime_s": rag_runtime_s,
            "llm_calls": llm_calls,
            "llm_input_tokens": llm_input_tokens,
            "llm_output_tokens": llm_output_tokens,
        }
    )
rag_phase_runtime_s = time.perf_counter() - rag_phase_start

ragas_dataset = EvaluationDataset(
    samples=[record["sample"] for record in sample_records]
)

# Setup LLM and embeddings for Ragas metrics using configured judge providers
llm = get_ragas_judge_llm()
embeddings = get_ragas_judge_embeddings()

# Initialize metrics
faithfulness_metric = Faithfulness(llm=llm)
answer_relevance_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
context_precision_metric = ContextPrecision(llm=llm)
context_recall_metric = ContextRecall(llm=llm)
factual_correctness_metric = FactualCorrectness(llm=llm)
factual_correctness_recall_metric = FactualCorrectness(llm=llm, mode="recall")

metric_scorers = {
    "faithfulness": lambda sample: faithfulness_metric.ascore(
        user_input=sample.user_input,
        response=sample.response,
        retrieved_contexts=sample.retrieved_contexts,
    ),
    "answer_relevance": lambda sample: answer_relevance_metric.ascore(
        user_input=sample.user_input,
        response=sample.response,
    ),
    "context_precision": lambda sample: context_precision_metric.ascore(
        user_input=sample.user_input,
        reference=sample.reference,
        retrieved_contexts=sample.retrieved_contexts,
    ),
    "context_recall": lambda sample: context_recall_metric.ascore(
        user_input=sample.user_input,
        reference=sample.reference,
        retrieved_contexts=sample.retrieved_contexts,
    ),
    "factual_correctness": lambda sample: factual_correctness_metric.ascore(
        response=sample.response,
        reference=sample.reference,
    ),
    "factual_correctness_recall": lambda sample: factual_correctness_recall_metric.ascore(
        response=sample.response,
        reference=sample.reference,
    ),
}

enabled_metric_names = []
unknown_metrics = []
for metric_name in ENABLED_RAGAS_METRICS:
    if metric_name in metric_scorers:
        enabled_metric_names.append(metric_name)
    else:
        unknown_metrics.append(metric_name)

if unknown_metrics:
    print(
        "[warn] Ignoring unknown metric names in ENABLED_RAGAS_METRICS: "
        + ", ".join(unknown_metrics)
    )

if not enabled_metric_names:
    print(
        "[info] No metrics enabled. Skipping RAGAS scoring and logging only per-sample outputs."
    )
else:
    print("Enabled metrics: " + ", ".join(enabled_metric_names))

print("Running RAGAS evaluation...")


# Evaluate each sample using the collections API
async def evaluate_samples():
    async def safe_ascore(metric_name: str, coro):
        try:
            score = await coro
            return score.value, None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    results = []
    for record in sample_records:
        sample = record["sample"]
        load_error = record.get("load_error")
        # Score each metric independently so one failure does not abort evaluation.
        metric_scores = {name: None for name in enabled_metric_names}
        metric_errors = {}

        eval_runtime_s = None
        if load_error:
            # Skip all scoring for samples missing required precomputed data.
            for metric_name in enabled_metric_names:
                metric_errors[metric_name] = load_error
        else:
            eval_start = time.perf_counter()
            for metric_name in enabled_metric_names:
                score_value, score_error = await safe_ascore(
                    metric_name,
                    metric_scorers[metric_name](sample),
                )
                metric_scores[metric_name] = score_value
                metric_errors[metric_name] = score_error
            eval_runtime_s = time.perf_counter() - eval_start

        failed_metrics = [name for name, err in metric_errors.items() if err]
        if failed_metrics:
            print(
                f"[warn] Metric failures for question: {sample.user_input[:80]}"
                f"... -> {', '.join(failed_metrics)}"
            )

        contexts = sample.retrieved_contexts or []
        context_tokens = sum(_count_tokens(c) for c in contexts)
        response_tokens = _count_tokens(sample.response or "")
        question_tokens = _count_tokens(sample.user_input or "")

        results.append(
            {
                "user_input": sample.user_input,
                "response": sample.response,
                "reference": sample.reference,
                "retrieved_contexts": contexts,
                "retrieved_context": "\n\n---\n\n".join(contexts),
                "subdomain": record["subdomain"],
                "question_class": record["question_class"],
                # Observable effort metrics (always available, including precomputed mode)
                "num_retrieved_contexts": len(contexts),
                "context_tokens": context_tokens,
                "response_tokens": response_tokens,
                "question_tokens": question_tokens,
                "input_tokens_approx": question_tokens + context_tokens,
                # Actual API token counts (live RAG mode only; None when precomputed)
                "llm_calls": record.get("llm_calls"),
                "llm_input_tokens": record.get("llm_input_tokens"),
                "llm_output_tokens": record.get("llm_output_tokens"),
                **metric_scores,
                "metric_errors": {
                    k: v for k, v in metric_errors.items() if v is not None
                },
            }
        )

    return results


# Run async evaluation
eval_phase_start = time.perf_counter()
results = asyncio.run(evaluate_samples())
eval_phase_runtime_s = time.perf_counter() - eval_phase_start
total_runtime_s = time.perf_counter() - total_start
score_df = pd.DataFrame(results)
score_cols = enabled_metric_names
for col in score_cols:
    score_df[col] = pd.to_numeric(score_df[col], errors="coerce")

effort_cols = ["num_retrieved_contexts", "context_tokens", "response_tokens", "input_tokens_approx"]
live_effort_cols = ["llm_calls", "llm_input_tokens", "llm_output_tokens"]

print("\n=== Per-sample Scores ===")
display_cols = ["user_input", "subdomain", "question_class"] + score_cols
print(score_df[display_cols].to_string(index=False))

mean_scores = score_df[score_cols].mean(skipna=True).to_dict()
print("\n=== Mean Scores ===")
for k, v in mean_scores.items():
    if pd.isna(v):
        print(f"  {k}: N/A (all samples failed)")
    else:
        print(f"  {k}: {v:.4f}")

print("\n=== LLM Effort (tiktoken, all modes) ===")
for col in effort_cols:
    if col in score_df.columns:
        total = score_df[col].sum()
        mean = score_df[col].mean()
        print(f"  {col}: total={int(total)}  mean={mean:.1f}")

if not USE_PRECOMPUTED_ANSWERS:
    print("\n=== LLM Effort (API token counts, live mode) ===")
    for col in live_effort_cols:
        if col in score_df.columns and score_df[col].notna().any():
            total = score_df[col].sum()
            mean = score_df[col].mean()
            print(f"  {col}: total={int(total)}  mean={mean:.1f}")

print("\n=== Runtime ===")
print(f"  RAG phase:        {rag_phase_runtime_s:.2f}s")
print(f"  Evaluation phase: {eval_phase_runtime_s:.2f}s")
print(f"  Total:            {total_runtime_s:.2f}s")

# MLflow run was started at the top of the script so its built-in Duration
# column covers everything above. Log metrics/params/artifacts into it now.
for metric_name, metric_value in mean_scores.items():
    mlflow.log_metric(metric_name, metric_value)
mlflow.log_metric("rag_phase_runtime_s", rag_phase_runtime_s)
mlflow.log_metric("eval_phase_runtime_s", eval_phase_runtime_s)

# LLM effort — tiktoken-based (always present)
for col in effort_cols:
    if col in score_df.columns:
        mlflow.log_metric(f"mean_{col}", score_df[col].mean())
        mlflow.log_metric(f"total_{col}", int(score_df[col].sum()))

# LLM effort — real API token counts (live mode only)
if not USE_PRECOMPUTED_ANSWERS:
    for col in live_effort_cols:
        if col in score_df.columns and score_df[col].notna().any():
            mlflow.log_metric(f"total_{col}", int(score_df[col].sum()))
            mlflow.log_metric(f"mean_{col}", score_df[col].mean())
mlflow.log_table(data=score_df, artifact_file="ragas_scores.json")
params = {
    "llm_provider": producer_llm_provider,
    "llm_model": producer_llm_model,
    "embedding_provider": producer_embed_provider,
    "embedding_model": producer_embed_model,
    "judge_provider": JUDGE_PROVIDER,
    "judge_model": JUDGE_LLM_MODEL,
    "judge_embedding_provider": JUDGE_EMBEDDING_PROVIDER,
    "ragas_version": "0.4.3",
    "num_samples": len(sample_records),
    "ragas_metrics": ",".join(enabled_metric_names),
    "langchain_graph_type": producer_graph_type,
    "use_precomputed_answers": USE_PRECOMPUTED_ANSWERS,
    "retriever_profile_override": RETRIEVER_PROFILE_OVERRIDE,
}
# Surface any extra keys from the dataset config (e.g. retrieval_profile,
# generated_at) so they show up in the MLflow run alongside the core ones.
if USE_PRECOMPUTED_ANSWERS:
    for key, value in dataset_config.items():
        params.setdefault(key, value)
mlflow.log_params(params)
mlflow.end_run()
print("\nMetrics logged to MLflow ✓")
