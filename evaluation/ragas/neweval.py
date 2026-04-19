import os
import sys
import uuid
import asyncio
import json
import mlflow
import pandas as pd
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import Faithfulness, ContextPrecision, ContextRecall, AnswerRelevancy, FactualCorrectness
from langchain_core.messages import AIMessage

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.graph import graph
from app.config import (
    JUDGE_LLM_MODEL, LLM_MODEL, LLM_PROVIDER, 
    EMBEDDING_MODEL, EMBEDDING_PROVIDER,
    JUDGE_PROVIDER, JUDGE_EMBEDDING_PROVIDER, RAG_GRAPH_IMPLEMENTATION,
)
from evaluation.ragas.ragas_factory import get_ragas_judge_llm, get_ragas_judge_embeddings


EVAL_DATASET_PATH = os.path.abspath(
    # os.path.join(os.path.dirname(__file__), "..", "mlflow", "eval_dataset.json")
    # os.path.join(os.path.dirname(__file__), "..", "eval_dataset.json")
    # os.path.join(os.path.dirname(__file__), "..", "question-sets", "test3.json")
    os.path.join(os.path.dirname(__file__), "..", "question-sets", "exp", "test5.json")
)

MAX_Q_RAW = 1
Q_INDICES_RAW = None       # e.g. "0,3,7"
FILTER_SUBDOMAIN = None #"qprov_provenance_taxonomy" #os.environ.get("FILTER_SUBDOMAIN")
FILTER_Q_CLASS = None

ENABLED_RAGAS_METRICS = [
    # "faithfulness",
    # "context_precision",
    "context_recall",
    # "answer_relevance",
    # "factual_correctness",
    "factual_correctness_recall"
]


def load_eval_dataset() -> list[dict[str, str]]:
    """Load eval dataset and normalize to {'user_input', 'reference'} shape."""
    max_q = int(MAX_Q_RAW) if MAX_Q_RAW else None
    q_indices = (
        [int(i.strip()) for i in Q_INDICES_RAW.split(",") if i.strip()]
        if Q_INDICES_RAW else None
    )

    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    normalized = []
    for item in raw_items:
        question = item.get("inputs", {}).get("question")
        reference = item.get("expectations", {}).get("expected_response")
        subdomain = item.get("subdomain")
        question_class = item.get("q_class")

        if question and reference:
            normalized.append({
                "user_input": question,
                "reference": reference,
                "subdomain": subdomain,
                "question_class": question_class,
            })

    if FILTER_SUBDOMAIN:
        normalized = [q for q in normalized if q["subdomain"] == FILTER_SUBDOMAIN]
    if FILTER_Q_CLASS:
        normalized = [q for q in normalized if q["question_class"] == FILTER_Q_CLASS]
    if q_indices is not None:
        normalized = [normalized[i] for i in q_indices if i < len(normalized)]
    elif max_q is not None:
        normalized = normalized[:max_q]

    return normalized

def run_rag(question: str) -> dict:
    """Run the RAG pipeline using the actual graph from app/graph.py"""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    result = graph.invoke(
        {"messages": question, "context": [], "retrieved": []},
        config=config,
    )
    
    # Extract the answer from the last AIMessage
    answer = ""
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            answer = m.content
            break
    
    # Extract retrieved contexts
    retrieved_contexts = [
        entry.content
        for entry in result.get("retrieved", [])
        if getattr(entry, "content", None)
    ]
    
    return {
        "response": answer,
        "retrieved_contexts": retrieved_contexts,
    }

# Define evaluation dataset with questions and reference answers
eval_dataset = load_eval_dataset()

print(f"Using {LLM_PROVIDER} LLM: {LLM_MODEL}")
print(f"Using {EMBEDDING_PROVIDER} embeddings: {EMBEDDING_MODEL}")
print(f"Using {JUDGE_PROVIDER} judge LLM for evaluation")
print(f"Using {JUDGE_EMBEDDING_PROVIDER} judge embeddings for evaluation")
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
    print("  Tip: filter with MAX_Q=N, Q_INDICES=0,3,7, FILTER_SUBDOMAIN=x, FILTER_Q_CLASS=x")
print("\nRunning RAG on evaluation questions...")

# Build samples by running RAG for each question
sample_records = []
for item in eval_dataset:
    question = item["user_input"]
    reference = item["reference"]
    subdomain = item["subdomain"]
    question_class = item["question_class"]

    result = run_rag(question)
    
    print(f"  Q: {question}")
    print(f"  A: {result['response']}")
    print(f"  Retrieved {len(result['retrieved_contexts'])} contexts\n")
    
    sample_records.append({
        "sample": SingleTurnSample(
            user_input=question,
            response=result["response"],
            retrieved_contexts=result["retrieved_contexts"],
            reference=reference,
        ),
        "subdomain": subdomain,
        "question_class": question_class,
    })

ragas_dataset = EvaluationDataset(samples=[record["sample"] for record in sample_records])

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
    print("[info] No metrics enabled. Skipping RAGAS scoring and logging only per-sample outputs.")
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
        # Score each metric independently so one failure does not abort evaluation.
        metric_scores = {}
        metric_errors = {}
        for metric_name in enabled_metric_names:
            score_value, score_error = await safe_ascore(
                metric_name,
                metric_scorers[metric_name](sample),
            )
            metric_scores[metric_name] = score_value
            metric_errors[metric_name] = score_error

        failed_metrics = [name for name, err in metric_errors.items() if err]
        if failed_metrics:
            print(
                f"[warn] Metric failures for question: {sample.user_input[:80]}"
                f"... -> {', '.join(failed_metrics)}"
            )
        
        results.append({
            "user_input": sample.user_input,
            "response": sample.response,
            "reference": sample.reference,
            "retrieved_contexts": sample.retrieved_contexts or [],
            "retrieved_context": "\n\n---\n\n".join(sample.retrieved_contexts or []),
            "subdomain": record["subdomain"],
            "question_class": record["question_class"],
            **metric_scores,
            "metric_errors": {
                k: v for k, v in metric_errors.items() if v is not None
            },
        })
    
    return results

# Run async evaluation
results = asyncio.run(evaluate_samples())
score_df = pd.DataFrame(results)
score_cols = enabled_metric_names
for col in score_cols:
    score_df[col] = pd.to_numeric(score_df[col], errors="coerce")

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

with mlflow.start_run():
    for metric_name, metric_value in mean_scores.items():
        mlflow.log_metric(metric_name, metric_value)
    mlflow.log_table(data=score_df, artifact_file="ragas_scores.json")
    mlflow.log_params({
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": EMBEDDING_MODEL,
        "judge_provider": JUDGE_PROVIDER,
        "judge_model": JUDGE_LLM_MODEL,
        "judge_embedding_provider": JUDGE_EMBEDDING_PROVIDER,
        "ragas_version": "0.4.3",
        "num_samples": len(sample_records),
        "ragas_metrics": ",".join(enabled_metric_names),
        "langchain_graph_type": RAG_GRAPH_IMPLEMENTATION
    })
    print("\nMetrics logged to MLflow ✓")
