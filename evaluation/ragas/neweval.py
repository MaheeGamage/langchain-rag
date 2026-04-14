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
    LLM_MODEL, LLM_PROVIDER, 
    EMBEDDING_MODEL, EMBEDDING_PROVIDER,
    JUDGE_PROVIDER, JUDGE_EMBEDDING_PROVIDER,
)
from evaluation.ragas.ragas_factory import get_ragas_judge_llm, get_ragas_judge_embeddings


EVAL_DATASET_PATH = os.path.abspath(
    # os.path.join(os.path.dirname(__file__), "..", "mlflow", "eval_dataset.json")
    # os.path.join(os.path.dirname(__file__), "..", "eval_dataset.json")
    os.path.join(os.path.dirname(__file__), "..", "question-sets", "qset-v2.json")
)
MAX_Q_RAW = 1


def load_eval_dataset() -> list[dict[str, str]]:
    """Load eval dataset and normalize to {'user_input', 'reference'} shape."""
    dataset_path = EVAL_DATASET_PATH
    
    max_q = int(MAX_Q_RAW) if MAX_Q_RAW else None

    with open(dataset_path, "r", encoding="utf-8") as f:
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

    if max_q is not None:
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
print("Tip: set MAX_Q to limit questions, e.g. MAX_Q=3")
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
        faithfulness_score, faithfulness_error = await safe_ascore(
            "faithfulness",
            faithfulness_metric.ascore(
                user_input=sample.user_input,
                response=sample.response,
                retrieved_contexts=sample.retrieved_contexts,
            ),
        )

        answer_relevance_score, answer_relevance_error = await safe_ascore(
            "answer_relevance",
            answer_relevance_metric.ascore(
                user_input=sample.user_input,
                response=sample.response,
            ),
        )

        context_precision_score, context_precision_error = await safe_ascore(
            "context_precision",
            context_precision_metric.ascore(
                user_input=sample.user_input,
                reference=sample.reference,
                retrieved_contexts=sample.retrieved_contexts,
            ),
        )

        context_recall_score, context_recall_error = await safe_ascore(
            "context_recall",
            context_recall_metric.ascore(
                user_input=sample.user_input,
                reference=sample.reference,
                retrieved_contexts=sample.retrieved_contexts,
            ),
        )

        factual_correctness_score, factual_correctness_error = await safe_ascore(
            "factual_correctness",
            factual_correctness_metric.ascore(
                response=sample.response,
                reference=sample.reference,
            ),
        )

        metric_errors = {
            "faithfulness": faithfulness_error,
            "context_precision": context_precision_error,
            "context_recall": context_recall_error,
            "answer_relevance": answer_relevance_error,
            "factual_correctness": factual_correctness_error,
        }

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
            "subdomain": record["subdomain"],
            "question_class": record["question_class"],
            "faithfulness": faithfulness_score,
            "context_precision": context_precision_score,
            "context_recall": context_recall_score,
            "answer_relevance": answer_relevance_score,
            "factual_correctness": factual_correctness_score,
            "metric_errors": {
                k: v for k, v in metric_errors.items() if v is not None
            },
        })
    
    return results

# Run async evaluation
results = asyncio.run(evaluate_samples())
score_df = pd.DataFrame(results)
score_cols = [
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevance",
    "factual_correctness",
]
for col in score_cols:
    score_df[col] = pd.to_numeric(score_df[col], errors="coerce")

print("\n=== Per-sample Scores ===")
print(score_df[["user_input", "subdomain", "question_class", "faithfulness", "context_precision", "context_recall", "answer_relevance", "factual_correctness", 
                ]].to_string(index=False))

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
        "judge_embedding_provider": JUDGE_EMBEDDING_PROVIDER,
        "ragas_version": "0.4.3",
        "num_samples": len(sample_records),
        "ragas_metrics": "faithfulness,context_precision,context_recall,answer_relevance,factual_correctness",
    })
    print("\nMetrics logged to MLflow ✓")
