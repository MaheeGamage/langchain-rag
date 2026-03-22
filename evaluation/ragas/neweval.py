import os
import sys
import uuid
import asyncio
import json
import mlflow
import pandas as pd
from openai import AsyncOpenAI
from ragas import EvaluationDataset, SingleTurnSample
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, ContextPrecision, ContextRecall, AnswerRelevancy
from langchain_core.messages import AIMessage

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.factory import get_judge_model_uri
from app.graph import graph
from app.config import LLM_MODEL, LLM_PROVIDER, EMBEDDING_MODEL, EMBEDDING_PROVIDER


def load_eval_dataset() -> list[dict[str, str]]:
    """Load eval dataset and normalize to {'user_input', 'reference'} shape."""
    dataset_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "mlflow", "eval_dataset.json")
    )
    MAX_Q_RAW = 1 #os.getenv("MAX_Q", "").strip()
    max_q = int(MAX_Q_RAW) if MAX_Q_RAW else None

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    normalized = []
    for item in raw_items:
        question = item.get("inputs", {}).get("question")
        reference = item.get("expectations", {}).get("expected_response")

        if question and reference:
            normalized.append({
                "user_input": question,
                "reference": reference,
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
print(f"Loaded {len(eval_dataset)} evaluation questions")
print("Tip: set MAX_Q to limit questions, e.g. MAX_Q=3")
print("\nRunning RAG on evaluation questions...")

# Build samples by running RAG for each question
samples = []
for item in eval_dataset:
    question = item["user_input"]
    reference = item["reference"]
    
    result = run_rag(question)
    
    print(f"  Q: {question}")
    print(f"  A: {result['response']}")
    print(f"  Retrieved {len(result['retrieved_contexts'])} contexts\n")
    
    samples.append(
        SingleTurnSample(
            user_input=question,
            response=result["response"],
            retrieved_contexts=result["retrieved_contexts"],
            reference=reference,
        )
    )

ragas_dataset = EvaluationDataset(samples=samples)

# ✅ Setup LLM for Ragas metrics using the collections API
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)


# Initialize metrics
faithfulness_metric = Faithfulness(llm=llm)
answer_relevance_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
context_precision_metric = ContextPrecision(llm=llm)
context_recall_metric = ContextRecall(llm=llm)

print("Running RAGAS evaluation...")

# Evaluate each sample using the collections API
async def evaluate_samples():
    results = []
    for sample in samples:
        # Score each metric for this sample
        faithfulness_score = await faithfulness_metric.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
        )

        answer_relevance_score = await answer_relevance_metric.ascore(
            user_input=sample.user_input,
            response=sample.response,
        )

        context_precision_score = await context_precision_metric.ascore(
            user_input=sample.user_input,
            reference=sample.reference,
            retrieved_contexts=sample.retrieved_contexts,
        )
        
        context_recall_score = await context_recall_metric.ascore(
            user_input=sample.user_input,
            reference=sample.reference,
            retrieved_contexts=sample.retrieved_contexts,
        )
        
        results.append({
            "user_input": sample.user_input,
            "response": sample.response,
            "reference": sample.reference,
            "faithfulness": faithfulness_score.value,
            "context_precision": context_precision_score.value,
            "context_recall": context_recall_score.value,
            "answer_relevance": answer_relevance_score.value,
        })
    
    return results

# Run async evaluation
results = asyncio.run(evaluate_samples())
score_df = pd.DataFrame(results)
print("\n=== Per-sample Scores ===")
print(score_df[["user_input", "faithfulness", "context_precision", "context_recall", "answer_relevance"]].to_string(index=False))

numeric_cols = score_df.select_dtypes(include="number").columns.tolist()
mean_scores = score_df[numeric_cols].mean().to_dict()
print("\n=== Mean Scores ===")
for k, v in mean_scores.items():
    print(f"  {k}: {v:.4f}")

mlflow.set_experiment("RAG Faithfulness Evaluation - Real System")
with mlflow.start_run():
    for metric_name, metric_value in mean_scores.items():
        mlflow.log_metric(metric_name, metric_value)
    mlflow.log_table(data=score_df, artifact_file="ragas_scores.json")
    mlflow.log_params({
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": EMBEDDING_MODEL,
        "ragas_version": "0.4.3",
        "num_samples": len(samples),
        "ragas_metrics": "faithfulness,context_precision,context_recall,answer_relevance",
    })
    print("\nMetrics logged to MLflow ✓")
