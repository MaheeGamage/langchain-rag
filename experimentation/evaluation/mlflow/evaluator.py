# quickstart_eval.py
import argparse
import json
import os
from urllib import response
import uuid
from typing import Any
import litellm

# LiteLLM (used by MLflow judge) reads OLLAMA_API_BASE, not OLLAMA_BASE_URL.
if "OLLAMA_API_BASE" not in os.environ:
    _ollama_base = os.getenv("OLLAMA_BASE_URL")
    if _ollama_base:
        os.environ["OLLAMA_API_BASE"] = _ollama_base

from langchain_core.messages import AIMessage, HumanMessage
import mlflow
from mlflow.entities import SpanType
from openai import OpenAI
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines
from mlflow.genai.scorers.ragas import ContextPrecision, ContextRecall, FactualCorrectness, Faithfulness, AnswerRelevancy

from app.config import LLM_MODEL, LLM_BASE_URL
from app.factory import get_judge_model_uri
from app.graph import graph

# Use different env variable when using a different LLM provider
mlflow.set_experiment("RAG Agent Evaluation 4")

# Enable MLflow autologging for LangChain to capture traces
mlflow.langchain.autolog(log_traces=True)

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_contexts(question: str, config: dict) -> tuple[list[str], dict]:
    """Separate retrieval function with RETRIEVER span."""
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)], "context": [], "retrieved": []},
        config=config,
    )
    
    retrieved_contexts = [
        entry.content
        for entry in result.get("retrieved", [])
        if getattr(entry, "content", None)
    ]
    
    return retrieved_contexts, result

@mlflow.trace(span_type=SpanType.CHAIN)
def rag_agent(question: str) -> dict[str, Any]:
    """Traced RAG prediction function for mlflow.genai.evaluate().
    
    Args:
        question: The question to answer (matches inputs['question'] key)
    
    Returns:
        dict with 'response' and 'retrieved_contexts' for RAGAS scorers
    """
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Call retrieval function (creates RETRIEVER span)
    retrieved_contexts, result = retrieve_contexts(question, config)

    # Extract answer from messages
    answer = ""
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            answer = m.content
            break

    # MLflow RAGAS scorers expect 'retrieved_contexts' in the outputs
    # for Faithfulness and other retrieval-based metrics to work
    return {
        "response": answer,
        "retrieved_contexts": retrieved_contexts,
    }

# Wrapper function for evaluation
def qa_predict_fn(question: str) -> dict[str, Any]:
    response = rag_agent(question)
    return response


# Evaluation dataset
_dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
def load_eval_dataset(num_eval_questions: int | None = None):
    with open(_dataset_path) as f:
        dataset = json.load(f)

    if num_eval_questions is None:
        return dataset

    return dataset[:num_eval_questions]


# Scorers
@scorer
def is_concise(outputs: Any) -> bool:
    if isinstance(outputs, str):
        answer_text = outputs
    else:
        answer_text = (outputs or {}).get("answer", "")
    return len(answer_text.split()) <= 5

scorers = [
    # Correctness(model=get_judge_model_uri()),
    # Faithfulness(model=get_judge_model_uri()),
    # ContextPrecision(model="openai:/gpt-4"),
    # Guidelines(name="is_english", guidelines="The answer must be in English"),
    # is_concise,

    Faithfulness(model=get_judge_model_uri()),
    FactualCorrectness(model=get_judge_model_uri()),
    AnswerRelevancy(model=get_judge_model_uri()),
    ContextPrecision(model=get_judge_model_uri()),
    ContextRecall(model=get_judge_model_uri()),
]

# Run evaluation
if __name__ == "__main__":
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:8567")
    
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluator on all questions or the first "
            "num_eval_questions questions."
        )
    )
    parser.add_argument(
        "--max-q",
        type=int,
        default=None,
        help="Number of questions to evaluate from the beginning. Defaults to all.",
    )
    args = parser.parse_args()

    eval_dataset = load_eval_dataset(args.max_q) # for testing I set this to 1, change it to args.max_q
    
    # Transform dataset to include ground_truth for RAGAS metrics
    # RAGAS Faithfulness needs ground_truth in expectations
    eval_data_transformed = []
    for item in eval_dataset:
        transformed = {
            "inputs": item["inputs"],
            "expectations": {
                "ground_truth": item["expectations"].get("expected_response", ""),
            }
        }
        eval_data_transformed.append(transformed)

    # Start an MLflow run for the evaluation
    with mlflow.start_run(run_name="ragas-faithfulness-test") as run:
        mlflow.log_param("num_samples", len(eval_data_transformed))
        mlflow.log_param("llm_model", LLM_MODEL)
        
        results = mlflow.genai.evaluate(
            data=eval_data_transformed,
            predict_fn=qa_predict_fn,
            scorers=scorers,
        )
        
        print(f"\nEvaluation complete! Run ID: {run.info.run_id}")
        print(f"View results at: http://localhost:8567/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")