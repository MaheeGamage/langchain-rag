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

from langchain_core.messages import AIMessage
import mlflow
from openai import OpenAI
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines
from mlflow.genai.scorers.ragas import Faithfulness

from app.config import LLM_MODEL, LLM_BASE_URL
from app.factory import get_judge_model_uri
from app.graph import graph

# Use different env variable when using a different LLM provider
mlflow.set_experiment("RAG Agent Evaluation 4")

@mlflow.trace
def rag_agent(question: str) -> dict[str, Any]:
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

    retrieved_contexts = [
        entry.content
        for entry in result.get("retrieved", [])
        if getattr(entry, "content", None)
    ]

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
    Faithfulness(model=get_judge_model_uri()),
    # ContextPrecision(model="openai:/gpt-4"),
    # Guidelines(name="is_english", guidelines="The answer must be in English"),
    # is_concise,
]

# Run evaluation
if __name__ == "__main__":

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

    eval_dataset = load_eval_dataset(1) # for testing I set this to 1, change it to args.max_q

    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=qa_predict_fn,
        scorers=scorers,
    )