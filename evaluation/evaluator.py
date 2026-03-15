# quickstart_eval.py
import json
import os
from urllib import response
import uuid
from langchain_core.messages import AIMessage
import mlflow
from openai import OpenAI
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines

from app.config import LLM_MODEL, LLM_BASE_URL
from app.factory import get_judge_model_uri
from app.graph import graph

# Use different env variable when using a different LLM provider
mlflow.set_experiment("RAG Agent Evaluation 2")

def rag_agent(question: str) -> str:
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

    return answer

# Wrapper function for evaluation
def qa_predict_fn(question: str) -> str:
    out = rag_agent(question)
    return out
# {"response": response}


# Evaluation dataset
_dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
with open(_dataset_path) as f:
    eval_dataset = json.load(f)


# Scorers
@scorer
def is_concise(outputs: str) -> bool:
    return len(outputs.split()) <= 5

scorers = [
    Correctness(model=get_judge_model_uri()),
    # Guidelines(name="is_english", guidelines="The answer must be in English"),
    # is_concise,
]

# Run evaluation
if __name__ == "__main__":
    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=qa_predict_fn,
        scorers=scorers,
    )