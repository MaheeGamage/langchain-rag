# app/evaluator.py
"""MLflow GenAI evaluation harness for the RAG pipeline.

Call ``run_evaluation()`` to execute ``mlflow.genai.evaluate()`` against a
dataset of question/answer pairs using LLM-as-a-Judge scorers.
"""

from __future__ import annotations

import math
import uuid
from typing import Any

import mlflow
import mlflow.genai
from langchain_core.messages import AIMessage, HumanMessage
from mlflow.genai.scorers import (
    Correctness,
    Fluency,
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalSufficiency,
    Safety,
)

from .config import MLFLOW_EXPERIMENT_NAME, MLFLOW_JUDGE_MODEL, MLFLOW_TRACKING_URI
from .graph import graph

# ---------------------------------------------------------------------------
# Custom guideline definitions
# ---------------------------------------------------------------------------

_DOMAIN_GUIDELINES: list[str] = [
    "The response must be relevant to MLflow experiment tracking or quantum software development.",
    "Use a professional, technical tone appropriate for software engineers.",
    "Do not present information outside the scope of experiment tracking with MLflow SDKs.",
]

_CONCISENESS_GUIDELINES: list[str] = [
    "The response should directly answer the question without unnecessary padding or repetition.",
    "Avoid restating the question back to the user.",
]

# ---------------------------------------------------------------------------
# Scorer factory
# ---------------------------------------------------------------------------

_SCORER_MAP: dict[str, Any] = {
    "relevance":    RelevanceToQuery,
    "correctness":  Correctness,
    "fluency":      Fluency,
    "groundedness": RetrievalGroundedness,
    "sufficiency":  RetrievalSufficiency,
    "safety":       Safety,
    "conciseness":  None,   # built via Guidelines — see _build_scorers
    "domain_tone":  None,   # built via Guidelines
}


def _build_scorers(scorer_names: list[str], judge_model: str | None) -> list:
    unknown = set(scorer_names) - set(_SCORER_MAP)
    if unknown:
        raise ValueError(
            f"Unknown scorers: {sorted(unknown)}. "
            f"Valid choices: {sorted(_SCORER_MAP)}"
        )

    scorers = []
    model = judge_model or None  # empty string → None (MLflow default)

    for name in scorer_names:
        if name == "conciseness":
            scorers.append(
                Guidelines(name="conciseness", guidelines=_CONCISENESS_GUIDELINES, model=model)
            )
        elif name == "domain_tone":
            scorers.append(
                Guidelines(name="domain_tone", guidelines=_DOMAIN_GUIDELINES, model=model)
            )
        else:
            scorers.append(_SCORER_MAP[name](model=model))

    return scorers


# ---------------------------------------------------------------------------
# Predict function (runs the full RAG pipeline)
# ---------------------------------------------------------------------------

def _predict(question: str) -> str:
    """Run the RAG graph for a single question and return the answer text.

    The parameter name ``question`` must match the key used in ``inputs`` dicts
    passed to ``mlflow.genai.evaluate()``, because MLflow calls this function as
    ``predict_fn(**sample["inputs"])``.
    """
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "context": [],
            "retrieved": [],
        },
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_evaluation(
    dataset: list[dict],
    scorer_names: list[str],
    judge_model: str | None,
    experiment_name: str | None,
) -> dict:
    """Run ``mlflow.genai.evaluate()`` and return a JSON-serialisable result dict.

    Parameters
    ----------
    dataset:
        List of ``{"inputs": {...}, "expectations": {...}}`` dicts, as built by
        the ``/evaluate`` API endpoint.
    scorer_names:
        Subset of the keys in ``_SCORER_MAP`` to activate.
    judge_model:
        LiteLLM model string (e.g. ``"openai:/gpt-4o-mini"``).  ``None`` or
        empty string → MLflow picks the default.
    experiment_name:
        Override the MLflow experiment.  Falls back to ``MLFLOW_EXPERIMENT_NAME``.
    """
    exp_name = experiment_name or MLFLOW_EXPERIMENT_NAME
    judge = judge_model or MLFLOW_JUDGE_MODEL or None

    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(exp_name)

    scorers = _build_scorers(scorer_names, judge)

    result = mlflow.genai.evaluate(
        data=dataset,
        scorers=scorers,
        predict_fn=_predict,
    )

    rows = (
        _sanitize_rows(result.result_df.to_dict(orient="records"))
        if result.result_df is not None
        else []
    )

    return {
        "run_id": result.run_id,
        "experiment_name": exp_name,
        "metrics": result.metrics,
        "rows": rows,
    }


def _sanitize_rows(rows: list[dict]) -> list[dict]:
    """Replace NaN/Inf floats with None so the list is JSON-serialisable."""
    out = []
    for row in rows:
        clean: dict = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        out.append(clean)
    return out
