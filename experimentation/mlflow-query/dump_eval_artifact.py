"""Dump an MLflow evaluation run into a {config, questions} JSON file.

Fetches a `ragas_scores.json` artifact from an MLflow run and reshapes it into
the precomputed-answers format consumed by `evaluation/ragas/neweval.py`
(see `evaluation/question-sets/questions-with-response/qset-v3-sample.json`
for the target schema).

Run selection mirrors `ragas_scores_analysis.ipynb`: pick an experiment and
either the latest run or a specific run_id.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


# --- Config ------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:8567"
EXPERIMENT_NAME = "exp_qsd"

# "latest" or an explicit run_id.
RUN_SELECTION = "53fdfeda71b644028d483906ed9cb406"

RAGAS_TABLE_ARTIFACT = "ragas_scores.json"

OUTPUT_DIR = Path(__file__).resolve().parents[1].parent / (
    "evaluation/question-sets/questions-with-response"
)
# If None, the filename is derived from the experiment + run_id.
OUTPUT_FILENAME: str | None = None

NOTES = "Generated from MLflow run via dump_eval_artifact.py."


# --- Helpers -----------------------------------------------------------------

def list_artifacts_recursive(client: MlflowClient, run_id: str, path: str = ""):
    items = []
    for art in client.list_artifacts(run_id, path):
        items.append(art.path)
        if art.is_dir:
            items.extend(list_artifacts_recursive(client, run_id, art.path))
    return items


def load_ragas_rows(client: MlflowClient, run_id: str) -> list[dict]:
    """Return the ragas_scores.json artifact as a list of row dicts."""
    artifact_paths = list_artifacts_recursive(client, run_id)
    matches = [p for p in artifact_paths if p.endswith(RAGAS_TABLE_ARTIFACT)]
    if not matches:
        raise FileNotFoundError(
            f"Run {run_id} has no {RAGAS_TABLE_ARTIFACT} artifact"
        )

    local_path = client.download_artifacts(run_id, matches[0])
    with open(local_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # mlflow.log_table can serialize as a list of records or as {columns, data}.
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "data" in payload and "columns" in payload:
        cols = payload["columns"]
        return [dict(zip(cols, row)) for row in payload["data"]]
    if isinstance(payload, dict):
        # Fall back to treating it as a column-oriented dict.
        keys = list(payload.keys())
        n = len(next(iter(payload.values())))
        return [{k: payload[k][i] for k in keys} for i in range(n)]
    raise ValueError(f"Unsupported {RAGAS_TABLE_ARTIFACT} structure")


def pick_run(client: MlflowClient, experiment_name: str, selection: str):
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    if selection == "latest":
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attribute.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise ValueError(f"No runs in experiment {experiment_name}")
        return runs[0], exp

    run = client.get_run(selection)
    return run, exp


def build_config_block(run, experiment_name: str) -> dict:
    params = run.data.params or {}
    tags = run.data.tags or {}

    started = datetime.fromtimestamp(
        run.info.start_time / 1000, tz=timezone.utc
    ).isoformat().replace("+00:00", "Z")

    config = {
        "llm_provider": params.get("llm_provider"),
        "llm_model": params.get("llm_model"),
        "embedding_provider": params.get("embedding_provider"),
        "embedding_model": params.get("embedding_model"),
        "langchain_graph_type": params.get("langchain_graph_type"),
        "retrieval_profile": params.get("retrieval_profile"),
        "generated_at": params.get("generated_at") or started,
        "source_mlflow_run_id": run.info.run_id,
        "source_experiment_name": experiment_name,
        "git_sha": (
            tags.get("mlflow.source.git.commit")
            or params.get("git_sha")
        ),
        "notes": NOTES,
    }
    return config


def row_to_question(row: dict) -> dict:
    retrieved = row.get("retrieved_contexts")
    # The artifact sometimes stores the list as a JSON string; normalize.
    if isinstance(retrieved, str):
        try:
            retrieved = json.loads(retrieved)
        except json.JSONDecodeError:
            retrieved = [retrieved]
    if retrieved is None:
        retrieved = []

    return {
        "inputs": {"question": row.get("user_input")},
        "expectations": {"expected_response": row.get("reference")},
        "precomputed": {
            "response": row.get("response"),
            "retrieved_contexts": list(retrieved),
        },
        "subdomain": row.get("subdomain"),
        "q_class": row.get("question_class"),
    }


# --- Main --------------------------------------------------------------------

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    run, exp = pick_run(client, EXPERIMENT_NAME, RUN_SELECTION)
    print(f"Using run {run.info.run_id} from experiment {exp.name}")

    rows = load_ragas_rows(client, run.info.run_id)
    print(f"Loaded {len(rows)} rows from {RAGAS_TABLE_ARTIFACT}")

    config = build_config_block(run, exp.name)
    questions = [row_to_question(r) for r in rows]
    output = {"config": config, "questions": questions}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = OUTPUT_FILENAME or f"{exp.name}-{run.info.run_id}.json"
    out_path = OUTPUT_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
