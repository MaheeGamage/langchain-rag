import os
import sys
import uuid
from datetime import datetime
from inspect import signature
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI, OpenAI
from langchain_core.messages import AIMessage

from ragas import EvaluationDataset, SingleTurnSample, experiment
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)


# Add project root to path to import app modules.
# File: evaluation/ragas/evals.py -> repo root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from app.graph import graph
from app.config import EMBEDDING_MODEL, LLM_BASE_URL, JUDGE_LLM_BASE_URL, JUDGE_LLM_MODEL


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _list_available_models(client: OpenAI) -> list[str]:
    try:
        return [model.id for model in client.models.list().data]
    except Exception:
        return []


def _resolve_judge_model(configured_model: str | None, available_models: list[str]) -> str:
    # Small models like tinyllama often fail strict JSON schema generation in Ragas metrics,
    # so prefer stronger local models when available.
    preferred_non_tiny = ["mistral:latest", "phi3.5:latest", "mistral", "phi3.5"]
    available = set(available_models)

    if available:
        if configured_model and configured_model.startswith("tinyllama"):
            for model in preferred_non_tiny:
                if model in available:
                    return model

        candidates: list[str] = []
        if configured_model:
            candidates.append(configured_model)
            if ":" not in configured_model:
                candidates.append(f"{configured_model}:latest")

        candidates.extend(preferred_non_tiny)
        candidates.extend(["tinyllama:latest", "tinyllama"])

        for model in candidates:
            if model in available:
                return model

        return available_models[0]

    if configured_model and not configured_model.startswith("tinyllama"):
        return configured_model

    return "mistral:latest"


def _resolve_embedding_model(configured_model: str | None, available_models: list[str]) -> str:
    available = set(available_models)

    candidates: list[str] = []
    if configured_model:
        candidates.append(configured_model)
        if ":" not in configured_model:
            candidates.append(f"{configured_model}:latest")

    candidates.extend(["nomic-embed-text:latest", "nomic-embed-text"])

    if available:
        for model in candidates:
            if model in available:
                return model

    return configured_model or "nomic-embed-text:latest"

# Create OpenAI-compatible client for Ollama (judge LLM)
# `EVAL_OLLAMA_BASE_URL` is script-specific and can override global settings.
eval_ollama_base_url = _normalize_base_url(
    os.getenv("EVAL_OLLAMA_BASE_URL")
    or os.getenv("OLLAMA_BASE_URL")
    or JUDGE_LLM_BASE_URL
    or LLM_BASE_URL
    or "http://localhost:11435"
)

model_probe_client = OpenAI(
    api_key="ollama",  # Ollama doesn't require a real key
    base_url=f"{eval_ollama_base_url}/v1",
)

ollama_client = AsyncOpenAI(
    api_key="ollama",  # Ollama doesn't require a real key
    base_url=f"{eval_ollama_base_url}/v1",
)

# Resolve a robust judge model from available local Ollama models.
available_judge_models = _list_available_models(model_probe_client)
judge_model = _resolve_judge_model(JUDGE_LLM_MODEL, available_judge_models)
judge_embedding_model = _resolve_embedding_model(EMBEDDING_MODEL, available_judge_models)

if judge_model != (JUDGE_LLM_MODEL or ""):
    print(f"Using judge model '{judge_model}' (configured='{JUDGE_LLM_MODEL}')")

judge_llm = llm_factory(judge_model, client=ollama_client)
judge_embeddings = OpenAIEmbeddings(client=ollama_client, model=judge_embedding_model)


def _evaluate_with_collections_metrics(dataset: EvaluationDataset) -> tuple[dict[str, float], pd.DataFrame]:
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    rows = [sample.model_dump(exclude_none=True) for sample in dataset.samples]

    for metric in metrics:
        required_args = [
            name for name in signature(metric.ascore).parameters.keys() if name != "self"
        ]

        try:
            metric_inputs = [
                {arg: row[arg] for arg in required_args}
                for row in rows
            ]
        except KeyError as exc:
            missing = str(exc).strip("'\"")
            for row in rows:
                row[metric.name] = float("nan")
                row[f"{metric.name}_error"] = f"Missing required field '{missing}'"
            continue

        try:
            metric_results = metric.batch_score(metric_inputs)
            for row, metric_result in zip(rows, metric_results):
                row[metric.name] = metric_result.value
        except Exception as exc:
            for row in rows:
                row[metric.name] = float("nan")
                row[f"{metric.name}_error"] = f"{type(exc).__name__}: {exc}"

    summary: dict[str, float] = {}
    for metric in metrics:
        values = [
            row[metric.name]
            for row in rows
            if isinstance(row.get(metric.name), (int, float)) and row[metric.name] == row[metric.name]
        ]
        summary[metric.name] = sum(values) / len(values) if values else float("nan")

    return summary, pd.DataFrame(rows)


def rag_agent(question: str) -> str:
    """Your RAG agent implementation"""
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


def load_dataset() -> EvaluationDataset:
    # Build a minimal single-turn dataset with real model output.
    question = "How do I log circuit depth in MLflow?"
    response = rag_agent(question)
    samples = [
        SingleTurnSample(
            user_input=question,
            retrieved_contexts=["<chunk from your FAISS index>"],
            response=response,
            reference="Use mlflow.log_metric('circuit_depth', circuit.depth())."
        ),
    ]

    dataset = EvaluationDataset(samples=samples)

    # data_samples = [
    #     # ── TYPE 1: Factual Lookup ─────────────────────────────────────────────
    #     {
    #         "question": "How do I log the number of shots in an MLflow experiment run?",
    #         "grading_notes": "Use mlflow.log_param('shots', value) where value is the integer number of shots passed to sampler.run(). The shots argument in sampler.run([circuit], shots=1024) is the QProv E3 field.",
    #         "reference_contexts": [
    #             "E3 — Number of Shots: The number of times the circuit is executed to build a statistical estimate of the output distribution. MLflow logging: mlflow.log_param('shots', 1024). Qiskit source: shots argument in sampler.run([circuit], shots=1024)."
    #         ],
    #     },
    #     {
    #         "question": "What Qiskit API should I use to get the circuit depth, and what MLflow call logs it?",
    #         "grading_notes": "Use circuit.depth() to get the circuit depth in Qiskit. Log it with mlflow.log_metric('circuit_depth', circuit.depth()). Per QProv, depth is the maximum number of gates executed sequentially on any single qubit.",
    #         "reference_contexts": [
    #             "Q5 — Circuit Depth: MLflow logging: mlflow.log_metric('circuit_depth', circuit.depth()). Qiskit source: circuit.depth() returns an integer representing the number of gate layers."
    #         ],
    #     },
    #     # ── TYPE 2: Conceptual Why ─────────────────────────────────────────────
    #     {
    #         "question": "Why is it important to record decoherence times (T1 and T2) as part of quantum experiment provenance?",
    #         "grading_notes": "Decoherence times define the coherence window within which a circuit must complete execution. Circuits deeper than the decoherence limit produce unreliable results because qubits lose their quantum state. T1 and T2 also change between hardware calibrations, so they must be captured at execution time to enable later analysis of why a specific run produced certain results.",
    #         "reference_contexts": [
    #             "QC2 — Decoherence Times: T1 (energy relaxation) and T2 (dephasing) times characterize how long qubits maintain their quantum state. Decoherence times define the coherence window within which the circuit must complete execution. Circuits with depth exceeding the decoherence limit produce unreliable results."
    #         ],
    #     },
    #     {
    #         "question": "Why does QProv require logging the random seed used during transpilation?",
    #         "grading_notes": "Qubit and gate mapping during transpilation is an NP-hard problem, so quantum compilers often use randomized algorithms. Without recording the random seed (QProv field C4), the exact qubit assignments and gate mappings cannot be reproduced, making results from that run irreproducible.",
    #         "reference_contexts": [
    #             "C4 — Random Seed: As the mapping of qubits and gates is an NP-hard problem, often randomised compilers are used. The random seed should be collected, as the resulting mappings are otherwise not reproducible."
    #         ],
    #     },
    #     # ── TYPE 3: Important Distinction ─────────────────────────────────────
    #     {
    #         "question": "What is the difference between circuit.width() and circuit.num_qubits in Qiskit, and which should I use to log QProv circuit width?",
    #         "grading_notes": "circuit.width() in Qiskit returns the total number of qubits plus classical bits combined. circuit.num_qubits returns only the qubit count. For QProv alignment, use circuit.num_qubits because QProv defines circuit width (Q4) strictly as the number of qubits, not including classical bits.",
    #         "reference_contexts": [
    #             "Q4 — Circuit Width: circuit.num_qubits returns the number of qubits in the circuit. Note: Qiskit's circuit.width() returns qubits plus classical bits combined, which is broader than the QProv definition. Use circuit.num_qubits for QProv-aligned logging."
    #         ],
    #     },
    #     {
    #         "question": "What is the difference between circuit size and circuit depth in QProv, and why does each matter?",
    #         "grading_notes": "Circuit depth (Q5) is the maximum number of gates executed sequentially on any single qubit — it determines execution time and noise accumulation relative to decoherence limits. Circuit size (Q6) is the total number of gate operations in the entire circuit — it indicates cumulative gate error across all qubits. A circuit can have high size but low depth if many gates run in parallel.",
    #         "reference_contexts": [
    #             "Q5 — Circuit Depth: The maximum number of gates executed sequentially on any single qubit. Deeper circuits take longer to execute and accumulate more noise.",
    #             "Q6 — Circuit Size: The total number of gate operations in the circuit. Circuit size gives a direct count of quantum operations, useful for comparing circuit complexity.",
    #         ],
    #     },
    #     {
    #         "question": "QProv field E7 covers error mitigation — does it include Zero-Noise Extrapolation or only readout mitigation?",
    #         "grading_notes": "Per the QProv specification, E7 (Applied Error Mitigation) specifically refers to readout-error mitigation only — techniques that post-process measurement results to reduce readout noise, such as calibration matrix inversion or iterative Bayesian unfolding. It does not cover gate-error mitigation techniques such as Zero-Noise Extrapolation.",
    #         "reference_contexts": [
    #             "E7 — Applied Error Mitigation: Per the QProv specification, E7 specifically refers to readout-error mitigation — techniques that post-process measurement results to reduce the influence of readout noise. It does not cover gate-error mitigation such as Zero-Noise Extrapolation."
    #         ],
    #     },
    #     # ── TYPE 4: Cross-document Synthesis ──────────────────────────────────
    #     {
    #         "question": "How do I log qubit connectivity for a quantum experiment in MLflow using Qiskit?",
    #         "grading_notes": "Qubit connectivity corresponds to QProv field QC3. Use mlflow.log_param('coupling_map', str(backend.coupling_map)) to log it. The coupling map describes which pairs of physical qubits can directly interact via two-qubit gates. Limited connectivity causes the compiler to insert SWAP gates, which increases depth and noise.",
    #         "reference_contexts": [
    #             "QC3 — Qubit Connectivity: The coupling map describes which pairs of physical qubits can directly interact. Limited connectivity requires SWAP gates during compilation, increasing circuit depth and noise. MLflow logging: mlflow.log_param('coupling_map', str(backend.coupling_map))."
    #         ],
    #     },
    #     {
    #         "question": "How should I log intermediate results for a VQE algorithm in MLflow to comply with QProv?",
    #         "grading_notes": "Use mlflow.log_metric('energy', value, step=iteration) inside the optimization loop to log the energy value at each iteration. This corresponds to QProv field E4 (Intermediate Results). Per QProv, E4 applies specifically to variational algorithms like VQE or QAOA where multiple rounds of quantum and classical processing occur — it is not applicable for standard non-variational circuits because measurements destroy superposition.",
    #         "reference_contexts": [
    #             "E4 — Intermediate Results: Per the QProv specification, this field is relevant specifically for variational algorithms (e.g., VQE, QAOA). MLflow logging: Log within the optimization loop using mlflow.log_metric('energy', value, step=iteration)."
    #         ],
    #     },
    #     {
    #         "question": "What are the four categories of quantum provenance defined in QProv, and what does each category cover?",
    #         "grading_notes": "QProv defines four categories: (1) Quantum Circuit — properties of the circuit itself such as gates, depth, width, and size; (2) Quantum Computer — hardware properties such as qubit count, decoherence times, connectivity, gate fidelities, and readout fidelities; (3) Compilation — how the circuit was transpiled to hardware, including qubit assignments, gate mappings, optimization goals, and the random seed; (4) Execution — what happened at runtime, including input/output data, number of shots, execution time, intermediate results, and applied error mitigation.",
    #         "reference_contexts": [
    #             "Quantum provenance (QProv) is a specification for systematically recording metadata about quantum software experiments. It defines four top-level categories: Quantum Circuit, Quantum Computer, Compilation, and Execution."
    #         ],
    #     },
    # ]

    # for sample in data_samples:
    #     row = {"question": sample["question"], "grading_notes": sample["grading_notes"]}
    #     # row = {"question": sample["question"], "ground_truth": sample["ground_truth"]}
    #     dataset.append(row)

    # # make sure to save it
    # dataset.save()
    return dataset


my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)


@experiment()
async def run_experiment(row):
    response = rag_agent(row["question"])

    score = my_metric.score(
        llm=judge_llm,
        response=response,
        grading_notes=row["grading_notes"],
    )

    experiment_view = {
        **row,
        "response": response,
        "score": score.value,
    }
    return experiment_view


def main():
    dataset = load_dataset()
    print("dataset loaded successfully", dataset)

    experiment_summary, results_df = _evaluate_with_collections_metrics(dataset)
    print("Experiment completed successfully!")
    print("Experiment results:", experiment_summary)

    # Save experiment results to CSV
    output_dir = PROJECT_ROOT / "evaluation" / "ragas" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"evals_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nExperiment results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
