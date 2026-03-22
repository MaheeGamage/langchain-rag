import os
import sys
import uuid
from pathlib import Path
from typing import Any
import asyncio

from langchain_core.messages import AIMessage

from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import Faithfulness

# Add project root to path to import app modules.
# File: evaluation/ragas/evals.py -> repo root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from app.graph import graph
from app.config import JUDGE_LLM_MODEL
from evaluation.ragas.judge_client import get_judge_client, get_ragas_async_judge_setup, resolve_judge_model

judge_client, judge_provider = get_ragas_async_judge_setup()

# Use the judge model from config, fallback to phi3.5
judge_model = resolve_judge_model(JUDGE_LLM_MODEL)
llm = llm_factory(judge_model, provider=judge_provider, client=judge_client)


# def rag_agent(question: str) -> str:
#     """Your RAG agent implementation"""
#     config = {"configurable": {"thread_id": str(uuid.uuid4())}}

#     result = graph.invoke(
#         {"messages": question, "context": [], "retrieved": []},
#         config=config,
#     )

#     answer = ""
#     for m in reversed(result["messages"]):
#         if isinstance(m, AIMessage):
#             answer = m.content
#             break

#     return answer

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

    retrieved_context = [
        entry.content
        for entry in result.get("retrieved", [])
        if getattr(entry, "content", None)
    ]

    # Keep an OpenAI-style chat payload so MLflow's built-in scorers read
    # the answer text from `messages[-1].content` while still exposing context.
    return {
        "messages": [{"role": "assistant", "content": answer}],
        "answer": answer,
        "retrieved_context": retrieved_context,
    }


def load_dataset():
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir="evaluation/ragas/eval",
    )

    # data_samples = [
    #     {
    #         "question": "What is ragas 0.3",
    #         "grading_notes": "- experimentation as the central pillar - provides abstraction for datasets, experiments and metrics - supports evals for RAG, LLM workflows and Agents",
    #     },
    #     {
    #         "question": "how are experiment results stored in ragas 0.3?",
    #         "grading_notes": "- configured using different backends like local, gdrive, etc - stored under experiments/ folder in the backend storage",
    #     },
    #     {
    #         "question": "What metrics are supported in ragas 0.3?",
    #         "grading_notes": "- provides abstraction for discrete, numerical and ranking metrics",
    #     },
    # ]

    data_samples = [
        # ── TYPE 1: Factual Lookup ─────────────────────────────────────────────
        {
            "question": "How do I log the number of shots in an MLflow experiment run?",
            "grading_notes": "Use mlflow.log_param('shots', value) where value is the integer number of shots passed to sampler.run(). The shots argument in sampler.run([circuit], shots=1024) is the QProv E3 field.",
            "reference_contexts": [
                "E3 — Number of Shots: The number of times the circuit is executed to build a statistical estimate of the output distribution. MLflow logging: mlflow.log_param('shots', 1024). Qiskit source: shots argument in sampler.run([circuit], shots=1024)."
            ],
        },
        # {
        #     "question": "What Qiskit API should I use to get the circuit depth, and what MLflow call logs it?",
        #     "grading_notes": "Use circuit.depth() to get the circuit depth in Qiskit. Log it with mlflow.log_metric('circuit_depth', circuit.depth()). Per QProv, depth is the maximum number of gates executed sequentially on any single qubit.",
        #     "reference_contexts": [
        #         "Q5 — Circuit Depth: MLflow logging: mlflow.log_metric('circuit_depth', circuit.depth()). Qiskit source: circuit.depth() returns an integer representing the number of gate layers."
        #     ],
        # },
        # # ── TYPE 2: Conceptual Why ─────────────────────────────────────────────
        # {
        #     "question": "Why is it important to record decoherence times (T1 and T2) as part of quantum experiment provenance?",
        #     "grading_notes": "Decoherence times define the coherence window within which a circuit must complete execution. Circuits deeper than the decoherence limit produce unreliable results because qubits lose their quantum state. T1 and T2 also change between hardware calibrations, so they must be captured at execution time to enable later analysis of why a specific run produced certain results.",
        #     "reference_contexts": [
        #         "QC2 — Decoherence Times: T1 (energy relaxation) and T2 (dephasing) times characterize how long qubits maintain their quantum state. Decoherence times define the coherence window within which the circuit must complete execution. Circuits with depth exceeding the decoherence limit produce unreliable results."
        #     ],
        # },
        # {
        #     "question": "Why does QProv require logging the random seed used during transpilation?",
        #     "grading_notes": "Qubit and gate mapping during transpilation is an NP-hard problem, so quantum compilers often use randomized algorithms. Without recording the random seed (QProv field C4), the exact qubit assignments and gate mappings cannot be reproduced, making results from that run irreproducible.",
        #     "reference_contexts": [
        #         "C4 — Random Seed: As the mapping of qubits and gates is an NP-hard problem, often randomised compilers are used. The random seed should be collected, as the resulting mappings are otherwise not reproducible."
        #     ],
        # },
        # # ── TYPE 3: Important Distinction ─────────────────────────────────────
        # {
        #     "question": "What is the difference between circuit.width() and circuit.num_qubits in Qiskit, and which should I use to log QProv circuit width?",
        #     "grading_notes": "circuit.width() in Qiskit returns the total number of qubits plus classical bits combined. circuit.num_qubits returns only the qubit count. For QProv alignment, use circuit.num_qubits because QProv defines circuit width (Q4) strictly as the number of qubits, not including classical bits.",
        #     "reference_contexts": [
        #         "Q4 — Circuit Width: circuit.num_qubits returns the number of qubits in the circuit. Note: Qiskit's circuit.width() returns qubits plus classical bits combined, which is broader than the QProv definition. Use circuit.num_qubits for QProv-aligned logging."
        #     ],
        # },
        # {
        #     "question": "What is the difference between circuit size and circuit depth in QProv, and why does each matter?",
        #     "grading_notes": "Circuit depth (Q5) is the maximum number of gates executed sequentially on any single qubit — it determines execution time and noise accumulation relative to decoherence limits. Circuit size (Q6) is the total number of gate operations in the entire circuit — it indicates cumulative gate error across all qubits. A circuit can have high size but low depth if many gates run in parallel.",
        #     "reference_contexts": [
        #         "Q5 — Circuit Depth: The maximum number of gates executed sequentially on any single qubit. Deeper circuits take longer to execute and accumulate more noise.",
        #         "Q6 — Circuit Size: The total number of gate operations in the circuit. Circuit size gives a direct count of quantum operations, useful for comparing circuit complexity.",
        #     ],
        # },
        # {
        #     "question": "QProv field E7 covers error mitigation — does it include Zero-Noise Extrapolation or only readout mitigation?",
        #     "grading_notes": "Per the QProv specification, E7 (Applied Error Mitigation) specifically refers to readout-error mitigation only — techniques that post-process measurement results to reduce readout noise, such as calibration matrix inversion or iterative Bayesian unfolding. It does not cover gate-error mitigation techniques such as Zero-Noise Extrapolation.",
        #     "reference_contexts": [
        #         "E7 — Applied Error Mitigation: Per the QProv specification, E7 specifically refers to readout-error mitigation — techniques that post-process measurement results to reduce the influence of readout noise. It does not cover gate-error mitigation such as Zero-Noise Extrapolation."
        #     ],
        # },
        # # ── TYPE 4: Cross-document Synthesis ──────────────────────────────────
        # {
        #     "question": "How do I log qubit connectivity for a quantum experiment in MLflow using Qiskit?",
        #     "grading_notes": "Qubit connectivity corresponds to QProv field QC3. Use mlflow.log_param('coupling_map', str(backend.coupling_map)) to log it. The coupling map describes which pairs of physical qubits can directly interact via two-qubit gates. Limited connectivity causes the compiler to insert SWAP gates, which increases depth and noise.",
        #     "reference_contexts": [
        #         "QC3 — Qubit Connectivity: The coupling map describes which pairs of physical qubits can directly interact. Limited connectivity requires SWAP gates during compilation, increasing circuit depth and noise. MLflow logging: mlflow.log_param('coupling_map', str(backend.coupling_map))."
        #     ],
        # },
        # {
        #     "question": "How should I log intermediate results for a VQE algorithm in MLflow to comply with QProv?",
        #     "grading_notes": "Use mlflow.log_metric('energy', value, step=iteration) inside the optimization loop to log the energy value at each iteration. This corresponds to QProv field E4 (Intermediate Results). Per QProv, E4 applies specifically to variational algorithms like VQE or QAOA where multiple rounds of quantum and classical processing occur — it is not applicable for standard non-variational circuits because measurements destroy superposition.",
        #     "reference_contexts": [
        #         "E4 — Intermediate Results: Per the QProv specification, this field is relevant specifically for variational algorithms (e.g., VQE, QAOA). MLflow logging: Log within the optimization loop using mlflow.log_metric('energy', value, step=iteration)."
        #     ],
        # },
        # {
        #     "question": "What are the four categories of quantum provenance defined in QProv, and what does each category cover?",
        #     "grading_notes": "QProv defines four categories: (1) Quantum Circuit — properties of the circuit itself such as gates, depth, width, and size; (2) Quantum Computer — hardware properties such as qubit count, decoherence times, connectivity, gate fidelities, and readout fidelities; (3) Compilation — how the circuit was transpiled to hardware, including qubit assignments, gate mappings, optimization goals, and the random seed; (4) Execution — what happened at runtime, including input/output data, number of shots, execution time, intermediate results, and applied error mitigation.",
        #     "reference_contexts": [
        #         "Quantum provenance (QProv) is a specification for systematically recording metadata about quantum software experiments. It defines four top-level categories: Quantum Circuit, Quantum Computer, Compilation, and Execution."
        #     ],
        # },
    ]

    for sample in data_samples:
        row = {"question": sample["question"], "grading_notes": sample["grading_notes"]}
        # row = {"question": sample["question"], "ground_truth": sample["ground_truth"]}
        dataset.append(row)

    # make sure to save it
    dataset.save()
    return dataset


my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)

metric_faithfulness = Faithfulness(llm=llm)

@experiment()
async def run_experiment(row):
    question = row["question"]
    response = rag_agent(row["question"])

    # score = my_metric.score(
    #     llm=llm,
    #     response=response["answer"],
    #     grading_notes=row["grading_notes"],
    # )

    score_faith = metric_faithfulness.score(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ]
)

    experiment_view = {
        **row,
        "answer": response["answer"],
        # "score": score.value,
        "faithfulness_score": score_faith.value,
    }
    return experiment_view


async def main():
    dataset = load_dataset()
    print("dataset loaded successfully", dataset)
    experiment_results = await run_experiment.arun(dataset)
    print("Experiment completed successfully!")
    print("Experiment results:", experiment_results)

    # Save experiment results to CSV
    experiment_results.save()
    csv_path = Path(".") / "experiments" / f"{experiment_results.name}.csv"
    print(f"\nExperiment results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
