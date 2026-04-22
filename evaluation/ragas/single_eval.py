import asyncio
import sys
from pathlib import Path

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import FactualCorrectness

try:
    from evaluation.ragas.ragas_factory import (
        get_ragas_judge_embeddings,
        get_ragas_judge_llm,
    )
except ModuleNotFoundError:
    # VS Code "Run Python File" executes by absolute file path, so repo root
    # may be missing from sys.path. Add it to support package imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evaluation.ragas.ragas_factory import (
        get_ragas_judge_embeddings,
        get_ragas_judge_llm,
    )

RESPONSE = """
The four main categories in the QProv taxonomy are:

1. **Quantum Circuit**: This category encompasses provenance fields that describe the structure and composition of the quantum circuit used in an experiment. It includes details about the gates used, measurements, and the execution order of those components.

2. **Quantum Computer**: This category includes provenance attributes related to the quantum computer's hardware characteristics, such as the number of qubits, decoherence times, and other relevant properties of the quantum computing platform.

3. **Compilation**: This category focuses on the processes involved in mapping the quantum circuit to specific machine instructions for execution on a quantum computer. It captures details like qubit assignments, gate mappings, and optimization goals used during the compilation process.

4. **Execution**: This category records metadata about the execution of the quantum circuit, including input data, output data, number of shots, execution time, and error mitigation techniques applied during the execution process. 

These categories help to systematically record metadata about quantum software experiments for better reproducibility and understanding.
"""

async def main() -> None:
    # Setup LLM
    llm = get_ragas_judge_llm()
    embeddings = get_ragas_judge_embeddings()

    # Create metric
    scorer = FactualCorrectness(llm=llm, mode="recall")

    # Evaluate
    result = await scorer.ascore(
        response=RESPONSE,
        # reference="The QProv taxonomy organizes quantum experiment data into the four primary categories of quantum circuit, quantum computer, compilation, and execution. The quantum circuit category describes structural algorithm components such as gate types, used measurements, circuit dimensions, and data encoding methods. Quantum computer provenance records physical hardware characteristics including the number of qubits, decoherence times, qubit connectivity, and gate fidelities. The compilation category tracks the adaptation of circuits for specific devices by capturing qubit assignments, gate mappings, and optimization goals. The execution category documents runtime specifics like input and output data, the number of shots, and applied readout-error mitigation techniques.",
        reference="The QProv provenance taxonomy groups quantum experiment information into four key areas. It covers the quantum circuit itself, the quantum hardware it runs on, how the circuit is compiled into executable instructions, and how it is executed, including the resulting data and performance details."
    )
    print(f"Factual Correctness Score: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())