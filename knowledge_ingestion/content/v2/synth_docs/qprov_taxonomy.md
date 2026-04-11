# QProv Quantum Provenance Taxonomy

**Document purpose:** Structured reference of all QProv provenance fields for quantum software experiments. Each section is a self-contained retrieval unit for vector database indexing. Fields are grouped by QProv category: Quantum Circuit, Quantum Computer, Compilation, and Execution.

Quantum provenance (QProv) is a specification for systematically recording metadata about quantum software experiments. It defines four top-level categories: Quantum Circuit, Quantum Computer, Compilation, and Execution.

---

## Category: Quantum Circuit

Quantum Circuit provenance fields describe the structure and composition of the quantum circuit used in an experiment. These fields capture what the circuit does and how it is built, independent of the hardware it runs on.

---

### Q1 — Used Gates

**Field ID:** Q1  
**Category:** Quantum Circuit  
**Name:** Used Gates  
**Description:** The set of quantum gate types used in the circuit (e.g., H, CNOT, RX, RZ, CZ). Records which gate operations appear in the circuit definition.  
**Why it matters:** Gate types determine hardware compatibility and noise characteristics. Different gates have different error rates on real QPUs.  
**MLflow logging:** `mlflow.log_param("used_gates", str(list(circuit.count_ops().keys())))`  
**Qiskit source:** `circuit.count_ops()` returns a dictionary of gate names to counts.

---

### Q2 — Used Measurements

**Field ID:** Q2  
**Category:** Quantum Circuit  
**Name:** Used Measurements  
**Description:** The measurement operations applied in the circuit, including which qubits are measured and into which classical bits.  
**Why it matters:** Measurement placement affects circuit structure and is required for Sampler-based execution. Mid-circuit vs terminal measurements have different noise profiles.  
**MLflow logging:** `mlflow.log_param("num_measurements", circuit.count_ops().get("measure", 0))`  
**Qiskit source:** `circuit.count_ops().get("measure", 0)` for count; full measurement structure visible in `circuit.data`.

---

### Q3 — Execution Order

**Field ID:** Q3  
**Category:** Quantum Circuit  
**Name:** Execution Order  
**Description:** The sequential order in which gates and operations are applied across the circuit. Captures the layered structure of the computation.  
**Why it matters:** Execution order determines logical dependencies between gates and affects the circuit's depth and parallelism.  
**MLflow logging:** Log the QPY circuit artifact to preserve the full execution order. `mlflow.log_artifact("circuit.qpy")`  
**Qiskit source:** `circuit.data` contains the ordered list of `CircuitInstruction` objects.

---

### Q4 — Circuit Width

**Field ID:** Q4  
**Category:** Quantum Circuit  
**Name:** Circuit Width  
**Description:** The number of qubits used in the circuit. Per the QProv specification, circuit width refers specifically to the qubit count, not including classical bits.  
**Why it matters:** Circuit width determines whether the circuit can be executed on a given quantum computer — the device must provide at least as many physical qubits as the circuit width. It is used together with circuit depth to assess hardware compatibility.  
**MLflow logging:** `mlflow.log_param("circuit_width", circuit.num_qubits)`  
**Qiskit source:** `circuit.num_qubits` — returns the number of qubits in the circuit. Note: Qiskit's `circuit.width()` returns qubits *plus* classical bits combined, which is broader than the QProv definition. Use `circuit.num_qubits` for QProv-aligned logging.

---

### Q5 — Circuit Depth

**Field ID:** Q5  
**Category:** Quantum Circuit  
**Name:** Circuit Depth  
**Description:** The maximum number of gates executed sequentially on any single qubit in the circuit. Per the QProv specification, depth is defined as this sequential gate count, not as a layer count. Circuits with parallel gates still count only the longest sequential chain on any one qubit.  
**Why it matters:** Deeper circuits take longer to execute and accumulate more noise. Circuit depth is one of the primary indicators of whether a circuit can run within a device's coherence window.  
**MLflow logging:** `mlflow.log_metric("circuit_depth", circuit.depth())`  
**Qiskit source:** `circuit.depth()` — returns an integer representing the number of gate layers.

---

### Q6 — Circuit Size

**Field ID:** Q6  
**Category:** Quantum Circuit  
**Name:** Circuit Size  
**Description:** The total number of gate operations in the circuit (excluding barriers and other non-gate instructions).  
**Why it matters:** Circuit size gives a direct count of quantum operations, useful for comparing circuit complexity across experiments.  
**MLflow logging:** `mlflow.log_metric("circuit_size", circuit.size())`  
**Qiskit source:** `circuit.size()` — returns total number of operations.

---

### Q7 — Applied Encoding

**Field ID:** Q7  
**Category:** Quantum Circuit  
**Name:** Applied Encoding  
**Description:** The data encoding strategy used to embed classical input data into the quantum circuit (e.g., amplitude encoding, angle encoding, basis encoding). Relevant for variational and machine learning quantum algorithms.  
**Why it matters:** The encoding scheme affects circuit depth, expressibility, and the relationship between classical inputs and quantum states.  
**MLflow logging:** `mlflow.log_param("encoding_type", "angle_encoding")`  
**Qiskit source:** User-defined. Typically recorded as a string parameter describing the encoding strategy used.

---

## Category: Quantum Computer

Quantum Computer provenance fields describe the hardware device on which the experiment is executed. These fields capture the physical properties and constraints of the QPU (quantum processing unit) or simulator.

---

### QC1 — Number of Qubits

**Field ID:** QC1  
**Category:** Quantum Computer  
**Name:** Number of Qubits  
**Description:** The total number of physical qubits available on the quantum hardware device used for the experiment.  
**Why it matters:** Determines the maximum circuit width that can be executed. Distinguishes between different hardware backends.  
**MLflow logging:** `mlflow.log_param("backend_num_qubits", backend.num_qubits)`  
**Qiskit source:** `backend.num_qubits` on a configured backend object.

---

### QC2 — Decoherence Times

**Field ID:** QC2  
**Category:** Quantum Computer  
**Name:** Decoherence Times  
**Description:** The T1 (energy relaxation) and T2 (dephasing) times of the qubits on the device, measured in microseconds. These characterize how long qubits maintain their quantum state.  
**Why it matters:** Decoherence times define the coherence window within which the circuit must complete execution. Circuits with depth exceeding the decoherence limit produce unreliable results.  
**MLflow logging:** `mlflow.log_param("t1_qubit_0", t1_value)` / `mlflow.log_param("t2_qubit_0", t2_value)`  
**Qiskit source:** Available from backend properties on real IBM QPUs via `backend.properties().t1(qubit)` and `backend.properties().t2(qubit)`.

---

### QC3 — Qubit Connectivity

**Field ID:** QC3  
**Category:** Quantum Computer  
**Name:** Qubit Connectivity  
**Description:** The coupling map of the quantum device — which pairs of physical qubits can directly interact via two-qubit gates (e.g., CNOT). Represented as a graph of qubit pairs.  
**Why it matters:** Limited connectivity requires SWAP gates during compilation, increasing circuit depth and noise. Connectivity directly affects transpilation and final circuit quality.  
**MLflow logging:** `mlflow.log_param("coupling_map", str(backend.coupling_map))`  
**Qiskit source:** `backend.coupling_map` on a configured backend object.

---

### QC4 — Gate Set

**Field ID:** QC4  
**Category:** Quantum Computer  
**Name:** Gate Set  
**Description:** The set of native gates supported by the quantum hardware device (basis gates). Circuits are transpiled to use only these gates before execution.  
**Why it matters:** The native gate set determines how circuits are compiled. Gates not in the native set must be decomposed, adding depth and noise.  
**MLflow logging:** `mlflow.log_param("basis_gates", str(backend.basis_gates))`  
**Qiskit source:** `backend.basis_gates` on a configured backend object.

---

### QC5 — Gate Fidelities

**Field ID:** QC5  
**Category:** Quantum Computer  
**Name:** Gate Fidelities  
**Description:** The average error rate (1 - fidelity) for single-qubit and two-qubit gates on the device. Higher fidelity means lower error rate.  
**Why it matters:** Gate fidelities directly determine the expected output accuracy. Low fidelity gates degrade result quality, especially in deep circuits.  
**MLflow logging:** `mlflow.log_metric("avg_cx_error", cx_error_rate)`  
**Qiskit source:** Available from `backend.properties()` on real QPUs.

---

### QC6 — Gate Times

**Field ID:** QC6  
**Category:** Quantum Computer  
**Name:** Gate Times  
**Description:** The physical execution duration of each gate type on the hardware, measured in nanoseconds or device-specific time units.  
**Why it matters:** Gate times determine how much decoherence accumulates during circuit execution. Longer gates on noisy hardware reduce result fidelity.  
**MLflow logging:** `mlflow.log_param("cx_gate_time_ns", cx_time)`  
**Qiskit source:** Available from `backend.properties()` on real QPUs via `backend.properties().gate_length(gate, qubits)`.

---

### QC7 — Readout Fidelities

**Field ID:** QC7  
**Category:** Quantum Computer  
**Name:** Readout Fidelities  
**Description:** The accuracy of the measurement operation for each qubit, characterizing how reliably the hardware reads out qubit states as 0 or 1.  
**Why it matters:** Readout errors directly corrupt the output distribution. Low readout fidelity means measured bitstrings may not reflect the true quantum state.  
**MLflow logging:** `mlflow.log_metric("readout_error_qubit_0", readout_error)`  
**Qiskit source:** Available from `backend.properties()` on real QPUs via `backend.properties().readout_error(qubit)`.

---

## Category: Compilation

Compilation provenance fields describe the transpilation process that transforms the logical circuit into a hardware-executable form. These fields capture how the circuit was adapted for a specific device.

---

### C1 — Qubit Assignments

**Field ID:** C1  
**Category:** Compilation  
**Name:** Qubit Assignments  
**Description:** The mapping from logical qubits in the circuit to physical qubits on the hardware device, determined during transpilation.  
**Why it matters:** Different qubit assignments lead to different circuit depths and error rates due to hardware connectivity constraints. Reproducibility requires recording which physical qubits were used.  
**MLflow logging:** `mlflow.log_param("qubit_layout", str(transpiled_circuit.layout.final_layout))`  
**Qiskit source:** `transpiled_circuit.layout` after running `transpile()`.

---

### C2 — Gate Mappings

**Field ID:** C2  
**Category:** Compilation  
**Name:** Gate Mappings  
**Description:** The decomposition of logical gates into native hardware gates during transpilation. Records how each logical gate was translated into the device's basis gate set.  
**Why it matters:** Gate mappings affect the final circuit depth and fidelity. Two different compilations of the same logical circuit may produce very different physical circuits.  
**MLflow logging:** Log the transpiled circuit as a QPY artifact: `mlflow.log_artifact("transpiled_circuit.qpy")`  
**Qiskit source:** The transpiled `QuantumCircuit` object after `transpile()` contains the mapped gate sequence.

---

### C3 — Optimization Goals

**Field ID:** C3  
**Category:** Compilation  
**Name:** Optimization Goals  
**Description:** The optimization objective and level used during transpilation (e.g., minimize depth, minimize gate count, balance between depth and fidelity). In Qiskit, this corresponds to the `optimization_level` parameter (0–3).  
**Why it matters:** Different optimization strategies produce different compiled circuits. Recording this ensures the compilation process is reproducible.  
**MLflow logging:** `mlflow.log_param("optimization_level", 3)`  
**Qiskit source:** `optimization_level` argument in `qiskit.transpile()` or `generate_preset_pass_manager()`.

---

### C4 — Random Seed

**Field ID:** C4  
**Category:** Compilation  
**Name:** Random Seed  
**Description:** The random seed used during transpilation, which controls stochastic decisions in the routing and layout algorithms.  
**Why it matters:** Transpilation involves randomized algorithms. Without a fixed seed, re-running transpilation on the same circuit may produce a different physical circuit. Recording the seed is essential for exact reproducibility.  
**MLflow logging:** `mlflow.log_param("transpile_seed", 42)`  
**Qiskit source:** `seed_transpiler` argument in `qiskit.transpile()`.

---

### C5 — Compilation Time

**Field ID:** C5  
**Category:** Compilation  
**Name:** Compilation Time  
**Description:** The wall-clock time taken to transpile the circuit, in seconds.  
**Why it matters:** Compilation time is a performance metric relevant for large-scale experiments where transpilation overhead is significant.  
**MLflow logging:** `mlflow.log_metric("compilation_time_s", elapsed)`  
**Qiskit source:** Measured by wrapping `transpile()` with `time.time()` before and after.

```python
import time
from qiskit import transpile

start = time.time()
transpiled = transpile(circuit, backend, seed_transpiler=42, optimization_level=3)
elapsed = time.time() - start

mlflow.log_metric("compilation_time_s", elapsed)
mlflow.log_param("transpile_seed", 42)
mlflow.log_param("optimization_level", 3)
```

---

## Category: Execution

Execution provenance fields describe what happened when the circuit was run, including inputs, outputs, and runtime metadata.

---

### E1 — Input Data

**Field ID:** E1  
**Category:** Execution  
**Name:** Input Data  
**Description:** The classical input data provided to the quantum circuit, particularly relevant for variational algorithms and quantum machine learning where classical data is encoded into circuit parameters.  
**Why it matters:** Recording input data ensures experiment reproducibility and enables correlation analysis between inputs and outcomes.  
**MLflow logging:** `mlflow.log_param("input_data", str(input_vector))` or log as artifact for large datasets.

---

### E2 — Output Data

**Field ID:** E2  
**Category:** Execution  
**Name:** Output Data  
**Description:** The raw output of the quantum execution — either a bitstring count distribution (Sampler) or an expectation value (Estimator).  
**Why it matters:** Output data is the primary result of the experiment. It must be recorded alongside all other provenance fields to make results interpretable and reproducible.  
**MLflow logging (Sampler):**
```python
counts = result[0].data["meas"].get_counts()
for bitstring, count in counts.items():
    mlflow.log_metric(f"count_{bitstring}", count)
```
**MLflow logging (Estimator):**
```python
mlflow.log_metric("expectation_value", float(result[0].data.evs))
```

---

### E3 — Number of Shots

**Field ID:** E3  
**Category:** Execution  
**Name:** Number of Shots  
**Description:** The number of times the circuit is executed (repeated measurements) to build a statistical estimate of the output distribution. A fundamental parameter of quantum execution.  
**Why it matters:** Quantum output is probabilistic. More shots reduce statistical uncertainty but increase execution time and cost. Shot count must be recorded to interpret the reliability of output distributions.  
**MLflow logging:** `mlflow.log_param("shots", 1024)`  
**Qiskit source:** `shots` argument in `sampler.run([circuit], shots=1024)`.

---

### E4 — Intermediate Results

**Field ID:** E4  
**Category:** Execution  
**Name:** Intermediate Results  
**Description:** Results captured at intermediate steps of a multi-iteration algorithm. Per the QProv specification, this field is relevant specifically for *variational algorithms* (e.g., VQE, QAOA) where multiple rounds of quantum and classical processing occur. For most standard quantum circuits, intermediate results cannot be collected because measurements destroy qubit superposition. For variational algorithms, intermediate results from each iteration — such as energy values or loss — should be gathered.  
**Why it matters:** Intermediate results enable convergence analysis, debugging, and understanding of algorithm behavior across iterations.  
**MLflow logging:** Log within the optimization loop using `mlflow.log_metric("energy", value, step=iteration)`.

```python
for iteration, (params, energy) in enumerate(optimization_history):
    mlflow.log_metric("energy", energy, step=iteration)
    for i, p in enumerate(params):
        mlflow.log_metric(f"param_{i}", p, step=iteration)
```

---

### E5 — Number of Iterations

**Field ID:** E5  
**Category:** Execution  
**Name:** Number of Iterations  
**Description:** The total number of optimization or execution iterations completed, relevant for variational algorithms (VQE, QAOA) and iterative quantum protocols.  
**Why it matters:** Records convergence behavior and total computational cost of iterative quantum algorithms.  
**MLflow logging:** `mlflow.log_metric("num_iterations", total_iterations)`

---

### E6 — Execution Time

**Field ID:** E6  
**Category:** Execution  
**Name:** Execution Time  
**Description:** The total wall-clock time for circuit execution, in seconds. Includes queue time, circuit execution, and result retrieval for hardware backends.  
**Why it matters:** Execution time is a key performance metric for benchmarking and resource planning.  
**MLflow logging:** `mlflow.log_metric("execution_time_s", elapsed)`

```python
import time

start = time.time()
result = sampler.run([circuit], shots=1024).result()
elapsed = time.time() - start

mlflow.log_metric("execution_time_s", elapsed)
```

---

### E7 — Applied Error Mitigation

**Field ID:** E7  
**Category:** Execution  
**Name:** Applied Error Mitigation  
**Description:** The readout-error mitigation technique applied to correct measurement errors in the output distribution. Per the QProv specification, E7 specifically refers to *readout-error* mitigation — techniques that post-process measurement results to reduce the influence of readout noise (e.g., calibration matrix inversion, iterative Bayesian unfolding, detector tomography-based correction). It does not cover gate-error mitigation such as Zero-Noise Extrapolation.  
**Why it matters:** Readout errors corrupt the output bitstring distribution. Mitigation substantially changes the reported results, making uncorrected and corrected results not directly comparable. Recording which technique was applied is essential for reproducibility and cross-experiment comparison.  
**MLflow logging:** `mlflow.log_param("readout_mitigation", "calibration_matrix")`  
**Note:** If no mitigation is applied, log explicitly: `mlflow.log_param("readout_mitigation", "none")`

---

## QProv Field Summary Table

| Field ID | Name | Category | MLflow Log Type | Qiskit Source |
|---|---|---|---|---|
| Q1 | Used Gates | Quantum Circuit | param | `circuit.count_ops().keys()` |
| Q2 | Used Measurements | Quantum Circuit | param | `circuit.count_ops().get("measure")` |
| Q3 | Execution Order | Quantum Circuit | artifact | `circuit.data` / QPY |
| Q4 | Circuit Width | Quantum Circuit | param | `circuit.num_qubits` |
| Q5 | Circuit Depth | Quantum Circuit | metric | `circuit.depth()` |
| Q6 | Circuit Size | Quantum Circuit | metric | `circuit.size()` |
| Q7 | Applied Encoding | Quantum Circuit | param | user-defined string |
| QC1 | Number of Qubits | Quantum Computer | param | `backend.num_qubits` |
| QC2 | Decoherence Times | Quantum Computer | param | `backend.properties().t1/t2()` |
| QC3 | Qubit Connectivity | Quantum Computer | param | `backend.coupling_map` |
| QC4 | Gate Set | Quantum Computer | param | `backend.basis_gates` |
| QC5 | Gate Fidelities | Quantum Computer | metric | `backend.properties()` |
| QC6 | Gate Times | Quantum Computer | param | `backend.properties().gate_length()` |
| QC7 | Readout Fidelities | Quantum Computer | metric | `backend.properties().readout_error()` |
| C1 | Qubit Assignments | Compilation | param | `transpiled.layout` |
| C2 | Gate Mappings | Compilation | artifact | transpiled QPY |
| C3 | Optimization Goals | Compilation | param | `optimization_level` |
| C4 | Random Seed | Compilation | param | `seed_transpiler` |
| C5 | Compilation Time | Compilation | metric | `time.time()` wrapper |
| E1 | Input Data | Execution | param/artifact | user-defined |
| E2 | Output Data | Execution | metric | `result[0].data` |
| E3 | Number of Shots | Execution | param | `shots` argument |
| E4 | Intermediate Results | Execution | metric (step) | optimizer callback |
| E5 | Number of Iterations | Execution | metric | optimizer callback |
| E6 | Execution Time | Execution | metric | `time.time()` wrapper |
| E7 | Applied Error Mitigation | Execution | param | user-defined string (readout mitigation only) |

---

*This document is part of the RAG knowledge base for AI-Assisted Experiment Tracking in Quantum Software Development (EM4QS project, University of Jyväskylä). It is intended for ingestion into a FAISS vector index. Each field definition (H3 section) is designed to be a self-contained retrieval chunk.*