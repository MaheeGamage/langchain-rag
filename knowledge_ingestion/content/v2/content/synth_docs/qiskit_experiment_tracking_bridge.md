# Qiskit APIs for QProv-Aligned Experiment Tracking

**Document purpose:** A complete mapping of Qiskit APIs to all 26 QProv provenance fields across all four QProv categories: Quantum Circuit, Quantum Computer, Compilation, and Execution. Each section is a self-contained retrieval unit. Intended for use with MLflow logging in the Qubernetes/Jupyter environment.

**QProv source:** Weder et al. (2021), "QProv: A Provenance System for Quantum Computing", IET Quantum Communication, DOI: 10.1049/qtc2.12012

---

## Category 1: Quantum Circuit Properties

These properties are extracted from a `QuantumCircuit` object before execution. They describe the structure and composition of the circuit itself, independent of hardware.

---

### Q1 — Used Gates

Log the gate types present in the circuit. `circuit.count_ops()` returns an `OrderedDict` mapping gate names to counts.

```python
from qiskit import QuantumCircuit
import mlflow

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)

# Log gate types as a parameter (Q1 - Used Gates)
gate_ops = qc.count_ops()
mlflow.log_param("used_gates", str(list(gate_ops.keys())))
# e.g. "['h', 'cx']"

# Log individual gate counts for finer provenance
for gate_name, count in gate_ops.items():
    mlflow.log_metric(f"gate_{gate_name}_count", count)
```

---

### Q2 — Used Measurements

Log the number and type of measurement operations in the circuit.

```python
# Q2 - Used Measurements
gate_ops = qc.count_ops()
num_measurements = gate_ops.get("measure", 0)
mlflow.log_param("num_measurements", num_measurements)

# For mid-circuit measurements, also log separately
num_mid_circuit_measurements = gate_ops.get("measure_2", 0)
mlflow.log_param("num_mid_circuit_measurements", num_mid_circuit_measurements)
```

**Note:** `measure` counts terminal measurements added via `circuit.measure()` or `circuit.measure_all()`. Mid-circuit measurements using `MidCircuitMeasure` appear as `measure_2` in `count_ops()`.

---

### Q3 — Execution Order

The full execution order is best preserved as a QPY artifact. This captures the exact ordered list of `CircuitInstruction` objects, including gate sequence and qubit assignments.

```python
from qiskit import qpy
import mlflow

# Q3 - Execution Order: preserved via QPY artifact
circuit_path = "circuit.qpy"
with open(circuit_path, "wb") as f:
    qpy.dump(qc, f)

mlflow.log_artifact(circuit_path)
# The QPY file preserves circuit.data — the ordered list of CircuitInstructions
```

To reload and inspect the execution order:

```python
with open("circuit.qpy", "rb") as f:
    restored = qpy.load(f)[0]  # always returns a list

# Inspect execution order
for i, instruction in enumerate(restored.data):
    print(f"Step {i}: {instruction.operation.name} on qubits {instruction.qubits}")
```

---

### Q4 — Circuit Width

Per the QProv specification, circuit width is the number of **qubits** used in the circuit. Use `circuit.num_qubits`, not `circuit.width()` — Qiskit's `width()` returns qubits plus classical bits combined, which is broader than the QProv definition.

```python
# Q4 - Circuit Width (QProv: number of qubits only)
mlflow.log_param("circuit_width", qc.num_qubits)

# Do NOT use circuit.width() for QProv alignment:
# circuit.width() = num_qubits + num_clbits (Qiskit-specific, not QProv)
# Log classical bits separately if needed:
mlflow.log_param("num_clbits", qc.num_clbits)
```

---

### Q5 — Circuit Depth

Per the QProv specification, circuit depth is the maximum number of gates executed sequentially on any single qubit. This measures the longest sequential chain of operations in the circuit.

```python
# Q5 - Circuit Depth
mlflow.log_metric("circuit_depth", qc.depth())
```

Circuit depth is critical because:
- It correlates with execution time on hardware.
- Deeper circuits accumulate more noise on NISQ devices.
- Decoherence limits (T1, T2) define the maximum executable depth.

```python
# Example: log pre- and post-transpilation depth for comparison
from qiskit import transpile

depth_before = qc.depth()
transpiled = transpile(qc, backend, optimization_level=3)
depth_after = transpiled.depth()

mlflow.log_metric("circuit_depth_logical", depth_before)
mlflow.log_metric("circuit_depth_physical", depth_after)
```

---

### Q6 — Circuit Size

Circuit size is the total number of gate operations in the circuit.

```python
# Q6 - Circuit Size
mlflow.log_metric("circuit_size", qc.size())
```

Circuit size influences the cumulative gate error in results — a larger circuit accumulates more individual gate errors, degrading output fidelity.

---

### Q7 — Applied Encoding

The encoding scheme used to embed classical input data into the circuit is a user-defined provenance parameter. Log it as a string before execution.

```python
# Q7 - Applied Encoding
# Common encoding types: "amplitude", "angle", "basis", "none"
mlflow.log_param("encoding_type", "angle")

# For angle encoding, also log the input data that was encoded
import numpy as np
input_data = np.array([0.5, 1.2, 0.8])
mlflow.log_param("input_data_encoded", str(input_data.tolist()))
```

**Common encoding types in Qiskit:**

| Encoding | Description | QProv value |
|---|---|---|
| Basis encoding | Input mapped to computational basis states via X gates | `"basis"` |
| Angle encoding | Input values used as rotation angles (RX, RY, RZ gates) | `"angle"` |
| Amplitude encoding | Input encoded in amplitudes of superposition state | `"amplitude"` |
| None | No encoding — circuit does not take classical input | `"none"` |

---

## Category 2: Quantum Computer Properties

These properties describe the hardware backend. For real QPU execution, log these from the backend object. For simulation, log the simulator name and relevant settings.

---

### QC1 — Number of Qubits

```python
# QC1 - Number of Qubits provided by the device
mlflow.log_param("backend_num_qubits", backend.num_qubits)
mlflow.log_param("backend_name", backend.name)
```

The backend must provide at least as many qubits as the circuit width (Q4). Logging both together enables hardware compatibility analysis.

```python
# Hardware compatibility check and logging
circuit_width = qc.num_qubits
backend_qubits = backend.num_qubits
mlflow.log_param("circuit_width", circuit_width)
mlflow.log_param("backend_num_qubits", backend_qubits)
mlflow.log_param("hardware_compatible", circuit_width <= backend_qubits)
```

---

### QC2 — Decoherence Times

T1 (energy relaxation) and T2 (dephasing) times define the coherence window for circuit execution. These change between calibrations and must be recorded at execution time.

```python
# QC2 - Decoherence Times (requires real backend with properties)
props = backend.properties()

for qubit_idx in range(backend.num_qubits):
    t1 = props.t1(qubit_idx)   # in seconds
    t2 = props.t2(qubit_idx)   # in seconds
    mlflow.log_param(f"t1_qubit_{qubit_idx}_us", round(t1 * 1e6, 2))  # convert to µs
    mlflow.log_param(f"t2_qubit_{qubit_idx}_us", round(t2 * 1e6, 2))
```

**Why this matters:** A circuit with depth D and average gate time G requires approximately D × G microseconds to execute. If this exceeds the T2 time of any used qubit, results will be significantly degraded.

---

### QC3 — Qubit Connectivity

The coupling map defines which qubit pairs can directly interact. Gates on non-adjacent qubits require SWAP insertions, increasing depth and error.

```python
# QC3 - Qubit Connectivity
coupling_map = backend.coupling_map
mlflow.log_param("coupling_map", str(list(coupling_map)))
# e.g. "[[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1], [3, 4], [4, 3]]"

# Also useful: number of connected qubit pairs
mlflow.log_param("num_qubit_connections", len(list(coupling_map)))
```

---

### QC4 — Gate Set

The native gate set (basis gates) of the device. Circuits are transpiled to use only these gates.

```python
# QC4 - Gate Set
mlflow.log_param("basis_gates", str(backend.basis_gates))
# e.g. "['cx', 'id', 'rz', 'sx', 'x']"
```

---

### QC5 — Gate Fidelities

Gate fidelity is 1 minus the gate error rate. Log error rates (lower is better) for the gates used in the circuit.

```python
# QC5 - Gate Fidelities
props = backend.properties()

# Single-qubit gate errors for each qubit
for qubit_idx in range(backend.num_qubits):
    try:
        sx_error = props.gate_error("sx", qubit_idx)
        mlflow.log_metric(f"sx_error_qubit_{qubit_idx}", sx_error)
    except Exception:
        pass

# Two-qubit (CX) gate errors for each connected pair
for edge in backend.coupling_map:
    try:
        cx_error = props.gate_error("cx", edge)
        mlflow.log_metric(f"cx_error_q{edge[0]}_q{edge[1]}", cx_error)
    except Exception:
        pass
```

---

### QC6 — Gate Times

Physical execution duration of each gate type, in nanoseconds. Longer gate times mean more decoherence during execution.

```python
# QC6 - Gate Times
props = backend.properties()

for qubit_idx in range(backend.num_qubits):
    try:
        sx_time = props.gate_length("sx", qubit_idx)  # in seconds
        mlflow.log_param(f"sx_time_qubit_{qubit_idx}_ns", round(sx_time * 1e9, 2))
    except Exception:
        pass

for edge in backend.coupling_map:
    try:
        cx_time = props.gate_length("cx", edge)
        mlflow.log_param(f"cx_time_q{edge[0]}_q{edge[1]}_ns", round(cx_time * 1e9, 2))
    except Exception:
        pass
```

---

### QC7 — Readout Fidelities

Readout fidelity measures how accurately the hardware reads out qubit states. Readout errors corrupt the output distribution and are the basis for E7 mitigation.

```python
# QC7 - Readout Fidelities
props = backend.properties()

for qubit_idx in range(backend.num_qubits):
    readout_error = props.readout_error(qubit_idx)
    mlflow.log_metric(f"readout_error_qubit_{qubit_idx}", readout_error)

# Log average readout error across all qubits
all_readout_errors = [props.readout_error(i) for i in range(backend.num_qubits)]
mlflow.log_metric("avg_readout_error", sum(all_readout_errors) / len(all_readout_errors))
```

---

### Logging All Hardware Properties Together

```python
import mlflow

def log_backend_properties(backend):
    """Log all QProv Quantum Computer category fields (QC1–QC7)."""
    props = backend.properties()

    # QC1 - Number of Qubits
    mlflow.log_param("backend_name", backend.name)
    mlflow.log_param("backend_num_qubits", backend.num_qubits)

    # QC3 - Qubit Connectivity
    mlflow.log_param("coupling_map", str(list(backend.coupling_map)))

    # QC4 - Gate Set
    mlflow.log_param("basis_gates", str(backend.basis_gates))

    for qubit_idx in range(backend.num_qubits):
        # QC2 - Decoherence Times
        mlflow.log_param(f"t1_q{qubit_idx}_us", round(props.t1(qubit_idx) * 1e6, 2))
        mlflow.log_param(f"t2_q{qubit_idx}_us", round(props.t2(qubit_idx) * 1e6, 2))

        # QC5 - Gate Fidelities (single-qubit)
        try:
            mlflow.log_metric(f"sx_error_q{qubit_idx}", props.gate_error("sx", qubit_idx))
        except Exception:
            pass

        # QC6 - Gate Times (single-qubit)
        try:
            mlflow.log_param(f"sx_time_q{qubit_idx}_ns", round(props.gate_length("sx", qubit_idx) * 1e9, 2))
        except Exception:
            pass

        # QC7 - Readout Fidelities
        mlflow.log_metric(f"readout_error_q{qubit_idx}", props.readout_error(qubit_idx))

    for edge in backend.coupling_map:
        # QC5 - Gate Fidelities (two-qubit)
        try:
            mlflow.log_metric(f"cx_error_q{edge[0]}_q{edge[1]}", props.gate_error("cx", edge))
        except Exception:
            pass

        # QC6 - Gate Times (two-qubit)
        try:
            mlflow.log_param(f"cx_time_q{edge[0]}_q{edge[1]}_ns", round(props.gate_length("cx", edge) * 1e9, 2))
        except Exception:
            pass
```

---

## Category 3: Compilation Properties

These properties describe the transpilation process. Log them around the `transpile()` call.

---

### C1 — Qubit Assignments

The mapping from logical qubits in the circuit to physical qubits on the hardware, determined by the transpiler.

```python
from qiskit import transpile

transpiled = transpile(qc, backend, seed_transpiler=42, optimization_level=3)

# C1 - Qubit Assignments
if transpiled.layout is not None:
    final_layout = transpiled.layout.final_layout
    mlflow.log_param("qubit_layout_final", str(final_layout))

    initial_layout = transpiled.layout.initial_layout
    mlflow.log_param("qubit_layout_initial", str(initial_layout))
```

---

### C2 — Gate Mappings

The transpiled circuit contains the full gate mapping — logical gates decomposed into native basis gates with physical qubit assignments. Log the transpiled circuit as a QPY artifact.

```python
# C2 - Gate Mappings: preserved in the transpiled circuit artifact
transpiled_path = "circuit_transpiled.qpy"
with open(transpiled_path, "wb") as f:
    qpy.dump(transpiled, f)

mlflow.log_artifact(transpiled_path)

# Also log how many gates were added during transpilation (SWAP overhead)
mlflow.log_metric("circuit_size_after_transpile", transpiled.size())
mlflow.log_metric("circuit_depth_after_transpile", transpiled.depth())
swap_overhead = transpiled.count_ops().get("swap", 0)
mlflow.log_metric("swap_gates_added", swap_overhead)
```

---

### C3 — Optimization Goals

The optimization level and target used during transpilation.

```python
# C3 - Optimization Goals
optimization_level = 3
mlflow.log_param("transpile_optimization_level", optimization_level)
# 0 = no optimization, 1 = light, 2 = medium, 3 = heavy (default recommended)

# If using a custom pass manager, log the target configuration
mlflow.log_param("transpile_target", backend.name)
```

---

### C4 — Random Seed

The random seed controls stochastic decisions in the transpiler's routing and layout algorithms. Without a fixed seed, the same circuit may transpile differently on each run.

```python
# C4 - Random Seed
transpile_seed = 42
mlflow.log_param("transpile_seed", transpile_seed)

transpiled = transpile(
    qc,
    backend,
    seed_transpiler=transpile_seed,  # fixes stochastic transpiler behavior
    optimization_level=3
)
```

---

### C5 — Compilation Time

```python
import time

# C5 - Compilation Time
start = time.time()
transpiled = transpile(qc, backend, seed_transpiler=42, optimization_level=3)
compilation_time = time.time() - start

mlflow.log_metric("compilation_time_s", round(compilation_time, 4))
```

---

### Logging All Compilation Properties Together

```python
import time
from qiskit import transpile, qpy
import mlflow

def log_compilation(circuit, backend, optimization_level=3, seed=42):
    """Log all QProv Compilation category fields (C1–C5)."""

    # C3 - Optimization Goals
    mlflow.log_param("transpile_optimization_level", optimization_level)
    mlflow.log_param("transpile_target_backend", backend.name)

    # C4 - Random Seed
    mlflow.log_param("transpile_seed", seed)

    # C5 - Compilation Time (wrap transpile call)
    start = time.time()
    transpiled = transpile(
        circuit, backend,
        optimization_level=optimization_level,
        seed_transpiler=seed
    )
    mlflow.log_metric("compilation_time_s", round(time.time() - start, 4))

    # C1 - Qubit Assignments
    if transpiled.layout is not None:
        mlflow.log_param("qubit_layout_final", str(transpiled.layout.final_layout))
        mlflow.log_param("qubit_layout_initial", str(transpiled.layout.initial_layout))

    # C2 - Gate Mappings: save transpiled circuit as artifact
    transpiled_path = "circuit_transpiled.qpy"
    with open(transpiled_path, "wb") as f:
        qpy.dump(transpiled, f)
    mlflow.log_artifact(transpiled_path)

    # Transpilation diagnostics
    mlflow.log_metric("circuit_depth_after_transpile", transpiled.depth())
    mlflow.log_metric("circuit_size_after_transpile", transpiled.size())
    mlflow.log_metric("swap_gates_added", transpiled.count_ops().get("swap", 0))

    return transpiled
```

---

## Category 4: Execution Properties

These properties describe what happened during circuit execution.

---

### E1 — Input Data

Classical input data encoded into the circuit should be logged before execution, particularly for variational and QML algorithms.

```python
import numpy as np
import mlflow

# E1 - Input Data
input_vector = np.array([0.5, 1.2, 0.8, 0.3])

# For small inputs: log as parameter
mlflow.log_param("input_data", str(input_vector.tolist()))
mlflow.log_param("input_dimension", len(input_vector))

# For large inputs: log as artifact
np.save("input_data.npy", input_vector)
mlflow.log_artifact("input_data.npy")
```

---

### E2 — Output Data

The raw output of the execution — bitstring distributions (Sampler) or expectation values (Estimator).

```python
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

# E2 - Output Data (Sampler: bitstring distribution)
sampler = StatevectorSampler()
qc_measured = qc.measure_all(inplace=False)
result = sampler.run([qc_measured], shots=1024).result()
counts = result[0].data["meas"].get_counts()

mlflow.log_metric("num_unique_outcomes", len(counts))
for bitstring, count in counts.items():
    mlflow.log_metric(f"count_{bitstring}", count)

# E2 - Output Data (Estimator: expectation value)
estimator = StatevectorEstimator()
observable = SparsePauliOp("ZZ")
result = estimator.run([(qc, observable)]).result()
mlflow.log_param("observable", "ZZ")
mlflow.log_metric("expectation_value", float(result[0].data.evs))
```

---

### E3 — Number of Shots

```python
# E3 - Number of Shots
shots = 1024
mlflow.log_param("shots", shots)

result = sampler.run([qc_measured], shots=shots).result()
```

---

### E4 — Intermediate Results

Per QProv, intermediate results are only collectible for **variational algorithms** (VQE, QAOA) where classical-quantum iterations occur. Standard circuits cannot produce intermediate results because measurement collapses superposition.

```python
import mlflow
import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

# E4 - Intermediate Results (variational algorithms only)
optimization_history = []

def vqe_callback(iteration, params, energy):
    """Capture intermediate results per QProv E4."""
    optimization_history.append((iteration, params, energy))
    mlflow.log_metric("intermediate_energy", energy, step=iteration)
    for i, p in enumerate(params):
        mlflow.log_metric(f"param_{i}", float(p), step=iteration)

# Example variational loop
params = ParameterVector("theta", 2)
ansatz = QuantumCircuit(2)
ansatz.ry(params[0], 0)
ansatz.ry(params[1], 1)
ansatz.cx(0, 1)

estimator = StatevectorEstimator()
for iteration in range(10):
    param_values = np.random.uniform(0, np.pi, 2)
    bound = ansatz.assign_parameters(dict(zip(params, param_values)))
    energy = float(estimator.run([(bound, SparsePauliOp("ZZ"))]).result()[0].data.evs)
    vqe_callback(iteration, param_values, energy)
```

---

### E5 — Number of Iterations

```python
# E5 - Number of Iterations (variational algorithms)
total_iterations = len(optimization_history)
mlflow.log_metric("num_iterations", total_iterations)
```

---

### E6 — Execution Time

```python
import time

# E6 - Execution Time
start = time.time()
result = sampler.run([qc_measured], shots=1024).result()
execution_time = time.time() - start

mlflow.log_metric("execution_time_s", round(execution_time, 4))
```

---

### E7 — Applied Error Mitigation (Readout)

Per the QProv specification, E7 refers specifically to **readout-error mitigation** — post-processing techniques that correct measurement errors in the output distribution. This does not include gate-error mitigation techniques like Zero-Noise Extrapolation, which are outside the QProv scope.

The standard approach is calibration matrix inversion: calibration circuits prepare each possible basis state, measure the results, and build a matrix that is then inverted and applied to raw execution counts.

```python
# E7 - Applied Error Mitigation (readout-error mitigation per QProv spec)

# Always log which technique was applied, even if "none"
mlflow.log_param("readout_mitigation", "calibration_matrix")
# Other valid values: "none", "iterative_bayesian_unfolding", "detector_tomography"

# Log calibration matrix metadata when applicable
mlflow.log_param("calibration_matrix_num_qubits", qc.num_qubits)
mlflow.log_param("calibration_shots", 8192)

# If no mitigation is applied, still log explicitly:
mlflow.log_param("readout_mitigation", "none")
```

**Important:** Always log this field. Even `"none"` is informative provenance — results with and without mitigation are not directly comparable.

---

## Complete QProv-Aligned Tracking Template

This template covers all 26 QProv fields across all four categories for a standard hardware execution.

```python
import time
import mlflow
import numpy as np
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.primitives import StatevectorSampler

def track_full_quantum_experiment(
    circuit: QuantumCircuit,
    backend,
    shots: int = 1024,
    optimization_level: int = 3,
    transpile_seed: int = 42,
    encoding_type: str = "none",
    input_data=None,
    readout_mitigation: str = "none",
    experiment_name: str = "quantum_experiment"
):
    """
    Full QProv-aligned experiment tracking across all four categories.
    Covers all 26 provenance fields: Q1-Q7, QC1-QC7, C1-C5, E1-E7.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():

        # ── CATEGORY 1: Quantum Circuit (Q1–Q7) ──────────────────────────

        gate_ops = circuit.count_ops()

        # Q1 - Used Gates
        mlflow.log_param("used_gates", str(list(gate_ops.keys())))
        for gate_name, count in gate_ops.items():
            mlflow.log_metric(f"gate_{gate_name}_count", count)

        # Q2 - Used Measurements
        mlflow.log_param("num_measurements", gate_ops.get("measure", 0))

        # Q3 - Execution Order (QPY artifact)
        circuit_path = "circuit_logical.qpy"
        with open(circuit_path, "wb") as f:
            qpy.dump(circuit, f)
        mlflow.log_artifact(circuit_path)

        # Q4 - Circuit Width (qubits only per QProv — NOT circuit.width())
        mlflow.log_param("circuit_width", circuit.num_qubits)

        # Q5 - Circuit Depth
        mlflow.log_metric("circuit_depth", circuit.depth())

        # Q6 - Circuit Size
        mlflow.log_metric("circuit_size", circuit.size())

        # Q7 - Applied Encoding
        mlflow.log_param("encoding_type", encoding_type)

        # ── CATEGORY 2: Quantum Computer (QC1–QC7) ───────────────────────

        props = backend.properties()

        # QC1 - Number of Qubits
        mlflow.log_param("backend_name", backend.name)
        mlflow.log_param("backend_num_qubits", backend.num_qubits)

        # QC3 - Qubit Connectivity
        mlflow.log_param("coupling_map", str(list(backend.coupling_map)))

        # QC4 - Gate Set
        mlflow.log_param("basis_gates", str(backend.basis_gates))

        for q in range(backend.num_qubits):
            # QC2 - Decoherence Times
            mlflow.log_param(f"t1_q{q}_us", round(props.t1(q) * 1e6, 2))
            mlflow.log_param(f"t2_q{q}_us", round(props.t2(q) * 1e6, 2))
            # QC5 - Gate Fidelities (single-qubit)
            try:
                mlflow.log_metric(f"sx_error_q{q}", props.gate_error("sx", q))
            except Exception:
                pass
            # QC6 - Gate Times (single-qubit)
            try:
                mlflow.log_param(f"sx_time_q{q}_ns", round(props.gate_length("sx", q) * 1e9, 2))
            except Exception:
                pass
            # QC7 - Readout Fidelities
            mlflow.log_metric(f"readout_error_q{q}", props.readout_error(q))

        for edge in backend.coupling_map:
            # QC5 - Gate Fidelities (two-qubit)
            try:
                mlflow.log_metric(f"cx_error_q{edge[0]}_q{edge[1]}", props.gate_error("cx", edge))
            except Exception:
                pass
            # QC6 - Gate Times (two-qubit)
            try:
                mlflow.log_param(f"cx_time_q{edge[0]}_q{edge[1]}_ns", round(props.gate_length("cx", edge) * 1e9, 2))
            except Exception:
                pass

        # ── CATEGORY 3: Compilation (C1–C5) ──────────────────────────────

        # C3 - Optimization Goals
        mlflow.log_param("transpile_optimization_level", optimization_level)
        mlflow.log_param("transpile_target_backend", backend.name)

        # C4 - Random Seed
        mlflow.log_param("transpile_seed", transpile_seed)

        # C5 - Compilation Time
        t_start = time.time()
        transpiled = transpile(
            circuit, backend,
            optimization_level=optimization_level,
            seed_transpiler=transpile_seed
        )
        mlflow.log_metric("compilation_time_s", round(time.time() - t_start, 4))

        # C1 - Qubit Assignments
        if transpiled.layout is not None:
            mlflow.log_param("qubit_layout_final", str(transpiled.layout.final_layout))
            mlflow.log_param("qubit_layout_initial", str(transpiled.layout.initial_layout))

        # C2 - Gate Mappings (transpiled circuit artifact)
        transpiled_path = "circuit_transpiled.qpy"
        with open(transpiled_path, "wb") as f:
            qpy.dump(transpiled, f)
        mlflow.log_artifact(transpiled_path)

        # Transpilation diagnostics
        mlflow.log_metric("circuit_depth_after_transpile", transpiled.depth())
        mlflow.log_metric("circuit_size_after_transpile", transpiled.size())
        mlflow.log_metric("swap_gates_added", transpiled.count_ops().get("swap", 0))

        # ── CATEGORY 4: Execution (E1–E7) ────────────────────────────────

        # E1 - Input Data
        if input_data is not None:
            mlflow.log_param("input_data", str(input_data))

        # E3 - Number of Shots
        mlflow.log_param("shots", shots)

        # E6 - Execution Time
        qc_measured = transpiled.measure_all(inplace=False)
        sampler = StatevectorSampler()
        e_start = time.time()
        result = sampler.run([qc_measured], shots=shots).result()
        mlflow.log_metric("execution_time_s", round(time.time() - e_start, 4))

        # E2 - Output Data
        counts = result[0].data["meas"].get_counts()
        mlflow.log_metric("num_unique_outcomes", len(counts))
        for bitstring, count in counts.items():
            mlflow.log_metric(f"count_{bitstring}", count)

        # E7 - Applied Error Mitigation (readout only per QProv spec)
        mlflow.log_param("readout_mitigation", readout_mitigation)


# ── Usage ─────────────────────────────────────────────────────────────────

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

track_full_quantum_experiment(
    circuit=qc,
    backend=backend,
    shots=1024,
    optimization_level=3,
    transpile_seed=42,
    encoding_type="none",
    readout_mitigation="none",
    experiment_name="bell_state_full_provenance"
)
```

---

## QProv Field Coverage Reference

| QProv ID | Field Name | Category | MLflow Key | Qiskit API |
|---|---|---|---|---|
| Q1 | Used Gates | Quantum Circuit | `used_gates`, `gate_<n>_count` | `circuit.count_ops()` |
| Q2 | Used Measurements | Quantum Circuit | `num_measurements` | `circuit.count_ops().get("measure")` |
| Q3 | Execution Order | Quantum Circuit | artifact `circuit_logical.qpy` | `circuit.data` / QPY |
| Q4 | Circuit Width | Quantum Circuit | `circuit_width` | `circuit.num_qubits` ⚠️ not `circuit.width()` |
| Q5 | Circuit Depth | Quantum Circuit | `circuit_depth` | `circuit.depth()` |
| Q6 | Circuit Size | Quantum Circuit | `circuit_size` | `circuit.size()` |
| Q7 | Applied Encoding | Quantum Circuit | `encoding_type` | user-defined string |
| QC1 | Number of Qubits | Quantum Computer | `backend_num_qubits` | `backend.num_qubits` |
| QC2 | Decoherence Times | Quantum Computer | `t1_q<n>_us`, `t2_q<n>_us` | `backend.properties().t1/t2(qubit)` |
| QC3 | Qubit Connectivity | Quantum Computer | `coupling_map` | `backend.coupling_map` |
| QC4 | Gate Set | Quantum Computer | `basis_gates` | `backend.basis_gates` |
| QC5 | Gate Fidelities | Quantum Computer | `sx_error_q<n>`, `cx_error_q<n>_q<m>` | `backend.properties().gate_error()` |
| QC6 | Gate Times | Quantum Computer | `sx_time_q<n>_ns`, `cx_time_q<n>_q<m>_ns` | `backend.properties().gate_length()` |
| QC7 | Readout Fidelities | Quantum Computer | `readout_error_q<n>` | `backend.properties().readout_error(qubit)` |
| C1 | Qubit Assignments | Compilation | `qubit_layout_final` | `transpiled.layout.final_layout` |
| C2 | Gate Mappings | Compilation | artifact `circuit_transpiled.qpy` | transpiled `QuantumCircuit` |
| C3 | Optimization Goals | Compilation | `transpile_optimization_level` | `optimization_level` arg in `transpile()` |
| C4 | Random Seed | Compilation | `transpile_seed` | `seed_transpiler` arg in `transpile()` |
| C5 | Compilation Time | Compilation | `compilation_time_s` | `time.time()` around `transpile()` |
| E1 | Input Data | Execution | `input_data` | user-defined |
| E2 | Output Data | Execution | `count_<bitstring>` or `expectation_value` | `result[0].data` |
| E3 | Number of Shots | Execution | `shots` | `shots` arg in `sampler.run()` |
| E4 | Intermediate Results | Execution | `intermediate_energy` (step=iter) | optimizer callback — variational only |
| E5 | Number of Iterations | Execution | `num_iterations` | optimizer callback — variational only |
| E6 | Execution Time | Execution | `execution_time_s` | `time.time()` around `sampler.run()` |
| E7 | Applied Error Mitigation | Execution | `readout_mitigation` | user-defined string — readout only per QProv |

---

*This document is part of the RAG knowledge base for AI-Assisted Experiment Tracking in Quantum Software Development (EM4QS project, University of Jyväskylä). It is intended for ingestion into a FAISS vector index alongside MLflow documentation and the QProv specification. Each H3 section is a self-contained retrieval chunk.*