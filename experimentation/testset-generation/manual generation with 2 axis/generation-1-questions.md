LLM used: Gemini 3.1 Pro

[SD1: NISQ constraints & QSD challenges]

[fact_single] - VALID
Q: According to the definition of the current era of quantum hardware, what does the acronym NISQ stand for and what is the typical number of qubits expected for these devices?
A: NISQ stands for Noisy Intermediate-Scale Quantum. It refers to quantum computers with a number of qubits ranging from 50 to a few hundred. 
Source: Preskill - 2018 (Section 4.1)

[summary] - VALID
Q: How does the focus of data collection shift as a quantum program progresses through the software development lifecycle?
A: In the early stages of development, quantum programs are run on simulators, and data collection focuses primarily on the algorithm or program itself. As the program matures and transitions to execution on actual quantum processing units (QPUs), data collection must expand to include hardware-specific metrics such as qubit quality and calibration data, which are necessary to navigate the unique constraints of the quantum environment. 
Source: Gamage et al. - 2025 (Section I. Introduction)

[reasoning] - LOW VALID
Q: Based on the constraints of near-term quantum hardware, why is there a strict ceiling limiting the execution of circuits to approximately 1000 fundamental two-qubit operations?
A: There is a strict ceiling on circuit size because the best available near-term hardware has two-qubit gate error rates above the 0.1% level. If a quantum algorithm requires significantly more than 1000 gates, the accumulated noise from these imperfect operations will overwhelm the computational signal. Furthermore, because implementing full quantum error correction requires a massive overhead in additional physical qubits, near-term devices must operate without it, preventing the reliable execution of larger, more complex circuits. 

[unanswerable] - LOW VALID
Q: What are the exact T1 and T2 decoherence times for the 50-qubit IBM quantum processor discussed as a milestone for the NISQ era?
A: The provided documents do not contain sufficient information to answer this question.
COMMENT FROM VALIDATION: Find some different question. These seems more domain oriented to IBM.


[SD2: Experiment tracking fundamentals]
Uploaded docs: mlflow paper and mlflow tracking docs
> [!note] May need to regenerate since these questions may not properly align with the subdomain

```
[fact_single] - INVALID
Q: In the context of adopting structured tracking for our workflows, how does MLflow define a "run," which we would use to represent a single execution of a quantum circuit or program?
A: A run is defined as a single execution of a piece of data science code that records metadata—such as parameters, metrics, start and end times—as well as artifacts, which are output files produced by the execution.
Source: MLflow Tracking document, "Concepts" -> "Runs" section; Zaharia et al. document, Section 3.1.
COMMENT FROM VALIDATION: contain more technical concept about MLflow. This is not properly related to this sub domain

[summary] - VALID
Q: As quantum developers, we struggle with recreating past environments and tracking complex iterative workflows. Based on the provided literature, what are the primary lifecycle challenges that experiment tracking platforms are designed to resolve?
A: Experiment tracking platforms are designed to address four main lifecycle challenges: managing the multitude of diverse software tools and libraries used across different phases of development; tracking the vast number of configurable parameters, code versions, and data inputs that go into each experiment; ensuring the reproducibility of results so that previous setups can be reliably recreated; and facilitating the reliable production deployment of models and workflows into diverse inference or scheduling environments.
Source: Zaharia et al. document, Section 2 (Challenges in Machine Learning Development).

[reasoning] - OKAY VALID
Q: If a developer logs specific input configurations (like algorithm hyperparameters) and resulting performance evaluations, how does the tracking framework differentiate between these data types, and what mechanism allows developers to correlate them later?
A: The framework differentiates these data types by categorizing the input configurations as "parameters" (which are arbitrary key-value pairs representing inputs) and the performance evaluations as "metrics" (which represent outcomes that can be updated over time). The mechanism that allows them to be correlated is the "run" concept; because both parameters and metrics are recorded within the same execution instance (run), developers can use the tracking API or UI to search, filter, and compare multiple runs to observe how different input parameters influenced the resulting metrics.
COMMENT FROM VALIDATION: Seems somewhat grounded on MLflow related concepts but can give it as a pass

[unanswerable] - INVALID
COMMENT FROM VALIDATION: Too much based on MLflow and QProv
Q: How does the MLflow tracking API map its native metric logging functions to the specific taxonomy of the QProv data schema when recording quantum hardware calibration data?
A: The provided documents do not contain sufficient information to answer this question.
```

## SD3: QProv Provenance Taxonomy
Uploaded docs: qprov paper

```
[fact_single] - LOW VALID
Q: In the QProv extension of the PROV meta-model, which core PROV element does a "Quantum Computer" extend?
A: A Quantum Computer extends the PROV "agent" element, because it performs an activity when executing a quantum circuit.
Source: Section 4.2

[summary] - Valid
Q: What specific provenance attributes does the taxonomy collect during the compilation phase, and why is this data necessary for reproducibility?
A: The taxonomy collects qubit assignments, gate mappings, optimization goals, the random seed, and compilation time. Tracking these attributes is necessary because the compiler maps abstract circuits to physical qubits and subroutines, which directly influences the execution time and error probability. Specifically, because mapping is an NP-hard problem, compilers often use randomization; therefore, capturing the random seed is essential to ensure the resulting mappings can be reproduced.
Source: Section 3.3

[reasoning] - VALID
Q: Why does the execution category of the QProv taxonomy emphasize collecting "intermediate results" and the "number of iterations" for variational algorithms, but note it as usually impossible for standard algorithms?
A: In standard quantum algorithms, measuring a qubit destroys its superposition, meaning intermediate state data cannot be gathered during an execution without disrupting the computation. However, variational algorithms, such as VQE and QAOA, naturally consist of multiple distinct iterations that alternate between quantum execution and classical processing. Because of this inherent hybrid loop, intermediate results are systematically generated at the end of each quantum iteration, making it possible to record them alongside the total number of iterations required.

[unanswerable]
Q: How are the QProv quantum computer attributes, such as gate fidelities and decoherence times, structurally mapped to MLflow tags versus MLflow metrics during an experiment run?
A: The provided documents do not contain sufficient information to answer this question.
```

## SD4: MLflow Tracking API

```
[fact_single] - VALID
Q: Which API function allows me to log an arbitrary output file, such as an image of a result histogram, to an active experiment run? 
A: You can log an arbitrary output file by using the `mlflow.log_artifact()` function. Source: Zaharia et al. PDF, Section 3.1 MLflow Tracking

[summary] - VALID
Q: As my quantum experimentation scales, how does MLflow structurally organize and group the tracking data from my code executions? 
A: MLflow structures tracking data primarily around "runs," which represent individual executions of data science code and store specific metadata (such as metrics, parameters, and start/end times) alongside output artifacts. To organize these executions on a larger scale, runs are grouped into "experiments" that focus on a specific task. For even finer organization, users can create hierarchical parent-child relationships between runs (such as grouping cross-validation folds) and apply arbitrary tags to filter and search through them. 
Source: mlflow tracking.md (Sections "Concepts" and "FAQ: How can I organize many MLflow Runs neatly?")

[reasoning] - NOT VALID
COMMENT: The questions contain mlflow model, which not relevant to this domain.
Q: If I am iteratively training a parameterized quantum circuit and want to evaluate its performance at various stages, why is it beneficial to log metrics using MLflow 3's specific checkpoint tracking rather than just saving a final model at the end of the run? 
A: Logging metrics using MLflow 3's checkpoint tracking is beneficial because it enables detailed traceability and comparison of intermediate states. By logging a model at a specific `step`, MLflow generates a unique `model_id` for that checkpoint. If you then pass this specific `model_id` when calling `mlflow.log_metric()`, the metric is explicitly linked to that exact checkpoint rather than just the overall run. This allows you to programmatically search, rank, and compare the performance of different checkpoints within the same training loop, which would be impossible if you only saved a single final model path.

[unanswerable] 
Q: How do I configure the `mlflow.log_dict()` function to automatically parse and map standard QProv hardware calibration properties into an MLflow run? 
A: The provided documents do not contain sufficient information to answer this question.
```

## SD5: Qiskit-Specific Experiment Tracking Using MLflow and QProv
uploaded docs: qiskit_experiment_tracking_bridge.md, link to qiskit docs

```
[fact_single] - VALID
Q: According to the QProv specification, what Qiskit property should be used to log the "Circuit Width" (Q4), and which property should explicitly be avoided?
A: To correctly align with the QProv specification, you must use circuit.num_qubits to log the circuit width, as QProv defines width solely by the number of qubits used. You should avoid using circuit.width(), because in Qiskit, this method returns the combined total of both quantum and classical bits, which violates the QProv definition.
Source: Category 1: Quantum Circuit Properties, Subsection: Q4 — Circuit Width

[summary]
Q: During the quantum experiment tracking lifecycle, for which specific QProv fields is it recommended to save and log physical artifact files to MLflow rather than using simple parameters or metrics?
A: Artifact files are utilized to capture complex structural data or large arrays that do not fit into standard parameters or metrics:

Execution Order (Q3) and Gate Mappings (C2): QPY artifact files (.qpy) are saved to preserve the exact ordered list of CircuitInstruction objects for the logical circuit, as well as the fully transpiled circuit containing native basis gates and physical qubit assignments.

Input Data (E1): NumPy artifact files (.npy) are saved to log large classical input vectors that are encoded into the circuit, whereas small inputs might just be logged as string parameters.
Source: Category 1 (Q3 — Execution Order), Category 3 (C2 — Gate Mappings), and Category 4 (E1 — Input Data)

[reasoning]
Q: When evaluating the feasibility of running a specific quantum circuit on a hardware backend, why is it critical to simultaneously analyze the tracked "Circuit Depth" (Q5) alongside the backend's "Decoherence Times" (QC2)?
A: "Circuit Depth" measures the longest sequential chain of gate operations on any single qubit, while "Decoherence Times" (like T1 and T2) dictate the physical time limits before a qubit loses its quantum state. Because each gate requires a physical duration (QC6), a deeper circuit takes longer to run. If the total execution time (depth multiplied by average gate time) surpasses the hardware's T2 decoherence time, the qubits will decohere before the computation finishes, resulting in heavily degraded and noisy outputs. Analyzing both fields together determines if the circuit can successfully execute within the hardware's coherence window.

[unanswerable]
Q: How can I extract and log the specific IBM Quantum job ID assigned by the Qiskit Runtime service to link my MLflow run with the IBM Quantum dashboard?
A: The provided documents do not contain sufficient information to answer this question.
```

# OLD ()
## Subdomain: NISQ constraints & QSD challenges
Uploaded docs: Forgot to upload any
```
[fact_single]

Q: What specific hardware calibration characteristics are noted to shift between different quantum program runs?

A: Decoherence times and gate fidelities shift between runs.

Source: Subdomain Description (Weder et al., 2021)

[summary]

Q: What factors contribute to the difficulty of reproducing quantum experiments on current-generation hardware?

A: Reproducibility is challenged by the inherent noise and limited qubit count/quality of NISQ devices, the probabilistic nature of the outputs (which requires many repetitions to obtain reliable distributions), and the continuous shifting of hardware calibration characteristics between runs.

Source: Subdomain Description (Synthesis of Preskill, 2018; Gamage et al., 2025; and Weder et al., 2021)

[reasoning]

Q: Based on the physical constraints of NISQ devices, why is the staged progression from simulation to QPU execution a necessary part of the quantum software development lifecycle?

A: Because actual QPUs suffer from inherent noise, limited qubit quality, and shifting calibration metrics that make outputs probabilistic and difficult to reproduce. Starting the lifecycle in a simulation environment likely allows developers to logically test and validate the complexity of their programs without interference from the shifting physical unreliability and noise of an actual QPU.

[unanswerable]

Q: Which specific MLflow or QProv tracking API functions should a developer call to log the shifting decoherence times of a QPU during an experiment run?

A: The provided documents do not contain sufficient information to answer this question.
```
