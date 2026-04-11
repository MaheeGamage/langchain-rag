Total Questions: 15

[SD1: NISQ constraints & QSD challenges]

[fact_single] - VALID
Q: What does the acronym NISQ stand for and what is the typical number of qubits expected for these devices?
A: NISQ stands for Noisy Intermediate-Scale Quantum. It refers to quantum computers with a number of qubits ranging from 50 to a few hundred. 
Source: Preskill - 2018 (Section 4.1)

[summary] - VALID
Q: How does the focus of data collection shift as a quantum program progresses through the software development lifecycle?
A: In the early stages of development, quantum programs are run on simulators, and data collection focuses primarily on the algorithm or program itself. As the program matures and transitions to execution on actual quantum processing units (QPUs), data collection must expand to include hardware-specific metrics such as qubit quality and calibration data, which are necessary to navigate the unique constraints of the quantum environment. 
Source: Gamage et al. - 2025 (Section I. Introduction)

[reasoning] - LOW VALID
Q: Based on the constraints of near-term quantum hardware, why is there a strict ceiling limiting the execution of circuits to approximately 1000 fundamental two-qubit operations?
A: There is a strict ceiling on circuit size because the best available near-term hardware has two-qubit gate error rates above the 0.1% level. If a quantum algorithm requires significantly more than 1000 gates, the accumulated noise from these imperfect operations will overwhelm the computational signal. Furthermore, because implementing full quantum error correction requires a massive overhead in additional physical qubits, near-term devices must operate without it, preventing the reliable execution of larger, more complex circuits. 



[SD2: Experiment tracking fundamentals]

[fact_single] - OKAY VALID
LLM : Claude Opus 4.6
Q: What are the three types of data that can be logged per experiment run using MLflow Tracking?
A: Parameters (key-value pairs), metrics (quantitative measures that can be updated during a run), and artifacts (arbitrary output files).
Source: Section 3.1 (MLflow Tracking), API code example.

[summary] - VALID
Q: As quantum developers, we struggle with recreating past environments and tracking complex iterative workflows. Based on the provided literature, what are the primary lifecycle challenges that experiment tracking platforms are designed to resolve?
A: Experiment tracking platforms are designed to address four main lifecycle challenges: managing the multitude of diverse software tools and libraries used across different phases of development; tracking the vast number of configurable parameters, code versions, and data inputs that go into each experiment; ensuring the reproducibility of results so that previous setups can be reliably recreated; and facilitating the reliable production deployment of models and workflows into diverse inference or scheduling environments.
Source: Zaharia et al. document, Section 2 (Challenges in Machine Learning Development).

[reasoning] - OKAY VALID
Q: If a developer logs specific input configurations (like algorithm hyperparameters) and resulting performance evaluations, how does the tracking framework differentiate between these data types, and what mechanism allows developers to correlate them later?
A: The framework differentiates these data types by categorizing the input configurations as "parameters" (which are arbitrary key-value pairs representing inputs) and the performance evaluations as "metrics" (which represent outcomes that can be updated over time). The mechanism that allows them to be correlated is the "run" concept; because both parameters and metrics are recorded within the same execution instance (run), developers can use the tracking API or UI to search, filter, and compare multiple runs to observe how different input parameters influenced the resulting metrics.



[SD3: QProv Provenance Taxonomy]

[fact_single] - VALID
Q: Within the quantum circuit category of the QProv taxonomy, which specific attribute dictates the cumulative gate error that is reflected in the execution results?
A: The circuit size, which represents the total number of gates executed in the quantum circuit.

[summary] - VALID
Q: What specific provenance attributes does the taxonomy collect during the compilation phase, and why is this data necessary for reproducibility?
A: The taxonomy collects qubit assignments, gate mappings, optimization goals, the random seed, and compilation time. Tracking these attributes is necessary because the compiler maps abstract circuits to physical qubits and subroutines, which directly influences the execution time and error probability. Specifically, because mapping is an NP-hard problem, compilers often use randomization; therefore, capturing the random seed is essential to ensure the resulting mappings can be reproduced.
Source: Section 3.3

[reasoning] - VALID
Q: Why does the execution category of the QProv taxonomy emphasize collecting "intermediate results" and the "number of iterations" for variational algorithms, but note it as usually impossible for standard algorithms?
A: In standard quantum algorithms, measuring a qubit destroys its superposition, meaning intermediate state data cannot be gathered during an execution without disrupting the computation. However, variational algorithms, such as VQE and QAOA, naturally consist of multiple distinct iterations that alternate between quantum execution and classical processing. Because of this inherent hybrid loop, intermediate results are systematically generated at the end of each quantum iteration, making it possible to record them alongside the total number of iterations required.



[SD4: MLflow Tracking API]

[fact_single] - VALID
Q: Which API function allows me to log an arbitrary output file, such as an image of a result histogram, to an active experiment run? 
A: You can log an arbitrary output file by using the `mlflow.log_artifact()` function. Source: Zaharia et al. PDF, Section 3.1 MLflow Tracking

[summary] - VALID
Q: As my quantum experimentation scales, how does MLflow structurally organize and group the tracking data from my code executions? 
A: MLflow structures tracking data primarily around "runs," which represent individual executions of data science code and store specific metadata (such as metrics, parameters, and start/end times) alongside output artifacts. To organize these executions on a larger scale, runs are grouped into "experiments" that focus on a specific task. For even finer organization, users can create hierarchical parent-child relationships between runs (such as grouping cross-validation folds) and apply arbitrary tags to filter and search through them. 
Source: mlflow tracking.md (Sections "Concepts" and "FAQ: How can I organize many MLflow Runs neatly?")

[reasoning] - VALID
Q: A quantum researcher is executing a hybrid quantum-classical loop where the number of measurement shots is fixed at 8192, but the circuit's variational rotation angles are continuously updated across 100 optimization steps. Based on the functional differences in the MLflow Tracking API, which specific logging functions should be used for the shot count versus the rotation angles, and why?
A: The researcher should use mlflow.log_param() for the fixed shot count and mlflow.log_metric() for the dynamically changing rotation angles. The provided documentation defines parameters as arbitrary key-value pairs that record the configuration of a run. Because the shot count remains constant at 8192, it acts as a baseline configuration parameter. Conversely, metrics are explicitly designed so that "each metric can also be updated throughout the run". Since the variational angles change continuously across the 100 optimization steps, they must be tracked as metrics to properly capture their evolution over time.



[SD5: Qiskit-Specific Experiment Tracking Using MLflow and QProv]

[fact_single] - VALID
Q: According to the QProv specification, what Qiskit property should be used to log the "Circuit Width" (Q4), and which property should explicitly be avoided?
A: To correctly align with the QProv specification, you must use circuit.num_qubits to log the circuit width, as QProv defines width solely by the number of qubits used. You should avoid using circuit.width(), because in Qiskit, this method returns the combined total of both quantum and classical bits, which violates the QProv definition.

[summary] - VALID
Q: During the quantum experiment tracking lifecycle, for which specific QProv fields is it recommended to save and log physical artifact files to MLflow rather than using simple parameters or metrics?
A: Artifact files are utilized to capture complex structural data or large arrays that do not fit into standard parameters or metrics. Specifically, QPY artifact files (.qpy) are used to log the execution order (Q3) and gate mappings (C2) to preserve the exact ordered list of CircuitInstruction objects for the logical circuit, as well as the fully transpiled circuit containing native basis gates and physical qubit assignments. Additionally, NumPy artifact files (.npy) are saved to log large classical input vectors (E1) that are encoded into the circuit, whereas small inputs might just be logged as string parameters.

[reasoning] - VALID
Q: When evaluating the feasibility of running a specific quantum circuit on a hardware backend, why is it critical to simultaneously analyze the tracked "Circuit Depth" (Q5) alongside the backend's "Decoherence Times" (QC2)?
A: "Circuit Depth" measures the longest sequential chain of gate operations on any single qubit, while "Decoherence Times" (T1 and T2) dictate the physical time limits before individual qubits lose their quantum state. Because each gate requires a finite physical duration (QC6), a deeper circuit takes longer to execute. If the total execution time exceeds the decoherence time of any qubit involved in that critical path, those qubits will decohere before the computation finishes, resulting in degraded and noisy outputs. Since decoherence times also vary across qubits and change between hardware calibrations, analyzing both fields together is essential to determine whether the circuit can successfully execute within the hardware's coherence window.