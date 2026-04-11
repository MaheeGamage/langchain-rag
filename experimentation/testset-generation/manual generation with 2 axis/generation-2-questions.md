LLM used: Gemini 3.1 Pro

[SD2: Experiment Tracking Fundamentals]
Uploaded docs: MLflow paper, gamage exp tracking in qsd paper, link form mlflow exp tracking

[fact_single] - OKAY VALID
LLM : Claude Opus 4.6
Q: What are the three types of data that can be logged per experiment run using MLflow Tracking?
A: Parameters (key-value pairs), metrics (quantitative measures that can be updated during a run), and artifacts (arbitrary output files).
Source: Section 3.1 (MLflow Tracking), API code example.

[fact_single] - INVALID
LLM : Claude Opus 4.6
COMMENT: Seems too complex
Q: In a structured experiment tracking system, what three categories of data can be logged for each experiment run to support traceability and reproducibility?
A: Each experiment run can log parameters (arbitrary key-value pairs such as hyperparameters or configuration settings), metrics (quantitative measures of performance that can be updated throughout the run), and artifacts (arbitrary output files such as plots or serialized models). In a quantum software development context, these same categories would map to inputs like backend configuration and circuit depth (parameters), output measures like fidelity or execution counts (metrics), and generated files like transpiled circuits or histograms (artifacts).
Source: Section 3.1 (MLflow Tracking), code example showing log_param, log_metric, and log_artifact API calls.

[fact_single] - INVALID
COMMENT: too depend on QPROV
Q: What are the four key categories of data captured by the QProv quantum provenance model to form the core of experiment tracking?
A: The four key categories captured by the QProv data schema are quantum circuit, quantum computer, compilation, and execution.
Source: Section II (BACKGROUND AND MOTIVATION) in Gamage et al.

[summary]
Q: How does the systematic collection of run data and artifacts through experiment tracking support the broader quantum software development lifecycle?
A: The collection of run data and artifacts supports quantum software development by ensuring reproducibility, which is essential because hardware reliability varies between runs and simulators cannot fully replace real hardware experimentation. Structured tracking also supports progress tracking by providing a clear view of how performance goals are being met, which helps guide the development to subsequent stages. Furthermore, logging these artifacts provides traceability, supports informed decision-making, and enhances collaboration by facilitating result sharing and co-development between software and hardware teams.
Source: Section II (BACKGROUND AND MOTIVATION) and Figure 1 in Gamage et al.

[reasoning]
Q: Based on the progression of quantum software development, why does the scope of experiment tracking need to expand when transitioning a program from a quantum simulator to a real Quantum Processing Unit (QPU)?
A: In the early stages of development using quantum simulators, experiment tracking primarily focuses on the algorithm or program itself. However, the transition to real QPUs requires an expanded tracking scope because real quantum environments involve physical constraints and variable hardware reliability that simulators cannot fully replicate. Consequently, developers must begin collecting hardware-specific information, such as calibration data and qubit quality, alongside the software data to accurately evaluate performance, ensure reproducibility, and address the technical limitations of Noisy Intermediate-Scale Quantum (NISQ) hardware.

[unanswerable]Q: What specific mathematical formulas should be used to calculate and track the variance of T1 and T2 coherence times when logging hardware calibration artifacts during an experiment run?A: The provided documents do not contain sufficient information to answer this question.



[SD3: QProv Provenance Taxonomy]

[fact_single] - VALID
Q: Within the quantum circuit category of the QProv taxonomy, which specific attribute dictates the cumulative gate error that is reflected in the execution results?
A: The circuit size, which represents the total number of gates executed in the quantum circuit.
Source: Section 3.1


[SD4: MLflow Tracking API]

[reasoning] - VALID
Q: A quantum developer is testing the fidelity of a single quantum circuit across five different physical QPU backends. Based on MLflow's organization strategies, why should they structure this process using one parent run with five child runs, rather than just logging five independent runs within the experiment?
A: MLflow provides the ability to create child runs under a single parent run specifically to group related executions together, much like how one would group individual folds in a cross-validation experiment. By inferring this structure for quantum workflows, testing the exact same circuit across five distinct QPUs represents sub-tasks of one overarching logical evaluation. Using a parent-child relationship clearly binds these hardware-specific executions together under a single cohesive test, keeping the overall experiment tracking workspace cleanly organized rather than cluttered with isolated runs.

[reasoning] - VALID
Q: A quantum researcher is executing a hybrid quantum-classical loop where the number of measurement shots is fixed at 8192, but the circuit's variational rotation angles are continuously updated across 100 optimization steps. Based on the functional differences in the MLflow Tracking API, which specific logging functions should be used for the shot count versus the rotation angles, and why?
A: The researcher should use mlflow.log_param() for the fixed shot count and mlflow.log_metric() for the dynamically changing rotation angles. The provided documentation defines parameters as arbitrary key-value pairs that record the configuration of a run. Because the shot count remains constant at 8192, it acts as a baseline configuration parameter. Conversely, metrics are explicitly designed so that "each metric can also be updated throughout the run". Since the variational angles change continuously across the 100 optimization steps, they must be tracked as metrics to properly capture their evolution over time.



[SD5: Qiskit-Specific Experiment Tracking Using MLflow and QProv]

[summary] - VALID
Q: During the quantum experiment tracking lifecycle, for which specific QProv fields is it recommended to save and log physical artifact files to MLflow rather than using simple parameters or metrics?
A: Artifact files are utilized to capture complex structural data or large arrays that do not fit into standard parameters or metrics. Specifically, QPY artifact files (.qpy) are used to log the execution order (Q3) and gate mappings (C2) to preserve the exact ordered list of CircuitInstruction objects for the logical circuit, as well as the fully transpiled circuit containing native basis gates and physical qubit assignments. Additionally, NumPy artifact files (.npy) are saved to log large classical input vectors (E1) that are encoded into the circuit, whereas small inputs might just be logged as string parameters.

[reasoning] - VALID
Answer LLM used: Claude Sonnet 3.6
Q: When evaluating the feasibility of running a specific quantum circuit on a hardware backend, why is it critical to simultaneously analyze the tracked "Circuit Depth" (Q5) alongside the backend's "Decoherence Times" (QC2)?
A: "Circuit Depth" measures the longest sequential chain of gate operations on any single qubit, while "Decoherence Times" (T1 and T2) dictate the physical time limits before individual qubits lose their quantum state. Because each gate requires a finite physical duration (QC6), a deeper circuit takes longer to execute. If the total execution time exceeds the decoherence time of any qubit involved in that critical path, those qubits will decohere before the computation finishes, resulting in degraded and noisy outputs. Since decoherence times also vary across qubits and change between hardware calibrations (Weder et al., 2021), analyzing both fields together — per qubit, not just on average — is essential to determine whether the circuit can successfully execute within the hardware's coherence window.