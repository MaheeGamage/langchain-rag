# Master Thesis Proposal

### Mahee Hewa Gamage

### Topic: AI-Assisted Experiment Tracking for Quantum Software Development

### Supervisor: Vlad Stirbu

### Degree Program: Master's Degree Program in Artificial Intelligence

## Introduction

This proposal is a continuation of the main research I work on with my research assistant in
Quantum Information and Computation research group contributing to EM4QS project. In the first
iteration of my research, I worked on Experiment Tracking for Quantum Software Development
(QSD) which was influenced by ML/AI workflows and tools and how to utilize them in QSD
workflow. In brief, I integrated MLflow to existing Qubernets [2] framework. This enables
Qubernetes users to send their quantum application logs to existing mlflow instance which
enabled Experiment Tracking capabilities to their quantum software development workflow.

This proposal presents the next iteration of the research. Main motivation is to how to utilize AI
related tools to enhance the QSD workflow and I planned to enhance existing experiment tracking
capabilities with RAG application which will further reduce entry barrier to adopt experiment
tracking to QSD workflow. I planned to use the Design Science Research methodology to work on
this research and below you can find an illustration of framing the research into DSR processes.
The research questions and objectives of the 2nd iteration mentioned in the diagram are also
mentioned on later pages for better readability.


## Background

The quantum computing field is growing rapidly as hardware improves, but most usable devices
are still in the NISQ (Noisy Intermediate-Scale Quantum) stage. NISQ hardware’s output accuracy
is too low to perform any process to generate meaningful outcome. On top of this, quantum
computer output by nature is probabilistic. Which means we need to run the quantum algorithm
multiple times to measure accurate output distribution. This noisy and probabilistic nature makes
every run / experiment unique which is harder to reproduce. These characteristics create
challenges in the development process of quantum algorithms, making experiments harder to
reproduce, and collaboration less effective.

Experiment tracking (widely used in ML/AI domain) can help address these issues. By
systematically recording inputs, outputs, metadata, and intermediate artifacts,


developers/researchers can better understand their experiments, reproduce results, and
collaborate more effectively. However, adding experiment tracking to existing quantum
development workflows requires users to learn specific tools and logging practices which act as a
barrier for many developers/researchers who are not familiar with the tooling that used for
experiment tracking.

To solve this, I propose developing an AI-assisted system using Retrieval-Augmented Generation
(RAG) program that helps users integrate experiment tracking into their quantum code. This
system would provide guidance or generate the necessary code segments, so users do not need
deep knowledge of the underlying tools.

## Research Questions

**Main question**

How can an AI-assisted system utilizing Retrieval-Augmented Generation (RAG) effectively lower
the barrier to entry for quantum software developers to adopt structured experiment tracking?

**Sub questions**

1. How effectively can a RAG-based system provide context-aware answers for Quantum
    experiment tracking questions?
2. To what extent can an AI assistant accurately modify and inject tracking code into existing
    quantum programs without requiring the developer to learn the underlying tracking API
3. How reliably can a natural language interface query and present specific historical data
    from stored experiments to support researcher decision-making?

## Research Objectives

1. Develop retrieval augmented generation system that contain knowledge about experiment
    tracking in Quantum Software Development and interact with user in natural language
2. Design the system to provide context aware guidance for user's natural language questions
    about Quantum experiment tracking using MLflow tools.
3. Enable the system to automatically modify and inject the necessary experiment tracking
    code into existing quantum programs
4. Develop a retrieval mechanism that allows researchers to query historical experiment data
    using natural language.


## Use cases

### Use Case 1: Intelligent Advisory for Tracking Standards & Tooling (2nd Objective)

```
Figure: Use case 1
```
The user interacts with the system to understand concepts and standards rather than just asking
for code fixes. The system acts as an expert consultant on Quantum provenance (QProv [5]) and
MLflow usage.

**Actor:** Quantum Researcher / Software Engineer.

**Prerequisites:** The user is unfamiliar with specific tracking standards (like QProv) or how MLflow
terminology maps to quantum experiments.

**Trigger:** The user asks a conceptual question in the chat interface.

```
Example : "How to log a parameter using MLflow?"
```
**System Action:**

- The system uses RAG to search its knowledge base, which includes indexed
    documentation on MLflow, QProv specifications, and quantum software development best
    practices.
- It synthesizes an answer that explains the concept in natural language.

**Output:** The user receives a clear, context-aware explanation or summary of the standard,
helping them understand how and what to track before they attempt to write the code.


### Use case 2: Automated Injection of Experiment Tracking Code (3rd Objective)

```
Figure: Use case 2
```
The system automatically modifies the user's quantum program to include the necessary logging
logic using MLflow SDK.

**Actor:** Quantum Researcher / Developer.

**Prerequisites:** The user has a functional quantum program (e.g., in Qiskit or PennyLane) but it
lacks any logging or tracking logic.

**Trigger:** The user prompts the system: "Modify this code to track the circuit depth and the
measurement outcome distribution."

**System Action:**

- The system parses the existing quantum code to understand its structure.
- It retrieves the correct syntax for MLflow instrumentation.
- It automatically rewrites the code block, injecting the necessary mlflow.log_param() and
    mlflow.log_metric() functions without altering the core quantum logic.

**Output:** A new, executable code block is generated inside already existing quantum program that
performs the experiment and simultaneously logs the requested data to the tracking server.


### Use Case 3: Natural Language Querying of Past Experiments (4th Objective)

```
Figure: Use case 3
```
The user queries the MLflow for past experiments using natural language to find tracked data.

**Actor:** Quantum Researcher / Developer.

**Prerequisites:** Multiple experiments have been executed and logged in the MLflow backend. The
user needs to view the results but does not want to write complex MLflow queries.

**Trigger:** The user asks: "Show me the experiment that had highest number of qubits which logged
yesterday"

**System Action:**

- The system interprets the intent (comparison) and entities (shots, fidelity, date) from the
    natural language prompt.
- It translates this into a query for the MLflow backend.
- It retrieves the relevant historical data.


**Output:** The system returns an experiment that has the highest number of qubits from all the
experiments that ran yesterday.

## System Design

For this research I will utilize already existing tools and infrastructures. Some of these are
developed by QIC research group. This includes,

1. Qubernetes environment - Kubernetes platform that provide environment to run
    containerized quantum workload in CPU, GPU (for simulation) and QPU (quantum
    processing units)
2. MLflow - Opensource platform that enable experiment tracking. This can be utilized to save
    and query experiments
3. Jupyter Notebook - Interactive environment where users can write, execute code in real-
    time and visualize data. In this instance this Jupyter Notebook is connected to Qubernetes
    environment with different kernel implementations to execute the code.

The proposed program will be integrated with the already existing Jupyter Notebook with a chat-
style interface to support researchers to add experiment tracking capabilities to their
development workflow


```
Figure: Proposed system design
```
## Other

This thesis work will be conducted under the guidance of Quantum Information and Computation
Research Group, which is part of national efforts supported by the Finnish Quantum Flagship. The
thesis topic will also align with the goals of the EM4QS project, supported by my ongoing research
assistant work on experiment tracking for quantum application development.

Reference

1. Quantum Computing in the NISQ era - https://arxiv.org/abs/1801.
2. Qubernetes - https://arxiv.org/abs/2408.
3. Iterative software development lifecycle approach - https://arxiv.org/abs/2507.
4. Experiment tracking in Quantum application development -
    https://arxiv.org/abs/2507.
5. QProv - https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/qtc2.


