You are building an evaluation dataset for a RAG system that assists quantum software 
developers with experiment tracking using MLflow and QProv standards.

The system's users are quantum researchers and software engineers who may be unfamiliar 
with experiment tracking tooling. They interact via a Jupyter Notebook chat interface.

The questions must sound like something a real quantum developer would type into a 
chat interface — not a textbook exercise.

Generate 10 FACT_SINGLE questions according to below description

[FACT_SINGLE]
Rule: The answer is a single, discrete piece of information directly stated in the context. 
It cannot be partially correct — either the full fact is retrieved or not.
Good question types: "What is the default value of...", "Which function is used to...", 
"What does X return when..."
Avoid: multi-part questions, questions requiring comparison.

Question: <question>
Answer: <concise factual answer, directly from context>
Source sentence: <quote the exact sentence from context that contains the answer>