import os
import litellm
import openai
import mlflow
from mlflow.genai.scorers import Correctness, Guidelines

# Tell litellm where your Ollama is
os.environ["OLLAMA_API_BASE"] = "http://localhost:11435"

# 1. Define a simple QA dataset
dataset = [
    {
        "inputs": {"question": "Can MLflow manage prompts?"},
        "expectations": {"expected_response": "Yes!"},
    },
    {
        "inputs": {"question": "Can MLflow create a taco for my lunch?"},
        "expectations": {
            "expected_response": "No, unfortunately, MLflow is not a taco maker."
        },
    },
]


# 2. Define a prediction function to generate responses
def predict_fn(question: str) -> str:
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini", messages=[{"role": "user", "content": question}]
    # )
    response = litellm.completion(
        model="ollama/phi3.5",
        messages=[{"role": "user", "content": question}],
        api_base="http://localhost:11435",
    )
    return response.choices[0].message.content


# 3.Run the evaluation
results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=predict_fn,
    scorers=[
        # Built-in LLM judge
        Correctness(model="ollama:/phi3.5"),
        # Custom criteria using LLM judge
        Guidelines(name="is_english", guidelines="The answer must be in English"),
    ],
)
