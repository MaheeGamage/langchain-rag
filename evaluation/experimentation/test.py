import os
import litellm
import mlflow
from mlflow.genai.scorers import Correctness

# Tell litellm where your Ollama is
os.environ["OLLAMA_API_BASE"] = "http://localhost:11435"

# Optional: verify connectivity first
response = litellm.completion(
    model="ollama/phi3.5",
    messages=[{"role": "user", "content": "ping"}],
    api_base="http://localhost:11435",
)
print(response.choices[0].message.content)

# # Now use it as a judge in MLflow
# results = mlflow.genai.evaluate(
#     data=eval_dataset,
#     predict_fn=predict_fn,
#     scorers=[
#         Correctness(model="ollama/phi3.5"),
#     ],
# )