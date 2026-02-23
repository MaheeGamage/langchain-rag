# app/api.py

from fastapi import FastAPI
from .graph import graph

app = FastAPI()


@app.post("/query")
async def query(question: str):
    result = graph.invoke({"question": question})
    return {"answer": result["answer"]}
