from fastapi import FastAPI
from pydantic import BaseModel

from .graph import build_graph


app = FastAPI(title="Gold RAG")
graph = None


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    global graph
    if graph is None:
        graph = build_graph()
    state = graph.invoke({"question": request.question})
    return {"answer": state.get("answer"), "confidence": state.get("confidence")}

