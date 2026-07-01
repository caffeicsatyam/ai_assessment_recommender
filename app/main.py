from __future__ import annotations

try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from fastapi import FastAPI
from app.models import ChatRequest, ChatResponse
from app.recommender import generate_response

app = FastAPI(title="SHL Assessment Recommender")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    messages = [m.model_dump() for m in request.messages]

    reply, recommendations, end = generate_response(messages)

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end,
    )