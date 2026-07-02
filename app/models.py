from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    recommendations: Optional[List[dict]] = None

class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool = False


class CatalogRecord(BaseModel):
    entity_id: str
    name: str
    link: str
    scraped_at: str
    job_levels: list[str] = Field(default_factory=list)
    job_levels_raw: str = ""
    languages: list[str] = Field(default_factory=list)
    languages_raw: str = ""
    duration: str = ""
    duration_raw: str = ""
    status: str = "ok"
    remote: str = "yes"
    adaptive: str = "no"
    description: str = ""
    keys: list[str] = Field(default_factory=list)
    is_job_solution: Optional[bool] = None
    is_report_only: Optional[bool] = None
    test_type: Optional[str] = None


KEYS_TO_CODE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}
