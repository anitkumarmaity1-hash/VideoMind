from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


class QuestionRequest(BaseModel):
    question: str
    answer_mode: Literal["standard", "simple", "detailed", "technical"] = "standard"


class EvidenceItem(BaseModel):
    start_time: float
    end_time: float
    start_formatted: str
    end_formatted: str
    modality: Literal["text", "visual"]
    content: str
    score: float


class AnswerResponse(BaseModel):
    answer_id: str
    question_id: str
    video_id: str
    question: str
    answer: str
    evidence: List[EvidenceItem]
    question_type: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SummaryRequest(BaseModel):
    summary_type: Literal["short", "detailed"] = "short"


class SummaryResponse(BaseModel):
    video_id: str
    summary_type: str
    summary: str
    bullet_points: Optional[List[str]] = None
    sections: Optional[List[dict]] = None
