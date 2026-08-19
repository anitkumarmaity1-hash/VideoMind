from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ProcessingJob(BaseModel):
    job_id: str
    video_id: str
    stage: str
    progress: int = 0
    status: str = "queued"  # queued | running | done | failed
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
