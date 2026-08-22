from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    UPLOADED = "uploaded"
    DOWNLOADING = "downloading"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    EXTRACTING_FRAMES = "extracting_frames"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class VideoMetadata(BaseModel):
    video_id: str
    filename: str
    duration: Optional[float] = None
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    processing_status: ProcessingStatus = ProcessingStatus.UPLOADED
    storage_path: str
    source: str = "upload"  # "upload" or "youtube"
    source_url: Optional[str] = None
    error_message: Optional[str] = None

    model_config = {"use_enum_values": True}


class VideoResponse(BaseModel):
    video_id: str
    filename: str
    duration: Optional[float]
    upload_time: datetime
    processing_status: str
    source: str


class VideoStatusResponse(BaseModel):
    video_id: str
    processing_status: str
    progress: int = 0
    stage: Optional[str] = None
    error_message: Optional[str] = None
