from pydantic import BaseModel
from typing import List


class VideoSegment(BaseModel):
    segment_id: str
    video_id: str
    chunk_id: int
    transcript: str
    start_time: float
    end_time: float
    frame_timestamps: List[float] = []


class SegmentResponse(BaseModel):
    chunk_id: int
    start_time: float
    end_time: float
    transcript: str
    start_formatted: str
    end_formatted: str
