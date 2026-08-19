"""
Temporal chunk creation.

Splits a transcript (list of {start, end, text} segments from Whisper) into
fixed-size overlapping time windows, and attaches the transcript text and
sampled frame timestamps that fall inside each window.
"""
from typing import List, Dict, Any
from app.config import settings


def create_temporal_chunks(
    transcript_segments: List[Dict[str, Any]],
    video_duration: float,
    chunk_size: int = None,
    overlap: int = None,
) -> List[Dict[str, Any]]:
    """
    transcript_segments: [{"start": float, "end": float, "text": str}, ...]
    Returns a list of chunk dicts:
        {chunk_id, start_time, end_time, transcript}
    """
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size_seconds
    overlap = overlap if overlap is not None else settings.chunk_overlap_seconds

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    step = chunk_size - overlap
    chunks = []
    chunk_id = 0
    start = 0.0

    while start < video_duration:
        end = min(start + chunk_size, video_duration)

        # Gather transcript text whose midpoint falls within [start, end)
        texts = []
        for seg in transcript_segments:
            seg_mid = (seg["start"] + seg["end"]) / 2
            if start <= seg_mid < end:
                texts.append(seg["text"].strip())

        chunks.append(
            {
                "chunk_id": chunk_id,
                "start_time": round(start, 2),
                "end_time": round(end, 2),
                "transcript": " ".join(texts).strip(),
            }
        )
        chunk_id += 1
        start += step

        if end >= video_duration:
            break

    return chunks


def attach_frame_timestamps(chunks: List[Dict[str, Any]], frame_timestamps: List[float]) -> List[Dict[str, Any]]:
    """Attach the subset of frame_timestamps that fall inside each chunk window."""
    for chunk in chunks:
        chunk["frame_timestamps"] = [
            ts for ts in frame_timestamps if chunk["start_time"] <= ts < chunk["end_time"]
        ]
    return chunks
