"""
Frame sampling using OpenCV.

extract_frames_at_timestamps() is the one actually used by the pipeline: it
seeks directly to a small list of target timestamps (one per transcript
chunk) instead of decoding the entire video and discarding most frames.

extract_frames() (dense uniform sampling) is kept for callers that
genuinely want every Nth frame, but the pipeline no longer uses it — see
the note in pipeline_runner.py for why it was a major bottleneck on long
videos.
"""
import os
import cv2
from typing import List, Tuple, Optional
from app.config import settings


def extract_frames_at_timestamps(
    video_path: str, timestamps: List[float], output_dir: str
) -> List[Optional[Tuple[float, str]]]:
    """
    Grabs exactly one frame per requested timestamp via direct seek
    (CAP_PROP_POS_MSEC) instead of decoding every frame in between.

    Returns a list the SAME LENGTH and in the SAME ORDER as `timestamps`,
    with `None` in place of any timestamp whose seek/read failed — callers
    zip this positionally against their own timestamp list, so a shorter
    list here would silently misalign every entry after the first failure.

    For a 1-hour video with 10s chunks (~450 chunks), this does ~450 seeks
    instead of decoding ~90,000+ frames at native fps and discarding all
    but ~7,200 of them (the old dense-sample-then-average approach) — the
    single biggest contributor to long "extracting_frames" times on long
    videos. Seek accuracy can land on the nearest keyframe rather than the
    exact frame, which is irrelevant here since frames are only used as a
    coarse visual representative for a 10-second window, not for
    frame-exact retrieval.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    results: List[Optional[Tuple[float, str]]] = []
    try:
        for idx, ts in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(ts, 0) * 1000)
            ret, frame = cap.read()
            if not ret:
                results.append(None)
                continue
            frame_path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            results.append((round(ts, 2), frame_path))
    finally:
        cap.release()

    return results


def extract_frames(video_path: str, output_dir: str, sample_fps: Optional[float] = None) -> List[Tuple[float, str]]:
    """Dense uniform sampling at `sample_fps` (default 2 FPS). Decodes every
    frame in the video, so only use this when you actually need many
    frames per unit time — not for the coarse one-frame-per-chunk case."""
    sample_fps = sample_fps or settings.frame_sample_fps
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(int(round(video_fps / sample_fps)), 1)

    results = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            frame_path = os.path.join(output_dir, f"frame_{saved_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            results.append((round(timestamp, 2), frame_path))
            saved_idx += 1

        frame_idx += 1

    cap.release()
    return results
