"""
Frame sampling using OpenCV. Extracts frames at a configurable FPS
(default 2 FPS) rather than every frame, and streams frame-by-frame to
avoid loading the whole video into memory.
"""
import os
import cv2
from typing import List, Tuple, Optional
from app.config import settings


def extract_frames(video_path: str, output_dir: str, sample_fps: Optional[float] = None) -> List[Tuple[float, str]]:
    """
    Returns a list of (timestamp_seconds, frame_file_path) tuples.
    """
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
