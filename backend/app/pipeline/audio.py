"""
Audio extraction from video using ffmpeg (via subprocess for reliability
across ffmpeg-python versions).
"""
import subprocess
import os


def extract_audio(video_path: str, output_audio_path: str, sample_rate: int = 16000) -> str:
    """
    Extract mono PCM WAV audio at `sample_rate` Hz from a video file.
    Uses ffmpeg CLI directly (must be installed on the system / container).
    """
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i", video_path,
        "-vn",  # no video
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")
    return output_audio_path


def get_video_duration(video_path: str) -> float:
    """Use ffprobe to get video duration in seconds."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())
