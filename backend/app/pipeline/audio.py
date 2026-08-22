"""
Audio extraction from video using ffmpeg (via subprocess for reliability
across ffmpeg-python versions).
"""
import subprocess
import os
from typing import List, NamedTuple


class AudioPart(NamedTuple):
    """One slice of a larger audio file, ready for independent transcription.

    `offset` is where this part's audio actually starts in the *original*
    file (including any lead-in overlap) — add it to a segment's local
    start/end times to get back to global time.

    `core_start`/`core_end` mark the non-overlapping range this part is
    responsible for. Segments whose (global) start time falls inside
    [core_start, core_end) belong to this part; segments starting in the
    overlap padding belong to a neighboring part instead and should be
    dropped here to avoid double-counting the same speech twice.
    """
    path: str
    offset: float
    core_start: float
    core_end: float


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


def split_audio(
    audio_path: str,
    duration: float,
    num_parts: int,
    overlap_seconds: float,
    out_dir: str,
) -> List[AudioPart]:
    """
    Split a WAV file into `num_parts` roughly-equal chunks for parallel
    transcription, each padded with `overlap_seconds` of extra audio on
    either side (clamped to the file's actual bounds) so faster-whisper
    has context right up to the cut point instead of hard-truncating
    mid-word. Uses ffmpeg's own seek/duration flags on the already-decoded
    PCM WAV, which is a cheap stream copy (no re-encode).
    """
    if num_parts < 2:
        return [AudioPart(path=audio_path, offset=0.0, core_start=0.0, core_end=duration)]

    os.makedirs(out_dir, exist_ok=True)
    core_len = duration / num_parts
    parts: List[AudioPart] = []
    for i in range(num_parts):
        core_start = i * core_len
        core_end = duration if i == num_parts - 1 else (i + 1) * core_len
        extract_start = max(0.0, core_start - overlap_seconds)
        extract_end = min(duration, core_end + overlap_seconds)

        part_path = os.path.join(out_dir, f"part_{i:03d}.wav")
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-ss", f"{extract_start:.3f}",
            "-to", f"{extract_end:.3f}",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            part_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio split failed for part {i}: {result.stderr}")

        parts.append(AudioPart(
            path=part_path,
            offset=extract_start,
            core_start=core_start,
            core_end=core_end,
        ))
    return parts


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
