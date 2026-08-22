"""
Isolated YouTube adapter. Validates URLs, extracts video IDs, and
downloads only permitted (non-restricted, non-age-gated, non-DRM) videos
via yt-dlp. This module never attempts to bypass platform restrictions —
if yt-dlp reports a video as unavailable/restricted, we surface that as
an error rather than working around it.
"""
import re
from typing import Any, cast

YOUTUBE_URL_PATTERN = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})"
)


class YouTubeError(Exception):
    pass


def extract_video_id(url: str) -> str:
    match = YOUTUBE_URL_PATTERN.search(url)
    if not match:
        raise YouTubeError(
            f"Could not extract a valid YouTube video ID from URL: {url}")
    return match.group(1)


def validate_url(url: str) -> bool:
    return bool(YOUTUBE_URL_PATTERN.search(url))


def get_permitted_metadata(url: str) -> dict:
    """Fetch title/duration without downloading, respecting platform restrictions."""
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        # The "web" client alone increasingly gets served a signature/PO-token
        # challenge that looks like "video unavailable" from yt-dlp's side.
        # Falling through android -> web -> tv mirrors yt-dlp's own current
        # recommendation and avoids most of those false negatives.
        "extractor_args": {"youtube": {"player_client": ["android", "web", "tv"]}},
    }
    try:
        # yt-dlp's type stubs declare YoutubeDL(params: _Params | None), a
        # TypedDict that Pylance won't structurally match against a plain
        # dict[str, ...] literal even though this is exactly how yt-dlp's
        # own docs show it being called. cast(Any, ...) here, not a
        # behavior change — same dict, just silences a stub mismatch.
        with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        raise YouTubeError(f"Video unavailable or restricted: {e}")

    if info.get("age_limit", 0) > 0 or info.get("is_live"):
        raise YouTubeError(
            "This video is restricted (age-gated or live) and cannot be processed.")

    return {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "duration": info.get("duration"),
    }


def download_video(url: str, output_path: str) -> str:
    import yt_dlp

    # raises YouTubeError if restricted/unavailable
    get_permitted_metadata(url)
    ydl_opts = {
        # Previous selector "mp4/bestvideo+bestaudio" tried a literal format
        # id called "mp4" (which essentially never exists) and, on falling
        # through to bestvideo+bestaudio, merged into whatever container
        # yt-dlp defaults to (often webm) despite output_path ending in
        # .mp4. This selector prefers a pre-merged mp4 (no ffmpeg mux
        # needed, faster) and only falls back to muxing separate mp4
        # video/m4a audio streams, always landing on an actual mp4.
        "format": "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "merge_output_format": "mp4",
        "outtmpl": output_path,
        "noplaylist": True,
        "quiet": True,
        "extractor_args": {"youtube": {"player_client": ["android", "web", "tv"]}},
    }
    try:
        with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
            ydl.download([url])
    except Exception as e:
        raise YouTubeError(f"Download failed: {e}")
    return output_path
