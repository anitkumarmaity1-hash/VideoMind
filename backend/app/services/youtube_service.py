"""
Isolated YouTube adapter. Validates URLs, extracts video IDs, and
downloads only permitted (non-restricted, non-age-gated, non-DRM) videos
via yt-dlp. This module never attempts to bypass platform restrictions —
if yt-dlp reports a video as unavailable/restricted, we surface that as
an error rather than working around it.
"""
import re

YOUTUBE_URL_PATTERN = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})"
)


class YouTubeError(Exception):
    pass


def extract_video_id(url: str) -> str:
    match = YOUTUBE_URL_PATTERN.search(url)
    if not match:
        raise YouTubeError(f"Could not extract a valid YouTube video ID from URL: {url}")
    return match.group(1)


def validate_url(url: str) -> bool:
    return bool(YOUTUBE_URL_PATTERN.search(url))


def get_permitted_metadata(url: str) -> dict:
    """Fetch title/duration without downloading, respecting platform restrictions."""
    import yt_dlp

    ydl_opts = {"quiet": True, "skip_download": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        raise YouTubeError(f"Video unavailable or restricted: {e}")

    if info.get("age_limit", 0) > 0 or info.get("is_live"):
        raise YouTubeError("This video is restricted (age-gated or live) and cannot be processed.")

    return {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "duration": info.get("duration"),
    }


def download_video(url: str, output_path: str) -> str:
    import yt_dlp

    get_permitted_metadata(url)  # raises YouTubeError if restricted/unavailable
    ydl_opts = {
        "format": "mp4/bestvideo+bestaudio",
        "outtmpl": output_path,
        "noplaylist": True,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path
