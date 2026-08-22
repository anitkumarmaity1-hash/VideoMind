"""
Isolated YouTube adapter. Validates URLs, extracts video IDs, and
downloads only permitted (non-restricted, non-age-gated, non-DRM) videos
via yt-dlp. This module never attempts to bypass platform restrictions —
if yt-dlp reports a video as unavailable/restricted, we surface that as
an error rather than working around it.
"""
import re
from typing import Any, cast
from app.config import settings

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


def _base_ydl_opts() -> dict:
    """Shared yt-dlp options for both metadata fetch and download.

    Explicitly targets player clients that currently don't require a PO
    Token (see settings.youtube_player_clients) instead of relying on
    yt-dlp's own default ("web", generally), which now needs both a PO
    Token we can't generate and a JS runtime (for nsig deciphering) that
    isn't installed here — the combination silently drops every real
    video format and leaves only thumbnail/image "formats", which is why
    downloads were failing with "Requested format is not available"
    even after the "Precondition check failed" warnings looked resolved.
    """
    return {
        "extractor_args": {
            "youtube": {
                "player_client": [
                    c.strip() for c in settings.youtube_player_clients.split(",") if c.strip()
                ]
            }
        },
    }


def get_permitted_metadata(url: str) -> dict:
    """Fetch title/duration without downloading, respecting platform restrictions."""
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        **_base_ydl_opts(),
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

    # Ordered fallback, cheapest/most-compatible first, ending in format 18
    # (360p mp4) — the one format that has stayed exempt from the GVS PO
    # Token requirement across every client, so it's a guaranteed-available
    # last resort rather than another chance to hit the same wall.
    format_selector = (
        "bestvideo[protocol*=m3u8]+bestaudio[protocol*=m3u8]"
        "/best[protocol*=m3u8]"
        "/best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]"
        "/18"
        "/best"
    )

    clients = [
        c.strip() for c in settings.youtube_player_clients.split(",") if c.strip()
    ]

    # IMPORTANT: try clients one at a time, not all together.
    # Passing multiple player_client values to a single yt-dlp call (the old
    # _base_ydl_opts() approach) makes yt-dlp pool every client's formats
    # into one list and pick a single best match for the format string.
    # If that pooled "best" match happens to come from a client whose
    # formats now need a PO Token we don't have (android_vr did, as of the
    # Aug 2026 regression — see the comment on youtube_player_clients), the
    # whole download fails even though an earlier, still-working client in
    # the list had perfectly good formats. This is exactly what happened
    # here: the log shows only android_vr's PO Token warning even though
    # web_embedded and tv were also configured. Trying each client in its
    # own isolated call and falling through to the next on failure avoids
    # that — a working client earlier in the list gets used instead of
    # being silently outvoted by a broken one later in the list.
    last_error: Exception | None = None
    for client in clients:
        ydl_opts = {
            "format": format_selector,
            "merge_output_format": "mp4",
            "outtmpl": output_path,
            "noplaylist": True,
            "quiet": True,
            "extractor_args": {"youtube": {"player_client": [client]}},
        }
        try:
            with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
                ydl.download([url])
            return output_path
        except Exception as e:
            last_error = e
            continue

    err_text = str(last_error)
    # yt-dlp's own message here is accurate but easy to miss in a wall of
    # warnings — surface it plainly, and only blame the JS runtime for the
    # "n challenge" part, since that's the part it actually fixes. A PO
    # Token 403 that survives every client in youtube_player_clients means
    # YouTube changed which clients are exempt again; check
    # https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide and update that
    # setting (or add "ios" / "tv_simply" to it) rather than editing code.
    if "403" in err_text or "PO Token" in err_text:
        raise YouTubeError(
            "Download failed on every configured player client "
            f"({', '.join(clients)}) — HTTP 403 / PO Token issue. This can "
            "mean (a) no JS runtime is installed, which breaks the 'n "
            "challenge' step yt-dlp needs for some formats (install Deno: "
            "`winget install DenoLand.Deno` on Windows, restart your "
            "terminal), or (b) YouTube has changed which clients are exempt "
            "from PO Tokens again — try adding 'ios' or 'tv_simply' to "
            "youtube_player_clients, or upgrade yt-dlp "
            "(`pip install -U --pre yt-dlp` for the nightly build, which "
            "often has extractor fixes before they reach a stable release). "
            f"Original error: {err_text}"
        )
    raise YouTubeError(f"Download failed: {err_text}")
