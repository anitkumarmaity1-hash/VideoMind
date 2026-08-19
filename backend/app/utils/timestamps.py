"""
Timestamp conversion utilities.
Seconds (float) <-> "MM:SS" / "HH:MM:SS" display format.
"""


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS, or HH:MM:SS if >= 1 hour."""
    if seconds is None or seconds < 0:
        seconds = 0
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_timestamp(ts: str) -> float:
    """Convert 'MM:SS' or 'HH:MM:SS' string to seconds (float)."""
    parts = [int(p) for p in ts.strip().split(":")]
    if len(parts) == 2:
        minutes, secs = parts
        return float(minutes * 60 + secs)
    if len(parts) == 3:
        hours, minutes, secs = parts
        return float(hours * 3600 + minutes * 60 + secs)
    raise ValueError(f"Invalid timestamp format: {ts}")
