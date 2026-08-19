"""
File validation, size limits, and path-traversal protection.
"""
import os
import uuid
from app.config import settings


class ValidationError(Exception):
    pass


def validate_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in settings.allowed_extensions_list:
        raise ValidationError(
            f"Unsupported file extension '{ext}'. Allowed: {settings.allowed_extensions_list}"
        )
    return ext


def validate_size(size_bytes: int) -> None:
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise ValidationError(
            f"File too large ({size_bytes / 1e6:.1f} MB). Max allowed: {settings.max_upload_size_mb} MB"
        )


def generate_video_id() -> str:
    return f"vid_{uuid.uuid4().hex[:12]}"


def safe_join(base_dir: str, *paths: str) -> str:
    """
    Join paths safely, preventing path traversal outside base_dir.
    Raises ValidationError if the resolved path escapes base_dir.
    """
    base_dir = os.path.abspath(base_dir)
    target = os.path.abspath(os.path.join(base_dir, *paths))
    if not target.startswith(base_dir + os.sep) and target != base_dir:
        raise ValidationError("Path traversal detected — resolved path escapes base directory.")
    return target
