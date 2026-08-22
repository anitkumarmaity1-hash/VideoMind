# frontend/services/api_client.py — full replacement
"""
Thin HTTP client wrapping the FastAPI backend for the Streamlit frontend.
All calls have explicit timeouts so a stuck backend surfaces as a clear
error in the UI instead of an infinite spinner.
"""
import os
import requests

API_BASE_URL = os.environ.get("VIDEOMIND_API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 15
LLM_TIMEOUT = 90  # question endpoint does retrieval + one Groq call
SUMMARY_TIMEOUT = 300  # summary endpoint paces several Groq calls under a low TPM cap


def _auth_headers(token: str | None) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def register(username: str, password: str):
    resp = requests.post(
        f"{API_BASE_URL}/api/auth/register",
        json={"username": username, "password": password},
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def login(username: str, password: str):
    resp = requests.post(
        f"{API_BASE_URL}/api/auth/login",
        json={"username": username, "password": password},
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def get_history(video_id: str, token: str, limit: int = 50):
    resp = requests.get(
        f"{API_BASE_URL}/api/videos/{video_id}/history",
        params={"limit": limit},
        headers=_auth_headers(token),
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def upload_video(file_bytes: bytes, filename: str):
    files = {"file": (filename, file_bytes)}
    resp = requests.post(
        f"{API_BASE_URL}/api/videos/upload", files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()


def upload_from_url(url: str):
    resp = requests.post(f"{API_BASE_URL}/api/videos/url",
                         data={"url": url}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def get_video(video_id: str):
    resp = requests.get(
        f"{API_BASE_URL}/api/videos/{video_id}", timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_status(video_id: str):
    resp = requests.get(
        f"{API_BASE_URL}/api/videos/{video_id}/status", timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_segments(video_id: str):
    resp = requests.get(
        f"{API_BASE_URL}/api/videos/{video_id}/segments", timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def ask_question(video_id: str, question: str, answer_mode: str = "standard", token: str | None = None):
    resp = requests.post(
        f"{API_BASE_URL}/api/videos/{video_id}/questions",
        json={"question": question, "answer_mode": answer_mode},
        headers=_auth_headers(token),
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def get_summary(video_id: str, summary_type: str = "short", token: str | None = None):
    resp = requests.post(
        f"{API_BASE_URL}/api/videos/{video_id}/summary",
        json={"summary_type": summary_type},
        headers=_auth_headers(token),
        # Summaries now make far fewer, but individually rate-limit-paced,
        # Groq calls (see summarizer.py) — a long video's summary can
        # legitimately take a few minutes on a low-TPM Groq tier rather
        # than erroring out, so this gets a longer timeout than other
        # LLM-backed calls instead of failing fast.
        timeout=SUMMARY_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def delete_video(video_id: str):
    resp = requests.delete(
        f"{API_BASE_URL}/api/videos/{video_id}", timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def health_check():
    resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
    resp.raise_for_status()
    return resp.json()
