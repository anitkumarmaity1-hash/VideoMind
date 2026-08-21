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
LLM_TIMEOUT = 90  # question/summary endpoints do retrieval + Groq calls


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


def ask_question(video_id: str, question: str, answer_mode: str = "standard"):
    resp = requests.post(
        f"{API_BASE_URL}/api/videos/{video_id}/questions",
        json={"question": question, "answer_mode": answer_mode},
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def get_summary(video_id: str, summary_type: str = "short"):
    resp = requests.post(
        f"{API_BASE_URL}/api/videos/{video_id}/summary",
        json={"summary_type": summary_type},
        timeout=LLM_TIMEOUT,
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
