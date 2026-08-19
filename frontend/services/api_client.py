"""
Thin HTTP client wrapping the FastAPI backend for the Streamlit frontend.
"""
import os
import requests

API_BASE_URL = os.environ.get("VIDEOMIND_API_URL", "http://localhost:8000")


def upload_video(file_bytes: bytes, filename: str):
    files = {"file": (filename, file_bytes)}
    resp = requests.post(f"{API_BASE_URL}/api/videos/upload", files=files)
    resp.raise_for_status()
    return resp.json()


def upload_from_url(url: str):
    resp = requests.post(f"{API_BASE_URL}/api/videos/url", data={"url": url})
    resp.raise_for_status()
    return resp.json()


def get_video(video_id: str):
    resp = requests.get(f"{API_BASE_URL}/api/videos/{video_id}")
    resp.raise_for_status()
    return resp.json()


def get_status(video_id: str):
    resp = requests.get(f"{API_BASE_URL}/api/videos/{video_id}/status")
    resp.raise_for_status()
    return resp.json()


def get_segments(video_id: str):
    resp = requests.get(f"{API_BASE_URL}/api/videos/{video_id}/segments")
    resp.raise_for_status()
    return resp.json()


def ask_question(video_id: str, question: str, answer_mode: str = "standard"):
    resp = requests.post(
        f"{API_BASE_URL}/api/videos/{video_id}/questions",
        json={"question": question, "answer_mode": answer_mode},
    )
    resp.raise_for_status()
    return resp.json()


def get_summary(video_id: str, summary_type: str = "short"):
    resp = requests.post(
        f"{API_BASE_URL}/api/videos/{video_id}/summary",
        json={"summary_type": summary_type},
    )
    resp.raise_for_status()
    return resp.json()


def delete_video(video_id: str):
    resp = requests.delete(f"{API_BASE_URL}/api/videos/{video_id}")
    resp.raise_for_status()
    return resp.json()


def health_check():
    resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
    resp.raise_for_status()
    return resp.json()
