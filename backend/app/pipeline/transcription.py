# backend/app/pipeline/transcription.py — full replacement
"""
Speech-to-text transcription. Two backends:
- "local": faster-whisper, CPU/GPU. Fine for dev, too slow for a CPU-only
  prod host like Render's web dyno.
- "groq": Groq's hosted Whisper API (OpenAI-compatible). No local model,
  no GPU needed, transcribes ~1hr audio in seconds. Use this in prod.
"""
from typing import List, Dict, Any
from app.config import settings

_model = None


def _get_local_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    return _model


def _transcribe_local(audio_path: str) -> List[Dict[str, Any]]:
    model = _get_local_model()
    segments, _info = model.transcribe(
        audio_path, beam_size=5, vad_filter=True)
    return [
        {"start": round(s.start, 2), "end": round(
            s.end, 2), "text": s.text.strip()}
        for s in segments
    ]


def _transcribe_groq(audio_path: str) -> List[Dict[str, Any]]:
    from groq import Groq
    client = Groq(api_key=settings.groq_api_key, timeout=120.0, max_retries=2)
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            file=(audio_path, f.read()),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )
    # The installed groq SDK's type stub for the transcription response
    # doesn't declare `.segments` (Pylance flags it), but the API does
    # return it for response_format="verbose_json". Read it via the raw
    # dict form so this doesn't depend on the stub being accurate.
    raw = resp.model_dump() if hasattr(
        resp, "model_dump") else dict(resp)  # type: ignore[arg-type]
    segments = raw.get("segments") or []
    return [
        {"start": round(seg["start"], 2), "end": round(
            seg["end"], 2), "text": seg["text"].strip()}
        for seg in segments
    ]


def transcribe_audio(audio_path: str) -> List[Dict[str, Any]]:
    """Returns [{"start": float, "end": float, "text": str}, ...]."""
    if settings.whisper_backend == "groq":
        return _transcribe_groq(audio_path)
    return _transcribe_local(audio_path)
