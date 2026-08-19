"""
Speech-to-text transcription using faster-whisper.

Note: model loading is expensive (esp. on CPU). This module lazily loads
and caches a single WhisperModel instance per process. In the Colab/GPU
worker, WHISPER_DEVICE=cuda makes this fast; in local CPU-only dev it will
be slow, which is expected for a portfolio dev environment.
"""
from typing import List, Dict, Any
from app.config import settings

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    return _model


def transcribe_audio(audio_path: str) -> List[Dict[str, Any]]:
    """
    Returns a list of {"start": float, "end": float, "text": str} segments,
    preserving Whisper's native timestamped segmentation.
    """
    model = _get_model()
    segments, _info = model.transcribe(audio_path, beam_size=5, vad_filter=True)

    result = []
    for seg in segments:
        result.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
    return result
