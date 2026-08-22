"""
Speech-to-text transcription using faster-whisper.

Note: model loading is expensive (esp. on CPU). This module lazily loads
and caches a single WhisperModel instance per process. In the Colab/GPU
worker, WHISPER_DEVICE=cuda makes this fast; in local CPU-only dev it will
be slow, which is expected for a portfolio dev environment.
"""
from typing import List, Dict, Any, Callable, Optional
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


def transcribe_audio(
    audio_path: str,
    duration: Optional[float] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns a list of {"start": float, "end": float, "text": str} segments,
    preserving Whisper's native timestamped segmentation.

    If `duration` (total audio length in seconds) and `progress_callback`
    are given, the callback is invoked with a 0-100 float roughly every
    ~5% of the audio as it's transcribed — faster-whisper's `segments` is
    a generator that only does work as it's iterated, so without this the
    caller (and the /status endpoint the frontend polls) has no visibility
    into a stage that can otherwise sit "transcribing" for many minutes
    with no feedback and look hung.
    """
    model = _get_model()
    segments, _info = model.transcribe(
        audio_path,
        beam_size=settings.whisper_beam_size,
        vad_filter=True,
        # condition_on_previous_text=True (the default) feeds each segment's
        # transcript back in as context for the next, which on long/quiet
        # or repetitive audio can make Whisper loop and re-emit the same
        # phrase in consecutive segments. Turning it off trades a small
        # amount of cross-segment context for much lower repetition risk.
        condition_on_previous_text=False,
        # Bail out of a segment early if it looks like a repetition loop.
        compression_ratio_threshold=2.4,
    )

    result = []
    last_reported_pct = -1.0
    for seg in segments:
        result.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
        if progress_callback and duration:
            pct = min(99.0, (seg.end / duration) * 100)
            if pct - last_reported_pct >= 5:
                progress_callback(pct)
                last_reported_pct = pct

    if progress_callback:
        progress_callback(100.0)

    return _dedupe_repeated_segments(result)


def _dedupe_repeated_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop a segment if its text is a near-exact repeat of the one right
    before it. Whisper occasionally re-emits the same sentence across two
    consecutive segments; this is a cheap safety net on top of the
    transcription-time guards above, not a substitute for them."""
    deduped: List[Dict[str, Any]] = []
    for seg in segments:
        if deduped:
            prev_text = deduped[-1]["text"].strip().lower()
            cur_text = seg["text"].strip().lower()
            if prev_text and (cur_text == prev_text or cur_text in prev_text or prev_text in cur_text):
                continue
        deduped.append(seg)
    return deduped
