"""
Speech-to-text transcription using faster-whisper.

Note: model loading is expensive (esp. on CPU). This module lazily loads
and caches WhisperModel instance(s) per process.

Performance note (why this file is more than a thin wrapper):
A single `WhisperModel.transcribe()` call is CPU-bound on ctranslate2's
internal thread pool, which by default does *not* spread across every
core on the machine — on a local multi-core dev box that alone can leave
most cores idle for the entire run, which is why a 20-minute clip could
take ~2 hours end to end. Two things fix that:
  1. Explicitly size `cpu_threads` instead of leaving it at ctranslate2's
     default.
  2. Split the audio into N segments and transcribe them *concurrently*
     (see `whisper_parallel_chunks`) — ctranslate2 releases the GIL
     during inference, so N Python threads each driving their own
     transcribe() call get close to linear speedup with core count.
On the Colab/GPU worker (WHISPER_DEVICE=cuda) a single pass is already
fast, so parallel splitting is skipped there by default.
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Tuple

from app.config import settings
from app.pipeline import audio as audio_pipeline

_model = None
_model_lock = threading.Lock()


def _resolve_thread_and_worker_counts(num_parts: int) -> Tuple[int, int]:
    """
    Divide available CPU cores between `num_workers` (how many transcribe()
    calls run truly concurrently) and `cpu_threads` (how many threads each
    of those calls gets internally), so the product stays close to the
    machine's core count instead of oversubscribing it.
    """
    total_cores = os.cpu_count() or 4
    if settings.whisper_cpu_threads > 0:
        # Explicit override: honor it per-worker as given.
        cpu_threads = settings.whisper_cpu_threads
        num_workers = max(1, min(num_parts, total_cores // max(1, cpu_threads)) or 1)
    else:
        num_workers = max(1, min(num_parts, total_cores))
        cpu_threads = max(1, total_cores // num_workers)
    return cpu_threads, num_workers


def _get_model(cpu_threads: int, num_workers: int):
    """
    Cache a single WhisperModel per process, sized for the concurrency the
    caller needs. If a later call asks for more workers than the cached
    model supports, rebuild it — this only happens once in practice since
    the pipeline always computes the same sizing for a given machine.
    """
    global _model
    with _model_lock:
        if _model is None or getattr(_model, "_videomind_num_workers", 1) < num_workers:
            from faster_whisper import WhisperModel
            _model = WhisperModel(
                settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
            )
            _model._videomind_num_workers = num_workers  # type: ignore[attr-defined]
        return _model


def _run_single_pass(model, audio_path: str, offset: float = 0.0) -> List[Dict[str, Any]]:
    segments, _info = model.transcribe(
        audio_path,
        beam_size=settings.whisper_beam_size,
        vad_filter=True,
        # condition_on_previous_text=True (the default) feeds each segment's
        # transcript back in as context for the next, which on long/quiet
        # or repetitive audio can make Whisper loop and re-emit the same
        # phrase in consecutive segments. Turning it off trades a small
        # amount of cross-segment context for much lower repetition risk.
        # It's also required for parallel-chunk mode: each part is decoded
        # independently, so there's no valid "previous text" across a part
        # boundary anyway.
        condition_on_previous_text=False,
        # Bail out of a segment early if it looks like a repetition loop.
        compression_ratio_threshold=2.4,
    )
    return [
        {"start": round(seg.start + offset, 2), "end": round(seg.end + offset, 2), "text": seg.text.strip()}
        for seg in segments
    ]


def transcribe_audio(
    audio_path: str,
    duration: Optional[float] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns a list of {"start": float, "end": float, "text": str} segments,
    preserving Whisper's native timestamped segmentation.

    Uses parallel-chunk transcription when running on CPU with a long
    enough clip (see whisper_parallel_chunks / whisper_parallel_min_duration_seconds
    in config); otherwise falls back to a single sequential pass.
    """
    use_parallel = (
        settings.whisper_device == "cpu"
        and settings.whisper_parallel_chunks > 1
        and duration is not None
        and duration >= settings.whisper_parallel_min_duration_seconds
    )

    if not use_parallel:
        cpu_threads, _ = _resolve_thread_and_worker_counts(1)
        model = _get_model(cpu_threads, 1)
        result = _run_single_pass(model, audio_path)
        if progress_callback:
            progress_callback(100.0)
        return _dedupe_repeated_segments(result)

    return _transcribe_parallel(audio_path, duration, progress_callback)


def _transcribe_parallel(
    audio_path: str,
    duration: float,
    progress_callback: Optional[Callable[[float], None]],
) -> List[Dict[str, Any]]:
    num_parts = min(settings.whisper_parallel_chunks, max(1, os.cpu_count() or 1))
    cpu_threads, num_workers = _resolve_thread_and_worker_counts(num_parts)
    model = _get_model(cpu_threads, num_workers)

    split_dir = os.path.join(os.path.dirname(audio_path), "parallel_parts")
    parts = audio_pipeline.split_audio(
        audio_path, duration, num_parts,
        settings.whisper_parallel_overlap_seconds, split_dir,
    )

    # Track progress across concurrently-completing parts: each part
    # reports the (duration-weighted) fraction of the clip it covers once
    # it finishes, so the sum only ever moves forward regardless of which
    # thread finishes first.
    progress_lock = threading.Lock()
    state = {"completed_seconds": 0.0, "last_reported_pct": -1.0}

    def _transcribe_one(part: audio_pipeline.AudioPart) -> List[Dict[str, Any]]:
        segs = _run_single_pass(model, part.path, offset=part.offset)
        # Keep only segments that "belong" to this part's core (non-overlap)
        # range — the overlap padding exists purely to give the model
        # context near the cut, not to be double-counted between parts.
        is_last_part = part.core_end >= duration - 0.01
        kept = [
            s for s in segs
            if part.core_start <= s["start"] < part.core_end
            or (is_last_part and s["start"] >= part.core_start)
        ]

        if progress_callback:
            with progress_lock:
                state["completed_seconds"] += (part.core_end - part.core_start)
                pct = min(99.0, (state["completed_seconds"] / duration) * 100)
                if pct - state["last_reported_pct"] >= 5:
                    progress_callback(pct)
                    state["last_reported_pct"] = pct
        return kept

    all_segments: List[Dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_transcribe_one, p): p for p in parts}
            for future in as_completed(futures):
                all_segments.extend(future.result())
    finally:
        for part in parts:
            if part.path != audio_path:
                try:
                    os.remove(part.path)
                except OSError:
                    pass

    all_segments.sort(key=lambda s: s["start"])
    if progress_callback:
        progress_callback(100.0)
    return _dedupe_repeated_segments(all_segments)


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
