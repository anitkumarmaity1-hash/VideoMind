"""
Orchestrates the full video-processing pipeline end to end:
audio extraction -> ASR -> chunking -> frame sampling -> embeddings ->
Pinecone indexing -> mark video ready.

Designed to run as a FastAPI BackgroundTask locally (CPU, slow), or be
invoked from the Colab GPU worker notebook (fast) against the same
MongoDB Atlas + Pinecone backing services. Either caller just imports
and calls `run_pipeline(video_id)`.
"""
import os
import uuid
from datetime import datetime

from app.config import settings
from app.database.mongo import (
    videos_collection,
    video_segments_collection,
    processing_jobs_collection,
)
from app.models.video import ProcessingStatus
from app.pipeline import audio, transcription, chunking, frames, text_embeddings, visual_embeddings
from app.services import vector_store


async def _update_status(video_id: str, status: ProcessingStatus, error: str = None):
    update = {"processing_status": status.value}
    if error:
        update["error_message"] = error
    await videos_collection().update_one({"video_id": video_id}, {"$set": update})


async def _log_job(video_id: str, stage: str, progress: int, status: str, error: str = None):
    await processing_jobs_collection().update_one(
        {"video_id": video_id, "stage": stage},
        {
            "$set": {
                "job_id": f"job_{uuid.uuid4().hex[:8]}",
                "video_id": video_id,
                "stage": stage,
                "progress": progress,
                "status": status,
                "error_message": error,
                "updated_at": datetime.utcnow(),
            },
            "$setOnInsert": {"created_at": datetime.utcnow()},
        },
        upsert=True,
    )


async def run_pipeline(video_id: str):
    """Synchronous-style orchestration using async Mongo calls between CPU-bound steps."""
    video_doc = await videos_collection().find_one({"video_id": video_id})
    if not video_doc:
        raise ValueError(f"Video {video_id} not found")

    video_path = video_doc["storage_path"]
    audio_dir = os.path.join(settings.local_data_dir, "audio", video_id)
    frames_dir = os.path.join(settings.local_data_dir, "frames", video_id)
    audio_path = os.path.join(audio_dir, "audio.wav")

    try:
        # --- Stage: audio extraction ---
        await _update_status(video_id, ProcessingStatus.EXTRACTING_AUDIO)
        await _log_job(video_id, "extracting_audio", 10, "running")
        duration = audio.get_video_duration(video_path)
        audio.extract_audio(video_path, audio_path)
        await videos_collection().update_one({"video_id": video_id}, {"$set": {"duration": duration}})
        await _log_job(video_id, "extracting_audio", 100, "done")

        # --- Stage: transcription ---
        await _update_status(video_id, ProcessingStatus.TRANSCRIBING)
        await _log_job(video_id, "transcribing", 10, "running")
        transcript_segments = transcription.transcribe_audio(audio_path)
        await _log_job(video_id, "transcribing", 100, "done")

        # --- Stage: temporal chunking ---
        chunks = chunking.create_temporal_chunks(transcript_segments, duration)

        # --- Stage: frame extraction ---
        await _update_status(video_id, ProcessingStatus.EXTRACTING_FRAMES)
        await _log_job(video_id, "extracting_frames", 10, "running")
        frame_results = frames.extract_frames(video_path, frames_dir)
        frame_timestamps = [ts for ts, _path in frame_results]
        chunks = chunking.attach_frame_timestamps(chunks, frame_timestamps)
        await _log_job(video_id, "extracting_frames", 100, "done")

        # --- Persist segments to MongoDB ---
        segment_docs = []
        for c in chunks:
            segment_docs.append({
                "segment_id": f"{video_id}_seg_{c['chunk_id']}",
                "video_id": video_id,
                "chunk_id": c["chunk_id"],
                "transcript": c["transcript"],
                "start_time": c["start_time"],
                "end_time": c["end_time"],
                "frame_timestamps": c["frame_timestamps"],
            })
        if segment_docs:
            await video_segments_collection().delete_many({"video_id": video_id})
            await video_segments_collection().insert_many(segment_docs)

        # --- Stage: embeddings ---
        await _update_status(video_id, ProcessingStatus.EMBEDDING)
        await _log_job(video_id, "embedding", 10, "running")

        text_vectors = text_embeddings.embed_texts([c["transcript"] or " " for c in chunks])

        # pick the middle frame for each chunk (if any) as its visual representative
        chunk_to_frame = {}
        for c in chunks:
            if c["frame_timestamps"]:
                mid_ts = c["frame_timestamps"][len(c["frame_timestamps"]) // 2]
                closest = min(frame_results, key=lambda fr: abs(fr[0] - mid_ts))
                chunk_to_frame[c["chunk_id"]] = closest[1]

        visual_paths = list(chunk_to_frame.values())
        visual_vectors = visual_embeddings.embed_images(visual_paths) if visual_paths else []
        visual_vector_map = dict(zip(chunk_to_frame.keys(), visual_vectors))

        await _log_job(video_id, "embedding", 100, "done")

        # --- Stage: Pinecone indexing ---
        await _update_status(video_id, ProcessingStatus.INDEXING)
        await _log_job(video_id, "indexing", 10, "running")

        text_upserts = []
        for c, vec in zip(chunks, text_vectors):
            text_upserts.append({
                "id": f"{video_id}_text_{c['chunk_id']}",
                "values": vec,
                "metadata": {
                    "video_id": video_id,
                    "chunk_id": c["chunk_id"],
                    "start_time": c["start_time"],
                    "end_time": c["end_time"],
                    "modality": "text",
                    "transcript": c["transcript"][:1000],
                },
            })
        if text_upserts:
            vector_store.upsert("text", text_upserts)

        visual_upserts = []
        for chunk_id, vec in visual_vector_map.items():
            c = next(c for c in chunks if c["chunk_id"] == chunk_id)
            visual_upserts.append({
                "id": f"{video_id}_visual_{chunk_id}",
                "values": vec,
                "metadata": {
                    "video_id": video_id,
                    "chunk_id": chunk_id,
                    "start_time": c["start_time"],
                    "end_time": c["end_time"],
                    "modality": "visual",
                },
            })
        if visual_upserts:
            vector_store.upsert("visual", visual_upserts)

        await _log_job(video_id, "indexing", 100, "done")

        # --- Done ---
        await _update_status(video_id, ProcessingStatus.READY)

    except Exception as e:
        await _update_status(video_id, ProcessingStatus.FAILED, error=str(e))
        await _log_job(video_id, "pipeline", 0, "failed", error=str(e))
        raise
