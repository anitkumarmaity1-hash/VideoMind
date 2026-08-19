import os
import tempfile
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Form
from app.config import settings
from app.utils.validation import validate_extension, validate_size, generate_video_id, ValidationError
from app.database.mongo import videos_collection, video_segments_collection
from app.models.video import VideoMetadata, VideoResponse, VideoStatusResponse
from app.models.segment import SegmentResponse
from app.services.storage import get_storage_backend
from app.services.youtube_service import validate_url, get_permitted_metadata, download_video, YouTubeError
from app.pipeline.pipeline_runner import run_pipeline
from app.services import vector_store
from app.utils.timestamps import format_timestamp

router = APIRouter(prefix="/api/videos", tags=["videos"])


@router.post("/upload", response_model=VideoResponse)
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        validate_extension(file.filename)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    contents = await file.read()
    try:
        validate_size(len(contents))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    video_id = generate_video_id()
    ext = os.path.splitext(file.filename)[1].lower()

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(tmp_fd, "wb") as f:
        f.write(contents)

    storage = get_storage_backend()
    dest_relative = os.path.join("videos", f"{video_id}{ext}")
    final_path = storage.save(tmp_path, dest_relative)

    doc = VideoMetadata(
        video_id=video_id,
        filename=file.filename,
        storage_path=final_path,
        source="upload",
    )
    await videos_collection().insert_one(doc.model_dump())

    background_tasks.add_task(run_pipeline, video_id)

    return VideoResponse(
        video_id=video_id,
        filename=file.filename,
        duration=None,
        upload_time=doc.upload_time,
        processing_status=doc.processing_status,
        source="upload",
    )


@router.post("/url", response_model=VideoResponse)
async def upload_from_youtube(background_tasks: BackgroundTasks, url: str = Form(...)):
    if not validate_url(url):
        raise HTTPException(status_code=400, detail="Invalid or unsupported YouTube URL")

    try:
        metadata = get_permitted_metadata(url)
    except YouTubeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    video_id = generate_video_id()
    storage = get_storage_backend()
    dest_relative = os.path.join("videos", f"{video_id}.mp4")

    tmp_path = os.path.join(tempfile.gettempdir(), f"{video_id}.mp4")
    try:
        download_video(url, tmp_path)
    except YouTubeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    final_path = storage.save(tmp_path, dest_relative)

    doc = VideoMetadata(
        video_id=video_id,
        filename=metadata.get("title", "youtube_video"),
        storage_path=final_path,
        source="youtube",
        source_url=url,
    )
    await videos_collection().insert_one(doc.model_dump())

    background_tasks.add_task(run_pipeline, video_id)

    return VideoResponse(
        video_id=video_id,
        filename=doc.filename,
        duration=None,
        upload_time=doc.upload_time,
        processing_status=doc.processing_status,
        source="youtube",
    )


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(video_id: str):
    doc = await videos_collection().find_one({"video_id": video_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoResponse(
        video_id=doc["video_id"],
        filename=doc["filename"],
        duration=doc.get("duration"),
        upload_time=doc["upload_time"],
        processing_status=doc["processing_status"],
        source=doc.get("source", "upload"),
    )


@router.get("/{video_id}/status", response_model=VideoStatusResponse)
async def get_video_status(video_id: str):
    doc = await videos_collection().find_one({"video_id": video_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoStatusResponse(
        video_id=video_id,
        processing_status=doc["processing_status"],
        error_message=doc.get("error_message"),
    )


@router.get("/{video_id}/segments", response_model=list[SegmentResponse])
async def get_video_segments(video_id: str):
    cursor = video_segments_collection().find({"video_id": video_id}).sort("chunk_id", 1)
    segments = await cursor.to_list(length=10000)
    return [
        SegmentResponse(
            chunk_id=s["chunk_id"],
            start_time=s["start_time"],
            end_time=s["end_time"],
            transcript=s["transcript"],
            start_formatted=format_timestamp(s["start_time"]),
            end_formatted=format_timestamp(s["end_time"]),
        )
        for s in segments
    ]


@router.delete("/{video_id}")
async def delete_video(video_id: str):
    doc = await videos_collection().find_one({"video_id": video_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Video not found")

    storage = get_storage_backend()
    rel_path = os.path.relpath(doc["storage_path"], start=settings.local_data_dir)
    try:
        storage.delete(rel_path)
    except Exception:
        pass  # non-fatal — metadata cleanup still proceeds

    vector_store.delete_video(video_id)
    await video_segments_collection().delete_many({"video_id": video_id})
    await videos_collection().delete_one({"video_id": video_id})

    return {"deleted": True, "video_id": video_id}
