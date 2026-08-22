import asyncio
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException

from app.database.mongo import videos_collection, video_segments_collection, questions_collection, answers_collection
from app.config import settings
from app.models.question import QuestionRequest, AnswerResponse, EvidenceItem, SummaryRequest, SummaryResponse
from app.pipeline.question_router import classify_question, QuestionType
from app.pipeline.retrieval import retrieve_text, retrieve_visual, fuse_scores
from app.pipeline.reranking import expand_temporal_neighbors
from app.pipeline.summarizer import generate_hierarchical_summary
from app.services.llm_service import generate_grounded_answer
from app.utils.timestamps import format_timestamp

router = APIRouter(prefix="/api/videos", tags=["questions"])


@router.post("/{video_id}/questions", response_model=AnswerResponse)
async def ask_question(video_id: str, request: QuestionRequest):
    video = await videos_collection().find_one({"video_id": video_id})
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video["processing_status"] != "ready":
        raise HTTPException(
            status_code=409, detail=f"Video is not ready yet (status: {video['processing_status']})")

    question_type = classify_question(request.question)
    # Broad "list every X" questions need evidence pulled from across the
    # whole video rather than the usual narrow top-5 — see broad_question_top_k.
    is_broad = question_type in (
        QuestionType.SUMMARY, QuestionType.ENUMERATION)
    retrieval_top_k = settings.broad_question_top_k if is_broad else None

    # All chunks for temporal expansion context
    segments_cursor = video_segments_collection().find(
        {"video_id": video_id}).sort("chunk_id", 1)
    all_segments = await segments_cursor.to_list(length=10000)
    all_chunks_by_id = {s["chunk_id"]: s for s in all_segments}

    # embedding + Pinecone query + Groq are all synchronous/network calls;
    # run them off the event loop so one slow question doesn't stall
    # every other request (status polls, other users' questions, etc).
    text_results, visual_results = await asyncio.gather(
        asyncio.to_thread(retrieve_text, request.question,
                          video_id, retrieval_top_k),
        asyncio.to_thread(retrieve_visual, request.question,
                          video_id, retrieval_top_k),
    )
    fused = fuse_scores(text_results, visual_results)

    evidence_blocks = expand_temporal_neighbors(fused, all_chunks_by_id)

    text_evidence = [
        {
            "start_time": b["start_time"],
            "end_time": b["end_time"],
            "start_formatted": format_timestamp(b["start_time"]),
            "end_formatted": format_timestamp(b["end_time"]),
            "content": b["transcript"],
            "score": b["score"],
        }
        for b in evidence_blocks if b["transcript"]
    ]
    # Show the strongest evidence first, both to the person reading it and
    # (more importantly) to the LLM assembling the answer.
    text_evidence.sort(key=lambda e: e["score"], reverse=True)

    visual_evidence = [
        {
            "start_time": r["metadata"]["start_time"],
            "end_time": r["metadata"]["end_time"],
            "start_formatted": format_timestamp(r["metadata"]["start_time"]),
            "end_formatted": format_timestamp(r["metadata"]["end_time"]),
        }
        for r in visual_results
    ]

    answer_text = await asyncio.to_thread(
        generate_grounded_answer, request.question, text_evidence, visual_evidence, request.answer_mode
    )

    question_id = f"q_{uuid.uuid4().hex[:10]}"
    answer_id = f"ans_{uuid.uuid4().hex[:10]}"

    await questions_collection().insert_one({
        "question_id": question_id,
        "video_id": video_id,
        "question": request.question,
        "timestamp": datetime.utcnow(),
    })

    evidence_items = [
        EvidenceItem(
            start_time=e["start_time"], end_time=e["end_time"],
            start_formatted=e["start_formatted"], end_formatted=e["end_formatted"],
            modality="text", content="[transcript evidence]",
            score=e["score"],
        )
        for e in text_evidence
    ] + [
        EvidenceItem(
            start_time=e["start_time"], end_time=e["end_time"],
            start_formatted=e["start_formatted"], end_formatted=e["end_formatted"],
            modality="visual", content="[visual frame evidence]",
            score=next((r["score"] for r in visual_results if r["metadata"].get(
                "start_time") == e["start_time"]), 0.0),
        )
        for e in visual_evidence
    ]

    await answers_collection().insert_one({
        "answer_id": answer_id,
        "question_id": question_id,
        "answer": answer_text,
        "evidence": [e.model_dump() for e in evidence_items],
        "created_at": datetime.utcnow(),
    })

    return AnswerResponse(
        answer_id=answer_id,
        question_id=question_id,
        video_id=video_id,
        question=request.question,
        answer=answer_text,
        evidence=evidence_items,
        question_type=question_type.value,
    )


@router.post("/{video_id}/summary", response_model=SummaryResponse)
async def summarize_video(video_id: str, request: SummaryRequest):
    video = await videos_collection().find_one({"video_id": video_id})
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video["processing_status"] != "ready":
        raise HTTPException(
            status_code=409, detail=f"Video is not ready yet (status: {video['processing_status']})")

    segments_cursor = video_segments_collection().find(
        {"video_id": video_id}).sort("chunk_id", 1)
    chunks = await segments_cursor.to_list(length=10000)
    if not chunks:
        raise HTTPException(
            status_code=404, detail="No segments found for this video")

    result = await generate_hierarchical_summary(chunks, request.summary_type)

    if request.summary_type == "detailed":
        return SummaryResponse(video_id=video_id, summary_type="detailed", summary="", sections=result["sections"])

    summary_text = "\n".join(f"- {b}" for b in result["bullet_points"])
    return SummaryResponse(video_id=video_id, summary_type="short", summary=summary_text, bullet_points=result["bullet_points"])
