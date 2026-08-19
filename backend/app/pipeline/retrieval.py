"""
Multimodal retrieval: text retrieval, visual retrieval, and score fusion.
"""
from typing import List, Dict, Any
from app.config import settings
from app.services import vector_store
from app.pipeline.text_embeddings import embed_query
from app.pipeline.visual_embeddings import embed_text_for_visual_search


def retrieve_text(question: str, video_id: str, top_k: int = None) -> List[Dict[str, Any]]:
    vector = embed_query(question)
    return vector_store.query("text", vector, video_id, top_k=top_k)


def retrieve_visual(question: str, video_id: str, top_k: int = None) -> List[Dict[str, Any]]:
    vector = embed_text_for_visual_search(question)
    return vector_store.query("visual", vector, video_id, top_k=top_k)


def fuse_scores(
    text_results: List[Dict[str, Any]],
    visual_results: List[Dict[str, Any]],
    text_weight: float = None,
    visual_weight: float = None,
) -> List[Dict[str, Any]]:
    """
    Combine text and visual retrieval results keyed by chunk_id, using:
        final_score = text_weight * text_score + visual_weight * visual_score
    Chunks present in only one modality get a partial score (missing side = 0).
    """
    text_weight = settings.text_score_weight if text_weight is None else text_weight
    visual_weight = settings.visual_score_weight if visual_weight is None else visual_weight

    by_chunk: Dict[int, Dict[str, Any]] = {}

    for r in text_results:
        chunk_id = r["metadata"]["chunk_id"]
        by_chunk.setdefault(chunk_id, {"metadata": r["metadata"], "text_score": 0.0, "visual_score": 0.0})
        by_chunk[chunk_id]["text_score"] = r["score"]

    for r in visual_results:
        chunk_id = r["metadata"]["chunk_id"]
        by_chunk.setdefault(chunk_id, {"metadata": r["metadata"], "text_score": 0.0, "visual_score": 0.0})
        by_chunk[chunk_id]["visual_score"] = r["score"]

    fused = []
    for chunk_id, data in by_chunk.items():
        final_score = text_weight * data["text_score"] + visual_weight * data["visual_score"]
        fused.append({
            "chunk_id": chunk_id,
            "metadata": data["metadata"],
            "text_score": data["text_score"],
            "visual_score": data["visual_score"],
            "final_score": round(final_score, 4),
        })

    fused.sort(key=lambda x: x["final_score"], reverse=True)
    return fused
