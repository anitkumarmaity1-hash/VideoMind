"""
VectorStore abstraction over Pinecone, with two logical namespaces:
text embeddings and visual embeddings, held in two separate indexes
(simplest way to use different dims/metadata cleanly with Pinecone).

Methods: upsert(), query(), delete_video(), get_video_chunks()
"""
from typing import List, Dict, Any, Literal
from app.config import settings

_pc_client = None
_text_index = None
_visual_index = None


def _get_pinecone():
    global _pc_client
    if _pc_client is None:
        from pinecone import Pinecone
        _pc_client = Pinecone(api_key=settings.pinecone_api_key)
    return _pc_client


def _ensure_index(name: str, dim: int):
    from pinecone import ServerlessSpec
    pc = _get_pinecone()
    existing = [idx["name"] for idx in pc.list_indexes()]
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.pinecone_env),
        )
    return pc.Index(name)


def _text_idx():
    global _text_index
    if _text_index is None:
        _text_index = _ensure_index(settings.pinecone_text_index, settings.text_embedding_dim)
    return _text_index


def _visual_idx():
    global _visual_index
    if _visual_index is None:
        _visual_index = _ensure_index(settings.pinecone_visual_index, settings.visual_embedding_dim)
    return _visual_index


def _index_for(modality: Literal["text", "visual"]):
    return _text_idx() if modality == "text" else _visual_idx()


def upsert(modality: Literal["text", "visual"], vectors: List[Dict[str, Any]]) -> None:
    """
    vectors: [{"id": str, "values": [float,...], "metadata": {...}}, ...]
    Required metadata keys: video_id, chunk_id, start_time, end_time, modality
    """
    index = _index_for(modality)
    index.upsert(vectors=vectors)


def query(modality: Literal["text", "visual"], vector: List[float], video_id: str, top_k: int = None) -> List[Dict[str, Any]]:
    top_k = top_k or settings.top_k
    index = _index_for(modality)
    result = index.query(
        vector=vector,
        top_k=top_k,
        filter={"video_id": {"$eq": video_id}},
        include_metadata=True,
    )
    return [
        {
            "id": match["id"],
            "score": match["score"],
            "metadata": match.get("metadata", {}),
        }
        for match in result.get("matches", [])
    ]


def delete_video(video_id: str) -> None:
    for index in (_text_idx(), _visual_idx()):
        index.delete(filter={"video_id": {"$eq": video_id}})


def get_video_chunks(modality: Literal["text", "visual"], video_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Pinecone doesn't support arbitrary metadata scans well; for full chunk
    listing we rely on MongoDB (video_segments collection) as source of
    truth instead. This method is kept for interface completeness and uses
    a dummy zero-vector query as a fallback approximation.
    """
    index = _index_for(modality)
    dim = settings.text_embedding_dim if modality == "text" else settings.visual_embedding_dim
    result = index.query(
        vector=[0.0] * dim,
        top_k=limit,
        filter={"video_id": {"$eq": video_id}},
        include_metadata=True,
    )
    return [m.get("metadata", {}) for m in result.get("matches", [])]
