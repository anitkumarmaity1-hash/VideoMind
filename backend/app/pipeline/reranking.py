"""
Temporal neighbor expansion: merges retrieved chunks that are temporally
adjacent into coherent evidence blocks, and de-duplicates overlapping
evidence.
"""
from typing import List, Dict, Any
from app.config import settings


def expand_temporal_neighbors(
    fused_results: List[Dict[str, Any]],
    all_chunks_by_id: Dict[int, Dict[str, Any]],
    window: int = None,
) -> List[Dict[str, Any]]:
    """
    For each top result, pull in `window` neighboring chunk_ids on either
    side (if they exist in all_chunks_by_id) to build a fuller evidence
    block, then merge/deduplicate overlapping ranges.

    all_chunks_by_id: {chunk_id: {"start_time", "end_time", "transcript", ...}}
    """
    window = settings.temporal_neighbor_window if window is None else window

    expanded_chunk_ids = set()
    for r in fused_results:
        center = r["chunk_id"]
        for offset in range(-window, window + 1):
            neighbor_id = center + offset
            if neighbor_id in all_chunks_by_id:
                expanded_chunk_ids.add(neighbor_id)

    ordered_ids = sorted(expanded_chunk_ids)
    blocks = _merge_contiguous(ordered_ids, all_chunks_by_id)
    return blocks


def _merge_contiguous(chunk_ids: List[int], all_chunks_by_id: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge consecutive chunk_ids into single evidence blocks with combined transcript."""
    if not chunk_ids:
        return []

    blocks = []
    current_ids = [chunk_ids[0]]

    for cid in chunk_ids[1:]:
        if cid == current_ids[-1] + 1:
            current_ids.append(cid)
        else:
            blocks.append(_build_block(current_ids, all_chunks_by_id))
            current_ids = [cid]
    blocks.append(_build_block(current_ids, all_chunks_by_id))
    return blocks


def _build_block(ids: List[int], all_chunks_by_id: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    chunks = [all_chunks_by_id[i] for i in ids]
    return {
        "chunk_ids": ids,
        "start_time": chunks[0]["start_time"],
        "end_time": chunks[-1]["end_time"],
        "transcript": " ".join(c.get("transcript", "").strip() for c in chunks).strip(),
    }
