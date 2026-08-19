"""
Evaluation metrics for retrieval and temporal grounding.
No results are fabricated here — these are pure functions; you run them
against your own evaluation dataset (see eval_dataset_template.json) and
report whatever numbers come out.
"""
from typing import List


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of relevant items found within the top-k retrieved items."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & set(relevant_ids))
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """MRR for a single query: 1/rank of the first relevant item, or 0 if none found."""
    relevant_set = set(relevant_ids)
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_set:
            return 1.0 / rank
    return 0.0


def temporal_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Intersection-over-Union between a predicted and ground-truth time interval."""
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)

    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = union_end - union_start

    if union <= 0:
        return 0.0
    return intersection / union


def aggregate_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0
