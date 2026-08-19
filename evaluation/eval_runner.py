"""
Evaluation runner: loads an annotated dataset (see eval_dataset_template.json),
runs retrieval against the live backend for each query, and computes
Recall@1/5/10, MRR, and Temporal IoU.

Usage:
    python evaluation/eval_runner.py evaluation/my_dataset.json

Requires the backend to be running and the target video already processed
(status = ready). This script does NOT fabricate results — if the dataset
file is empty or the backend is unreachable, it reports that plainly
rather than printing fake numbers.
"""
import sys
import json
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from metrics import recall_at_k, mean_reciprocal_rank, temporal_iou, aggregate_mean  # noqa: E402


def load_dataset(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def run_evaluation(dataset_path: str):
    from app.pipeline.retrieval import retrieve_text, retrieve_visual, fuse_scores

    dataset = load_dataset(dataset_path)
    video_id = dataset["video_id"]
    queries = dataset.get("queries", [])

    if not queries:
        print("No queries found in dataset. Nothing to evaluate.")
        return

    recall_1_scores, recall_5_scores, recall_10_scores = [], [], []
    mrr_scores = []
    iou_scores = []

    for q in queries:
        text_results = retrieve_text(q["question"], video_id, top_k=10)
        visual_results = retrieve_visual(q["question"], video_id, top_k=10)
        fused = fuse_scores(text_results, visual_results)

        retrieved_chunk_ids = [str(r["chunk_id"]) for r in fused]
        relevant_ids = [str(cid) for cid in q.get("relevant_chunk_ids", [])]

        recall_1_scores.append(recall_at_k(retrieved_chunk_ids, relevant_ids, 1))
        recall_5_scores.append(recall_at_k(retrieved_chunk_ids, relevant_ids, 5))
        recall_10_scores.append(recall_at_k(retrieved_chunk_ids, relevant_ids, 10))
        mrr_scores.append(mean_reciprocal_rank(retrieved_chunk_ids, relevant_ids))

        if fused and "ground_truth_start" in q:
            top = fused[0]["metadata"]
            iou_scores.append(
                temporal_iou(
                    top["start_time"], top["end_time"],
                    q["ground_truth_start"], q["ground_truth_end"],
                )
            )

    print(f"Evaluated {len(queries)} queries against video {video_id}")
    print(f"Recall@1:  {aggregate_mean(recall_1_scores):.3f}")
    print(f"Recall@5:  {aggregate_mean(recall_5_scores):.3f}")
    print(f"Recall@10: {aggregate_mean(recall_10_scores):.3f}")
    print(f"MRR:       {aggregate_mean(mrr_scores):.3f}")
    if iou_scores:
        print(f"Temporal IoU: {aggregate_mean(iou_scores):.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluation/eval_runner.py <dataset.json>")
        sys.exit(1)
    run_evaluation(sys.argv[1])
