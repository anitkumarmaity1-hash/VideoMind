from app.pipeline.retrieval import fuse_scores


def make_result(chunk_id, score, start=0, end=10):
    return {
        "score": score,
        "metadata": {"chunk_id": chunk_id, "start_time": start, "end_time": end},
    }


def test_fuse_scores_weighted_sum():
    text_results = [make_result(0, 0.8)]
    visual_results = [make_result(0, 0.5)]
    fused = fuse_scores(text_results, visual_results, text_weight=0.6, visual_weight=0.4)
    assert fused[0]["chunk_id"] == 0
    assert fused[0]["final_score"] == round(0.6 * 0.8 + 0.4 * 0.5, 4)


def test_fuse_scores_text_only_chunk():
    text_results = [make_result(1, 0.9)]
    visual_results = []
    fused = fuse_scores(text_results, visual_results, text_weight=0.6, visual_weight=0.4)
    assert fused[0]["visual_score"] == 0.0
    assert fused[0]["final_score"] == round(0.6 * 0.9, 4)


def test_fuse_scores_sorted_descending():
    text_results = [make_result(0, 0.2), make_result(1, 0.9)]
    fused = fuse_scores(text_results, [], text_weight=1.0, visual_weight=0.0)
    assert fused[0]["chunk_id"] == 1
    assert fused[1]["chunk_id"] == 0
