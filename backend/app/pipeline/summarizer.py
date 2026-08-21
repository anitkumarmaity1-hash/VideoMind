# backend/app/pipeline/summarizer.py — full replacement
"""
Hierarchical summarization: chunk summaries -> section summaries -> final
summary. Chunk and section summaries are independent LLM calls, so they're
parallelized with ThreadPoolExecutor (same pattern used for Groq calls in
the Orchestrate hackathon project) instead of running ~90 sequential calls.
"""
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from app.services.llm_service import get_llm_provider
from app.utils.timestamps import format_timestamp

SECTION_SIZE = 6
MAX_WORKERS = 8  # keep under Groq's concurrent-request limits

CHUNK_SUMMARY_SYSTEM = "You summarize a short video transcript excerpt in one concise sentence. Do not add outside information."
SECTION_SUMMARY_SYSTEM = "You merge several short summaries (from consecutive time windows of a video) into one coherent paragraph, preserving key points."
SHORT_SUMMARY_SYSTEM = "You produce a short 5-8 bullet point summary of a video from its section summaries. Each bullet must be concise and factual."
DETAILED_SUMMARY_SYSTEM = "You produce a detailed, section-wise summary of a video from its section summaries, preserving structure and timestamps."


def _summarize_chunk(chunk: Dict[str, Any]) -> str:
    if not chunk.get("transcript", "").strip():
        return ""
    provider = get_llm_provider()
    prompt = f"Transcript excerpt ({format_timestamp(chunk['start_time'])}-{format_timestamp(chunk['end_time'])}):\n{chunk['transcript']}"
    return provider.generate(CHUNK_SUMMARY_SYSTEM, prompt)


def _summarize_section(chunk_summaries: List[str], start_time: float, end_time: float) -> str:
    provider = get_llm_provider()
    joined = "\n".join(f"- {s}" for s in chunk_summaries if s)
    prompt = f"Time range {format_timestamp(start_time)}-{format_timestamp(end_time)}:\n{joined}"
    return provider.generate(SECTION_SUMMARY_SYSTEM, prompt)


def generate_hierarchical_summary(chunks: List[Dict[str, Any]], summary_type: str = "short") -> Dict[str, Any]:
    # Level 1: chunk summaries, in parallel, order preserved via map()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        chunk_summaries = list(pool.map(_summarize_chunk, chunks))

    # Level 2: section summaries, also in parallel — each group is independent
    groups = []
    for i in range(0, len(chunks), SECTION_SIZE):
        group_chunks = chunks[i:i + SECTION_SIZE]
        group_summaries = chunk_summaries[i:i + SECTION_SIZE]
        if any(group_summaries):
            groups.append((group_chunks, group_summaries))

    def _do_section(group):
        group_chunks, group_summaries = group
        start_time = group_chunks[0]["start_time"]
        end_time = group_chunks[-1]["end_time"]
        return {
            "start_time": start_time,
            "end_time": end_time,
            "start_formatted": format_timestamp(start_time),
            "end_formatted": format_timestamp(end_time),
            "summary": _summarize_section(group_summaries, start_time, end_time),
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        sections = list(pool.map(_do_section, groups))

    provider = get_llm_provider()

    if summary_type == "detailed":
        return {"summary_type": "detailed", "sections": sections}

    joined_sections = "\n".join(
        f"[{s['start_formatted']}-{s['end_formatted']}] {s['summary']}" for s in sections)
    bullets_text = provider.generate(SHORT_SUMMARY_SYSTEM, joined_sections)
    bullet_points = [b.strip("-• ").strip()
                     for b in bullets_text.split("\n") if b.strip()]

    return {"summary_type": "short", "bullet_points": bullet_points, "sections": sections}
