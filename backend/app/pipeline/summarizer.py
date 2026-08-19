"""
Hierarchical summarization: chunk summaries -> section summaries -> final
summary. Uses the LLM provider for each level; timestamps are preserved
throughout so the final summary can reference where in the video each
point came from.
"""
from typing import List, Dict, Any
from app.services.llm_service import get_llm_provider
from app.utils.timestamps import format_timestamp

SECTION_SIZE = 6  # number of chunks grouped into one "section"

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
    """
    chunks: ordered list of {chunk_id, start_time, end_time, transcript}
    Returns short bullet summary or detailed section-wise summary.
    """
    # Level 1: chunk summaries
    chunk_summaries = [_summarize_chunk(c) for c in chunks]

    # Level 2: section summaries (group consecutive chunks)
    sections = []
    for i in range(0, len(chunks), SECTION_SIZE):
        group_chunks = chunks[i:i + SECTION_SIZE]
        group_summaries = chunk_summaries[i:i + SECTION_SIZE]
        if not any(group_summaries):
            continue
        start_time = group_chunks[0]["start_time"]
        end_time = group_chunks[-1]["end_time"]
        section_text = _summarize_section(group_summaries, start_time, end_time)
        sections.append({
            "start_time": start_time,
            "end_time": end_time,
            "start_formatted": format_timestamp(start_time),
            "end_formatted": format_timestamp(end_time),
            "summary": section_text,
        })

    provider = get_llm_provider()

    if summary_type == "detailed":
        return {"summary_type": "detailed", "sections": sections}

    # Level 3: final short summary (bullets) from section summaries
    joined_sections = "\n".join(f"[{s['start_formatted']}-{s['end_formatted']}] {s['summary']}" for s in sections)
    bullets_text = provider.generate(SHORT_SUMMARY_SYSTEM, joined_sections)
    bullet_points = [b.strip("-• ").strip() for b in bullets_text.split("\n") if b.strip()]

    return {"summary_type": "short", "bullet_points": bullet_points, "sections": sections}
