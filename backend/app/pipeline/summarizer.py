"""
Hierarchical summarization: chunk summaries -> section summaries -> final
summary. Uses the LLM provider for each level; timestamps are preserved
throughout so the final summary can reference where in the video each
point came from.
"""
import asyncio
from typing import List, Dict, Any
from app.services.llm_service import get_llm_provider
from app.utils.timestamps import format_timestamp

SECTION_SIZE = 6  # number of chunks grouped into one "section"
MAX_CONCURRENT_LLM_CALLS = 8  # cap parallel Groq calls to avoid rate-limit errors

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


async def generate_hierarchical_summary(chunks: List[Dict[str, Any]], summary_type: str = "short") -> Dict[str, Any]:
    """
    chunks: ordered list of {chunk_id, start_time, end_time, transcript}
    Returns short bullet summary or detailed section-wise summary.

    A long video can have hundreds of 10s chunks. The previous version made
    one blocking Groq call per chunk, one per section, and one for the
    final bullets — all synchronously, on the request's own event loop.
    For a ~450-chunk video that's 450+ sequential network round-trips
    blocking the whole server, which is what made "summarize" appear to
    hang (and made it fragile to any single transient connection error).
    This runs the per-chunk and per-section calls off the event loop with
    bounded concurrency instead of one call at a time.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

    async def _run(fn, *args):
        async with semaphore:
            return await asyncio.to_thread(fn, *args)

    # Level 1: chunk summaries, several in flight at once
    chunk_summaries = await asyncio.gather(*(_run(_summarize_chunk, c) for c in chunks))

    # Level 2: section summaries (group consecutive chunks)
    section_groups = []
    for i in range(0, len(chunks), SECTION_SIZE):
        group_chunks = chunks[i:i + SECTION_SIZE]
        group_summaries = chunk_summaries[i:i + SECTION_SIZE]
        if not any(group_summaries):
            continue
        section_groups.append(
            (group_chunks[0]["start_time"], group_chunks[-1]["end_time"], group_summaries))

    section_texts = await asyncio.gather(
        *(_run(_summarize_section, summaries, start, end) for start, end, summaries in section_groups)
    )

    sections = [
        {
            "start_time": start,
            "end_time": end,
            "start_formatted": format_timestamp(start),
            "end_formatted": format_timestamp(end),
            "summary": text,
        }
        for (start, end, _summaries), text in zip(section_groups, section_texts)
    ]

    if summary_type == "detailed":
        return {"summary_type": "detailed", "sections": sections}

    # Level 3: final short summary (bullets) from section summaries
    provider = get_llm_provider()
    joined_sections = "\n".join(
        f"[{s['start_formatted']}-{s['end_formatted']}] {s['summary']}" for s in sections)
    bullets_text = await asyncio.to_thread(provider.generate, SHORT_SUMMARY_SYSTEM, joined_sections)
    bullet_points = [b.strip("-• ").strip()
                     for b in bullets_text.split("\n") if b.strip()]

    return {"summary_type": "short", "bullet_points": bullet_points, "sections": sections}
