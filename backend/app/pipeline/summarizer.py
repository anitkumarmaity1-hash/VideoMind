"""
Hierarchical summarization: section summaries -> final summary. Uses the
LLM provider for each level; timestamps are preserved throughout so the
final summary can reference where in the video each point came from.
"""
import asyncio
from typing import List, Dict, Any
from app.config import settings
from app.services.llm_service import get_llm_provider
from app.utils.timestamps import format_timestamp

# Number of ~10s transcript chunks grouped into one "section" and
# summarized with a single Groq call. Chosen to keep each section's raw
# transcript comfortably within one request while keeping the total call
# count low — see the note on `generate_hierarchical_summary` below for
# why call count matters as much as concurrency here.
SECTION_SIZE = 9  # ~90s of video per section

SECTION_SUMMARY_SYSTEM = "You summarize a section of a video's spoken transcript (given as timestamped excerpts) into one coherent paragraph, preserving key points and factual detail. Do not add outside information."
SHORT_SUMMARY_SYSTEM = "You produce a short 5-8 bullet point summary of a video from its section summaries. Each bullet must be concise and factual."
DETAILED_SUMMARY_SYSTEM = "You produce a detailed, section-wise summary of a video from its section summaries, preserving structure and timestamps."


def _summarize_section(chunks: List[Dict[str, Any]], start_time: float, end_time: float) -> str:
    provider = get_llm_provider()
    transcript_block = "\n".join(
        f"[{format_timestamp(c['start_time'])}-{format_timestamp(c['end_time'])}] {c['transcript']}"
        for c in chunks if c.get("transcript", "").strip()
    )
    if not transcript_block:
        return ""
    prompt = f"Time range {format_timestamp(start_time)}-{format_timestamp(end_time)}, spoken transcript:\n{transcript_block}"
    return provider.generate(SECTION_SUMMARY_SYSTEM, prompt)


async def generate_hierarchical_summary(chunks: List[Dict[str, Any]], summary_type: str = "short") -> Dict[str, Any]:
    """
    chunks: ordered list of {chunk_id, start_time, end_time, transcript}
    Returns short bullet summary or detailed section-wise summary.

    Groq's free-tier account is capped at a strict, low tokens-per-minute
    budget (see llm_service._TokenRateLimiter). The original design made
    one LLM call per ~10s chunk *plus* one per section *plus* one final
    call — for a ~450-chunk video that's 450+ calls, each with its own
    fixed per-request overhead (system prompt, framing tokens), which
    burned through the TPM budget on overhead alone before the useful
    content did. Summarizing directly at the section level (one call per
    ~90s of video, skipping the intermediate per-chunk pass) cuts total
    call count roughly 6x for the same content, which matters more here
    than concurrency does — the token-rate limiter paces however many
    calls are actually made, but it can't make each call cheaper.
    Concurrency is still bounded (below) so a burst of section calls
    doesn't itself spike past the per-minute budget before the limiter
    can react.
    """
    semaphore = asyncio.Semaphore(settings.groq_max_concurrent_calls)

    async def _run(fn, *args):
        async with semaphore:
            return await asyncio.to_thread(fn, *args)

    section_groups = []
    for i in range(0, len(chunks), SECTION_SIZE):
        group_chunks = chunks[i:i + SECTION_SIZE]
        if not any(c.get("transcript", "").strip() for c in group_chunks):
            continue
        section_groups.append(
            (group_chunks[0]["start_time"], group_chunks[-1]["end_time"], group_chunks))

    section_texts = await asyncio.gather(
        *(_run(_summarize_section, group_chunks, start, end) for start, end, group_chunks in section_groups)
    )

    sections = [
        {
            "start_time": start,
            "end_time": end,
            "start_formatted": format_timestamp(start),
            "end_formatted": format_timestamp(end),
            "summary": text,
        }
        for (start, end, _group_chunks), text in zip(section_groups, section_texts)
        if text
    ]

    if summary_type == "detailed":
        return {"summary_type": "detailed", "sections": sections}

    if not sections:
        return {"summary_type": "short", "bullet_points": [], "sections": []}

    # Final short summary (bullets) from section summaries
    provider = get_llm_provider()
    joined_sections = "\n".join(
        f"[{s['start_formatted']}-{s['end_formatted']}] {s['summary']}" for s in sections)
    bullets_text = await asyncio.to_thread(provider.generate, SHORT_SUMMARY_SYSTEM, joined_sections)
    bullet_points = [b.strip("-• ").strip()
                     for b in bullets_text.split("\n") if b.strip()]

    return {"summary_type": "short", "bullet_points": bullet_points, "sections": sections}
