"""
LLM provider abstraction. Groq is the default/active provider.
Adding OpenAI / Anthropic / Gemini / Ollama later means implementing
LLMProvider and registering it in get_llm_provider() — no changes needed
elsewhere in the app.
"""
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from app.config import settings

# Rate-limit errors get a larger retry budget than other transient errors:
# they're *expected* under Groq's low free-tier TPM cap and are always
# recoverable by waiting, unlike a genuine server error.
MAX_RETRIES = 3
MAX_RATE_LIMIT_RETRIES = 6
RETRY_BACKOFF_SECONDS = 2.0


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        ...


def _estimate_tokens(*texts: str) -> int:
    """Rough token estimate (~4 chars/token for English) plus a fixed
    overhead for chat-completion framing (role wrappers, etc). Good enough
    to pace requests against a TPM budget — it doesn't need to be exact,
    just not wildly optimistic."""
    return sum(max(1, len(t) // 4) for t in texts) + 50


def _parse_retry_after_seconds(exc: Exception, default: float) -> float:
    """Groq's 429 body includes a human-readable 'Please try again in
    15.94s' — use it directly when present instead of guessing with a
    generic backoff, since it reflects exactly how long the account's
    rolling TPM window needs to drain."""
    match = re.search(r"try again in ([\d.]+)s", str(exc))
    if match:
        try:
            return float(match.group(1)) + 0.25
        except ValueError:
            pass
    return default


class _TokenRateLimiter:
    """
    Sliding-window token budget shared across every Groq call in this
    process (question answering + all levels of summarization). Groq's
    free-tier TPM cap is account-wide, not per-request, so bounding
    *concurrency* alone (see summarizer.MAX_CONCURRENT_LLM_CALLS) isn't
    enough — a handful of concurrent requests can each be small on their
    own and still blow through the cap in aggregate over a rolling
    60-second window. This blocks the calling thread until there's
    headroom, so callers naturally get paced instead of erroring out.
    """

    def __init__(self, tpm_limit: int):
        self._lock = threading.Lock()
        self._events: List[tuple] = []  # (timestamp, estimated_tokens)
        self._tpm_limit = tpm_limit

    def acquire(self, estimated_tokens: int) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - 60
                self._events = [(t, n) for t, n in self._events if t > cutoff]
                used = sum(n for _, n in self._events)
                if used + estimated_tokens <= self._tpm_limit or not self._events:
                    self._events.append((now, estimated_tokens))
                    return
                wait_for = 60 - (now - self._events[0][0]) + 0.1
            # Sleep outside the lock, in small increments, so other threads
            # can also check/update the window while this one waits.
            time.sleep(max(0.1, min(wait_for, 5.0)))

    def report_actual_usage(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Correct the most recent estimate once the real usage is known
        (from the API response), so estimation drift doesn't compound
        over a long run of many calls."""
        if actual_tokens <= 0:
            return
        with self._lock:
            for i in range(len(self._events) - 1, -1, -1):
                ts, n = self._events[i]
                if n == estimated_tokens:
                    self._events[i] = (ts, actual_tokens)
                    return


_rate_limiter = _TokenRateLimiter(settings.groq_tpm_limit)


class GroqProvider(LLMProvider):
    def __init__(self):
        from groq import Groq
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        from groq import RateLimitError

        estimated_tokens = _estimate_tokens(system_prompt, user_prompt)
        last_error: Optional[Exception] = None
        rate_limit_attempts = 0

        attempt = 0
        while attempt < MAX_RETRIES:
            _rate_limiter.acquire(estimated_tokens)
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                usage = getattr(response, "usage", None)
                if usage is not None:
                    _rate_limiter.report_actual_usage(
                        estimated_tokens, getattr(usage, "total_tokens", 0))
                return response.choices[0].message.content or ""
            except RateLimitError as e:
                # Doesn't count against the normal retry budget — this is
                # the expected/recoverable case the limiter above exists
                # for, and its own retry budget is larger.
                last_error = e
                rate_limit_attempts += 1
                if rate_limit_attempts >= MAX_RATE_LIMIT_RETRIES:
                    raise
                wait = _parse_retry_after_seconds(
                    e, default=RETRY_BACKOFF_SECONDS * rate_limit_attempts)
                time.sleep(wait)
                continue
            except Exception as e:
                # Groq (like any hosted API) occasionally drops a connection
                # mid-request ("Server disconnected without sending a
                # response"). That's transient, so retry a couple of times
                # with backoff before giving up.
                last_error = e
                attempt += 1
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        raise last_error if last_error is not None else RuntimeError(
            "Groq call failed with no captured exception")


_provider = None


def get_llm_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        _provider = GroqProvider()
    return _provider


# ---------------------------------------------------------------------
# Prompt construction for grounded, evidence-only answers
# ---------------------------------------------------------------------

ANSWER_MODE_INSTRUCTIONS = {
    "standard": "Answer clearly and concisely in a normal, neutral tone.",
    "simple": "Explain using short sentences and simple everyday words. Define any technical jargon in plain terms.",
    "detailed": "Give a thorough, well-structured explanation covering relevant context and nuance.",
    "technical": "Answer for a technically fluent audience (e.g. software/ML engineers), using precise terminology.",
}

SYSTEM_PROMPT = """You are VideoMind, an assistant that answers questions about a video using ONLY the
transcript and visual evidence provided to you.

Rules:
- Use only the supplied evidence. Never invent facts not present in the evidence.
- If the evidence is insufficient to answer confidently, say so explicitly.
- Be grounded and factual; do not speculate beyond what evidence supports.

Formatting (follow this exactly — do not use Markdown tables):
- Open with a single-sentence lead-in that frames the answer.
- If the answer has multiple distinct points (steps, ideas, items, reasons, etc.),
  present them as a numbered list. Each item: a short bold-style label, an em dash,
  then 1-2 sentences of concrete detail pulled from the evidence, ending with the
  supporting timestamp(s) in square brackets, e.g. [03:36-05:06]. Combine adjacent
  timestamps into one range if they support the same point rather than repeating
  a range across multiple items.
- If the answer is a single point rather than a list, write it as normal prose
  paragraphs with inline timestamp citations, still no table.
- If there's an overarching takeaway that ties the points together, close with one
  short paragraph stating it, without a timestamp unless one specific moment
  captures it.
- Never use a Markdown table under any circumstances, even if the evidence itself
  is tabular or comparative."""


def build_user_prompt(question: str, text_evidence: List[Dict], visual_evidence: List[Dict], answer_mode: str) -> str:
    mode_instruction = ANSWER_MODE_INSTRUCTIONS.get(
        answer_mode, ANSWER_MODE_INSTRUCTIONS["standard"])

    text_block = "\n".join(
        f"- [{e['start_formatted']}-{e['end_formatted']}] (spoken): {e['content']}"
        for e in text_evidence
    ) or "None"

    visual_block = "\n".join(
        f"- [{e['start_formatted']}-{e['end_formatted']}] (visual frame)"
        for e in visual_evidence
    ) or "None"

    return f"""Question: {question}

Answer style: {mode_instruction}

Spoken/transcript evidence:
{text_block}

Visual evidence:
{visual_block}

Answer the question using only the evidence above, following the required formatting rules (numbered list with inline timestamp citations, no tables)."""


def generate_grounded_answer(question: str, text_evidence: List[Dict], visual_evidence: List[Dict], answer_mode: str = "standard") -> str:
    provider = get_llm_provider()
    user_prompt = build_user_prompt(
        question, text_evidence, visual_evidence, answer_mode)
    return provider.generate(SYSTEM_PROMPT, user_prompt)
