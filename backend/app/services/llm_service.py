"""
LLM provider abstraction. Groq is the default/active provider.
Adding OpenAI / Anthropic / Gemini / Ollama later means implementing
LLMProvider and registering it in get_llm_provider() — no changes needed
elsewhere in the app.
"""
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from app.config import settings

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        ...


class GroqProvider(LLMProvider):
    def __init__(self):
        from groq import Groq
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        # Groq (like any hosted API) occasionally drops a connection mid-request
        # ("Server disconnected without sending a response"). That's transient,
        # so retry a couple of times with backoff before giving up.
        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
        # Unreachable unless MAX_RETRIES is 0, but keeps the type checker
        # (and anyone editing MAX_RETRIES later) honest about what this raises.
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
transcript and visual evidence provided to you. Rules:
- Use only the supplied evidence. Never invent facts not present in the evidence.
- If the evidence is insufficient to answer confidently, say so explicitly.
- Always cite the relevant timestamp(s) in your answer, e.g. [05:20-06:05].
- Clearly separate spoken/transcript evidence from visual evidence when both are used.
- Be grounded and factual; do not speculate beyond what evidence supports."""


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

Answer the question using only the evidence above, citing timestamps."""


def generate_grounded_answer(question: str, text_evidence: List[Dict], visual_evidence: List[Dict], answer_mode: str = "standard") -> str:
    provider = get_llm_provider()
    user_prompt = build_user_prompt(
        question, text_evidence, visual_evidence, answer_mode)
    return provider.generate(SYSTEM_PROMPT, user_prompt)
