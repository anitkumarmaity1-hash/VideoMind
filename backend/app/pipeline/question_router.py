"""
Rule-based question classification. Deliberately simple (per spec:
"do not over-engineer") — a lightweight keyword/regex classifier is
sufficient and easy to reason about / unit test.
"""
import re
from enum import Enum


class QuestionType(str, Enum):
    SUMMARY = "summary"
    TEXT = "text_question"
    VISUAL = "visual_question"
    TEMPORAL = "temporal_question"
    GENERAL = "general_question"


_SUMMARY_PATTERNS = [
    r"\bsummar(y|ize|ise)\b",
    r"\bmain topic\b",
    r"\bwhat is this video about\b",
    r"\btl;?dr\b",
]

_VISUAL_PATTERNS = [
    r"\b(look|looks|looking)\b",
    r"\bappear(s|ing)?\b",
    r"\bshown?\b",
    r"\bobjects?\b",
    r"\bwearing\b",
    r"\bcolor\b",
    r"\bvisual(ly)?\b",
    r"\bscene\b",
]

_TEMPORAL_PATTERNS = [
    r"\b\d{1,2}:\d{2}\b",           # explicit timestamp like 04:32
    r"\bwhen\b",
    r"\bwhat happened (after|before|around)\b",
    r"\bfind where\b",
    r"\btimestamp\b",
]


def classify_question(question: str) -> QuestionType:
    q = question.lower().strip()

    for pattern in _SUMMARY_PATTERNS:
        if re.search(pattern, q):
            return QuestionType.SUMMARY

    for pattern in _TEMPORAL_PATTERNS:
        if re.search(pattern, q):
            return QuestionType.TEMPORAL

    for pattern in _VISUAL_PATTERNS:
        if re.search(pattern, q):
            return QuestionType.VISUAL

    if q.endswith("?") or re.search(r"\b(what|who|why|how)\b", q):
        return QuestionType.TEXT

    return QuestionType.GENERAL
