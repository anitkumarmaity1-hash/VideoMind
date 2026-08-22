"""
Rule-based question classification. Deliberately simple (per spec:
"do not over-engineer") — a lightweight keyword/regex classifier is
sufficient and easy to reason about / unit test.
"""
import re
from enum import Enum


class QuestionType(str, Enum):
    SUMMARY = "summary"
    ENUMERATION = "enumeration_question"
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

# Questions that ask for every item in a set ("the five ideas", "how many
# ways", "list all the steps") need evidence gathered from across the whole
# video, not just the handful of chunks that best match the question's own
# wording — the same retrieval breadth a summary needs, even though the
# question isn't phrased as "summarize".
_ENUMERATION_PATTERNS = [
    r"\bhow many\b",
    r"\blist (all|the|out)?\b",
    r"\benumerate\b",
    r"\ball the\b.{0,40}\b(ideas?|ways?|steps?|points?|reasons?|methods?|tips?|things?|examples?|entry points?)\b",
    r"\bwhat are the\b.{0,40}\b(ideas?|ways?|steps?|points?|reasons?|methods?|tips?|things?|examples?|entry points?)\b",
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(ideas?|ways?|steps?|points?|reasons?|methods?|tips?|things?|entry points?)\b",
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

    for pattern in _ENUMERATION_PATTERNS:
        if re.search(pattern, q):
            return QuestionType.ENUMERATION

    for pattern in _TEMPORAL_PATTERNS:
        if re.search(pattern, q):
            return QuestionType.TEMPORAL

    for pattern in _VISUAL_PATTERNS:
        if re.search(pattern, q):
            return QuestionType.VISUAL

    if q.endswith("?") or re.search(r"\b(what|who|why|how)\b", q):
        return QuestionType.TEXT

    return QuestionType.GENERAL
