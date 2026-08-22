import time
from app.services.llm_service import _TokenRateLimiter, _estimate_tokens, _parse_retry_after_seconds


def test_estimate_tokens_scales_with_text_length():
    short = _estimate_tokens("hello")
    long = _estimate_tokens("hello " * 200)
    assert long > short


def test_rate_limiter_allows_calls_within_budget():
    limiter = _TokenRateLimiter(tpm_limit=1000)
    start = time.monotonic()
    limiter.acquire(200)
    limiter.acquire(200)
    limiter.acquire(200)
    elapsed = time.monotonic() - start
    # Comfortably within budget (600/1000) — should not block at all.
    assert elapsed < 0.5


def test_rate_limiter_always_admits_first_call_even_if_oversized():
    # A single request estimated above the whole budget must still be
    # allowed through (otherwise it would block forever) — the limiter
    # exists to pace *concurrent/repeated* usage, not to hard-reject any
    # individual call.
    limiter = _TokenRateLimiter(tpm_limit=100)
    start = time.monotonic()
    limiter.acquire(5000)
    elapsed = time.monotonic() - start
    assert elapsed < 0.5


def test_parse_retry_after_seconds_extracts_groq_message():
    exc = Exception(
        "Error code: 429 - {'error': {'message': 'Rate limit reached ... "
        "Please try again in 15.944999999s. Need more tokens?'}}"
    )
    wait = _parse_retry_after_seconds(exc, default=99.0)
    assert 15.9 < wait < 16.5


def test_parse_retry_after_seconds_falls_back_to_default():
    exc = Exception("some unrelated error with no timing info")
    wait = _parse_retry_after_seconds(exc, default=7.0)
    assert wait == 7.0
