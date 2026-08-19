import pytest
from app.utils.timestamps import format_timestamp, parse_timestamp


def test_format_seconds_under_minute():
    assert format_timestamp(5) == "00:05"


def test_format_minutes():
    assert format_timestamp(272) == "04:32"


def test_format_hours():
    assert format_timestamp(3661) == "01:01:01"


def test_format_negative_clamped_to_zero():
    assert format_timestamp(-5) == "00:00"


def test_parse_mm_ss():
    assert parse_timestamp("04:32") == 272.0


def test_parse_hh_mm_ss():
    assert parse_timestamp("01:01:01") == 3661.0


def test_parse_invalid_raises():
    with pytest.raises(ValueError):
        parse_timestamp("not-a-timestamp")


def test_roundtrip():
    for secs in [0, 5, 59, 60, 272, 3661]:
        assert parse_timestamp(format_timestamp(secs)) == float(secs)
