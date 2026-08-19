import pytest
from app.pipeline.chunking import create_temporal_chunks, attach_frame_timestamps


def test_basic_chunking_no_overlap():
    transcript = [{"start": 1.0, "end": 3.0, "text": "hello"}]
    chunks = create_temporal_chunks(transcript, video_duration=20, chunk_size=10, overlap=0)
    assert len(chunks) == 2
    assert chunks[0]["start_time"] == 0
    assert chunks[0]["end_time"] == 10
    assert chunks[1]["start_time"] == 10
    assert chunks[1]["end_time"] == 20


def test_chunking_with_overlap_step():
    chunks = create_temporal_chunks([], video_duration=30, chunk_size=10, overlap=2)
    starts = [c["start_time"] for c in chunks]
    # step = 8, so starts should be 0, 8, 16, 24
    assert starts == [0, 8, 16, 24]


def test_transcript_assigned_to_correct_chunk():
    transcript = [
        {"start": 2.0, "end": 4.0, "text": "first"},
        {"start": 12.0, "end": 14.0, "text": "second"},
    ]
    chunks = create_temporal_chunks(transcript, video_duration=20, chunk_size=10, overlap=0)
    assert chunks[0]["transcript"] == "first"
    assert chunks[1]["transcript"] == "second"


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        create_temporal_chunks([], video_duration=20, chunk_size=10, overlap=10)


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        create_temporal_chunks([], video_duration=20, chunk_size=0, overlap=0)


def test_attach_frame_timestamps():
    chunks = [{"chunk_id": 0, "start_time": 0, "end_time": 10, "transcript": ""}]
    frame_ts = [1.0, 5.0, 11.0]
    result = attach_frame_timestamps(chunks, frame_ts)
    assert result[0]["frame_timestamps"] == [1.0, 5.0]
