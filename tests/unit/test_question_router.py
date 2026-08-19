from app.pipeline.question_router import classify_question, QuestionType


def test_summary_question():
    assert classify_question("Summarize this video") == QuestionType.SUMMARY
    assert classify_question("What is the main topic?") == QuestionType.SUMMARY


def test_temporal_question_with_timestamp():
    assert classify_question("What happened around 04:32?") == QuestionType.TEMPORAL


def test_temporal_question_with_when():
    assert classify_question("When did the speaker discuss business opportunities?") == QuestionType.TEMPORAL


def test_visual_question():
    assert classify_question("What objects appear in the video?") == QuestionType.VISUAL


def test_general_text_question():
    assert classify_question("What did the speaker say about AI?") == QuestionType.TEXT


def test_general_fallback():
    assert classify_question("interesting") == QuestionType.GENERAL
