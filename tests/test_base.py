"""Tests for dgxarley."""

import dgxarley
from dgxarley.integration.repetition_detector import (
    RepetitionReport,
    detect_loops,
    detect_ngram_repetition,
    detect_repetition,
    detect_sentence_repetition,
)
from dgxarley.integration.streaming_repetition_guard import (
    FeedResult,
    GuardConfig,
    RepetitionGuard,
    StopReason,
)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def test_version_exists() -> None:
    """Verify that the package has a version string."""
    assert hasattr(dgxarley, "__version__")
    assert isinstance(dgxarley.__version__, str)
    assert len(dgxarley.__version__) > 0


def test_version_format() -> None:
    """Verify version follows semver pattern."""
    version = dgxarley.__version__
    parts = version.split(".")
    assert len(parts) >= 2, "Version should have at least major.minor"
    for part in parts:
        assert part.isdigit() or part[0].isdigit(), f"Version part '{part}' should start with a digit"


# ---------------------------------------------------------------------------
# Repetition Detector — N-Gram
# ---------------------------------------------------------------------------


def test_ngram_no_repetition() -> None:
    """Unique text should produce zero score."""
    text = "The quick brown fox jumps over the lazy dog near a quiet river."
    score, hits = detect_ngram_repetition(text)
    assert score == 0.0
    assert hits == []


def test_ngram_detects_repeated_phrase() -> None:
    """Repeated phrase should be detected."""
    text = "This is a test sentence. This is a test sentence. " "This is a test sentence. This is a test sentence."
    score, hits = detect_ngram_repetition(text, ns=(4,), min_count=2)
    assert score > 0.0
    assert len(hits) > 0
    assert hits[0].count >= 2


def test_ngram_short_text_returns_empty() -> None:
    """Text shorter than the n-gram size should return empty."""
    score, hits = detect_ngram_repetition("hello world", ns=(8,))
    assert score == 0.0
    assert hits == []


# ---------------------------------------------------------------------------
# Repetition Detector — Sentence
# ---------------------------------------------------------------------------


def test_sentence_no_repetition() -> None:
    """Distinct sentences should not match."""
    text = "The sun is bright. Rain falls in winter. Birds fly south."
    score, hits = detect_sentence_repetition(text)
    assert score == 0.0
    assert hits == []


def test_sentence_detects_similar_pair() -> None:
    """Near-identical sentences should be detected."""
    text = (
        "Artificial intelligence is an important field of computer science. "
        "Artificial intelligence is an important field of modern research."
    )
    score, hits = detect_sentence_repetition(text, similarity_threshold=0.7)
    assert score > 0.0
    assert len(hits) == 1
    assert hits[0].similarity >= 0.7


# ---------------------------------------------------------------------------
# Repetition Detector — Loop
# ---------------------------------------------------------------------------


def test_loop_no_repetition() -> None:
    """Non-repeating text should produce zero score."""
    text = "A completely unique paragraph with no repeating blocks whatsoever in it."
    score, hits = detect_loops(text)
    assert score == 0.0
    assert hits == []


def test_loop_detects_repeated_block() -> None:
    """Consecutively repeated block should be detected."""
    block = "This is a repeated block of text that keeps appearing. "
    text = block * 5
    score, hits = detect_loops(text, min_pattern_len=20, min_repetitions=2)
    assert score > 0.0
    assert len(hits) > 0
    assert hits[0].repetitions >= 2


# ---------------------------------------------------------------------------
# Repetition Detector — Combined
# ---------------------------------------------------------------------------


def test_detect_repetition_clean_text() -> None:
    """Clean text should have severity 'none'."""
    text = (
        "Python is a versatile language used in web development. "
        "Rust provides memory safety without garbage collection. "
        "Go excels at building concurrent networked services."
    )
    report = detect_repetition(text)
    assert isinstance(report, RepetitionReport)
    assert report.severity == "none"
    assert report.overall_score < 0.05


def test_detect_repetition_loopy_text() -> None:
    """Heavily repeated text should have high severity."""
    block = "The model keeps generating the same text over and over again. "
    text = block * 20
    report = detect_repetition(text)
    assert report.severity in ("high", "critical")
    assert report.overall_score > 0.3


def test_report_summary() -> None:
    """Summary should be a non-empty string with severity."""
    report = RepetitionReport(severity="low", overall_score=0.1)
    summary = report.summary()
    assert "[LOW]" in summary
    assert "0.10" in summary


# ---------------------------------------------------------------------------
# Streaming Repetition Guard
# ---------------------------------------------------------------------------


def test_guard_no_repetition() -> None:
    """Normal text should not trigger the guard."""
    guard = RepetitionGuard(GuardConfig(min_tokens_before_check=5, check_every_n=1))
    tokens = "The quick brown fox jumps over the lazy dog in the park".split()
    for token in tokens:
        result = guard.feed(token + " ")
        assert not result.should_stop


def test_guard_detects_ngram_flood() -> None:
    """Repeated phrase should trigger NGRAM_FLOOD."""
    guard = RepetitionGuard(
        GuardConfig(
            ngram_max_count=3,
            ngram_min_ratio=0.01,
            min_tokens_before_check=10,
            check_every_n=1,
        )
    )
    phrase = "quantum physics has many practical applications in technology "
    stopped = False
    for _ in range(30):
        result = guard.feed(phrase)
        if result.should_stop:
            assert result.reason == StopReason.NGRAM_FLOOD
            stopped = True
            break
    assert stopped, "Guard should have triggered NGRAM_FLOOD"


def test_guard_reset() -> None:
    """Reset should clear all state."""
    guard = RepetitionGuard()
    guard.feed("some tokens to build state ")
    guard.reset()
    stats = guard.get_stats()
    assert stats["tokens_seen"] == 0
    assert stats["feeds"] == 0


def test_guard_get_clean_text() -> None:
    """get_clean_text returns accumulated text."""
    guard = RepetitionGuard()
    guard.feed("Hello ")
    guard.feed("world")
    assert guard.get_clean_text() == "Hello world"


def test_guard_get_full_text() -> None:
    """get_full_text returns all accumulated text."""
    guard = RepetitionGuard()
    guard.feed("Hello ")
    guard.feed("world")
    assert guard.get_full_text() == "Hello world"


def test_feed_empty_chunk() -> None:
    """Empty chunk should be a no-op."""
    guard = RepetitionGuard()
    result = guard.feed("")
    assert not result.should_stop
    assert result.tokens_seen == 0


def test_guard_config_defaults() -> None:
    """GuardConfig defaults should be sensible."""
    cfg = GuardConfig()
    assert cfg.ngram_n == 4
    assert cfg.ngram_max_count == 5
    assert cfg.min_tokens_before_check == 40
    assert cfg.check_every_n == 3
