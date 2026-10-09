"""Tests for dgxarley.integration.gsm8k_chat."""

import json
from pathlib import Path
from typing import Any

import pytest

from dgxarley.integration.gsm8k_chat import is_correct, parse_prediction, summarize, to_number


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1,234", 1234.0), ("$18", 18.0), ("42.", 42.0), (" -3.5 ", -3.5), ("abc", None)],
)
def test_to_number(text: str, expected: float | None) -> None:
    assert to_number(text) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("so 3 + 4 = 7\n#### 7", 7.0),
        ("first #### 5 then corrected\n#### 6", 6.0),
        ("#### $1,250", 1250.0),
        ("no marker, but the answer is 12 apples", 12.0),
        ("no number at all", None),
    ],
)
def test_parse_prediction(content: str, expected: float | None) -> None:
    assert parse_prediction(content) == expected


def test_is_correct() -> None:
    assert is_correct(18.0, 18.0)
    assert is_correct(1000.5, 1000.0)
    assert not is_correct(19.0, 18.0)
    assert not is_correct(None, 18.0)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_summarize(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        {
            "id": 0,
            "gt": 7.0,
            "status": "correct",
            "exact": True,
            "start": 0.0,
            "end": 10.0,
            "latency": 10.0,
            "completion_tokens": 100,
            "content": "#### 7",
        },
        {
            "id": 1,
            "gt": 8.0,
            "status": "wrong",
            "pred": 9.0,
            "start": 0.0,
            "end": 10.0,
            "latency": 10.0,
            "completion_tokens": 100,
            "content": "#### 9",
        },
        {
            "id": 2,
            "gt": 5.0,
            "status": "truncated",
            "finish": "length",
            "start": 10.0,
            "end": 20.0,
            "latency": 10.0,
            "completion_tokens": 50,
            "content": "!" * 30,
        },
    ]
    path = tmp_path / "run.jsonl"
    _write_rows(path, rows)

    rep = summarize(str(path))

    assert rep["n"] == 3
    assert rep["status"] == {"correct": 1, "wrong": 1, "truncated": 1}
    assert rep["accuracy_scored"] == 0.5
    assert rep["exact"] == 1
    assert rep["wall"] == 20.0
    assert rep["aggregate_tok_s"] == 12.5
    assert rep["concurrent_peak_tok_s"] == 20.0
    assert rep["char_runs"] == [2]
    assert [f["id"] for f in rep["failures"]] == [1, 2]
