"""Tests for dgxarley.integration.gsm8k_chat."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from dgxarley.integration.gsm8k_chat import chat_call, is_correct, parse_prediction, summarize, to_number


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


def test_windowed_throughput_ignores_tail(tmp_path: Path) -> None:
    # Two requests share 0-10 s at 10 tok/s each; the second runs alone 10-20 s at 40 tok/s.
    rows: list[dict[str, Any]] = [
        {
            "id": 0,
            "gt": 1.0,
            "status": "correct",
            "start": 0.0,
            "end": 10.0,
            "latency": 10.0,
            "completion_tokens": 100,
            "content": "",
            "tl": [[i * 0.5, 5.0] for i in range(20)],
        },
        {
            "id": 1,
            "gt": 1.0,
            "status": "correct",
            "start": 0.0,
            "end": 20.0,
            "latency": 20.0,
            "completion_tokens": 500,
            "content": "",
            "tl": [[i * 0.5, 5.0] for i in range(20)] + [[10.0 + i * 0.5, 20.0] for i in range(20)],
        },
    ]
    path = tmp_path / "run.jsonl"
    _write_rows(path, rows)

    rep = summarize(str(path))

    assert rep["concurrent_peak_tok_s"] == 35.0
    assert rep["full_load_tok_s"] == 20.0
    assert rep["full_load_seconds"] == 10.0
    assert rep["full_load_min_active"] == 2
    assert rep["sustained_10s_tok_s"] == 40.0


def test_summarize_without_timeline(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write_rows(path, [{"id": 0, "gt": 1.0, "status": "correct", "start": 0.0, "end": 1.0, "latency": 1.0}])

    rep = summarize(str(path))

    assert rep["full_load_tok_s"] is None
    assert rep["sustained_10s_tok_s"] is None


def test_chat_call_stream() -> None:
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "think"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "#### 7"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}},
    ]
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    seen: dict[str, Any] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(sse.encode())

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    try:
        data, timeline = chat_call(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", "m", "q", 16, 0.0, 10)
    finally:
        thread.join()
        server.server_close()

    assert seen["body"]["stream"] is True
    assert seen["body"]["stream_options"] == {"include_usage": True}
    choice = data["choices"][0]
    assert choice["message"] == {"content": "#### 7", "reasoning_content": "think"}
    assert choice["finish_reason"] == "stop"
    assert data["usage"]["completion_tokens"] == 4
    assert sum(c for _, c in timeline) == 11.0
