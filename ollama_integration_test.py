"""Integration tests for standalone Ollama instance.

Tests the Ollama API directly (not via OpenWebUI) — covers health, model
availability, chat completions (streaming + non-streaming), and embeddings.
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

# Load .env from script directory (does not override existing env vars)
_env_files: list[Path] = [Path(__file__).resolve().parent / ".env", Path(__file__).resolve().parent / ".env.local"]
for _env_file in _env_files:
    if _env_file.is_file():
        for line in _env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "https://ollama.example.com")
TIMEOUT = (10, 120)

# Models expected to be loaded (from ollama_preload_models in defaults)
EXPECTED_MODELS = ["bge-m3", "qwen3-coder", "qwen2.5-coder"]
EMBEDDING_MODEL = "bge-m3"
CHAT_MODEL = "qwen2.5-coder:latest"


class TestResult:
    def __init__(self, name: str, passed: bool, duration: float, detail: str = ""):
        self.name = name
        self.passed = passed
        self.duration = duration
        self.detail = detail

    def __str__(self) -> str:
        status = "\033[32mPASS\033[0m" if self.passed else "\033[31mFAIL\033[0m"
        line = f"  [{status}] {self.name} ({self.duration:.2f}s)"
        if self.detail:
            line += f" — {self.detail}"
        return line


def test_health() -> TestResult:
    """GET / should return 200 'Ollama is running'."""
    t0 = time.monotonic()
    try:
        resp = requests.get(OLLAMA_URL, timeout=TIMEOUT)
        ok = resp.status_code == 200 and "Ollama is running" in resp.text
        return TestResult("health", ok, time.monotonic() - t0)
    except Exception as e:
        return TestResult("health", False, time.monotonic() - t0, str(e))


def test_list_models() -> TestResult:
    """GET /api/tags should list models; verify expected ones are present."""
    t0 = time.monotonic()
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=TIMEOUT)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        missing = [e for e in EXPECTED_MODELS if not any(e in m for m in models)]
        ok = len(missing) == 0
        detail = f"found: {models}" if ok else f"missing: {missing}, found: {models}"
        return TestResult("list_models", ok, time.monotonic() - t0, detail)
    except Exception as e:
        return TestResult("list_models", False, time.monotonic() - t0, str(e))


def test_model_info() -> TestResult:
    """POST /api/show, verify model metadata is returned."""
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/show",
            json={"name": CHAT_MODEL},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        ok = "modelfile" in data or "parameters" in data or "template" in data
        details = data.get("details", {})
        detail = f"family={details.get('family', '?')}, params={details.get('parameter_size', '?')}"
        return TestResult("model_info", ok, time.monotonic() - t0, detail)
    except Exception as e:
        return TestResult("model_info", False, time.monotonic() - t0, str(e))


def test_embeddings() -> TestResult:
    """POST /api/embed with two inputs, verify dimensions match."""
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": ["Hello world", "Integration test embedding"],
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        ok = (
            len(embeddings) == 2
            and all(isinstance(e, list) and len(e) > 0 for e in embeddings)
            and all(isinstance(v, float) for v in embeddings[0])
        )
        dims = [len(e) for e in embeddings] if embeddings else []
        detail = f"{len(embeddings)} embeddings, dims={dims}"
        return TestResult("embeddings", ok, time.monotonic() - t0, detail)
    except Exception as e:
        return TestResult("embeddings", False, time.monotonic() - t0, str(e))


def test_chat_non_streaming() -> TestResult:
    """POST /api/chat with stream=False, verify complete response."""
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
                "stream": False,
                "options": {"num_predict": 16},
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "").strip()
        ok = len(content) > 0 and data.get("done", False)
        detail = f"response: {content!r}"
        return TestResult("chat_non_streaming", ok, time.monotonic() - t0, detail)
    except Exception as e:
        return TestResult("chat_non_streaming", False, time.monotonic() - t0, str(e))


def test_chat_streaming() -> TestResult:
    """POST /api/chat with stream=True, collect tokens, verify non-empty."""
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": "Say 'hello world' and nothing else."}],
                "stream": True,
                "options": {"num_predict": 32},
            },
            stream=True,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        content = ""
        chunk_count = 0
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            msg = chunk.get("message", {}).get("content", "")
            content += msg
            if msg:
                chunk_count += 1
            if chunk.get("done"):
                break
        ok = chunk_count > 0 and len(content.strip()) > 0
        detail = f"{chunk_count} chunks, response: {content.strip()!r}"
        return TestResult("chat_streaming", ok, time.monotonic() - t0, detail)
    except Exception as e:
        return TestResult("chat_streaming", False, time.monotonic() - t0, str(e))


def main() -> None:
    print(f"Ollama integration tests — {OLLAMA_URL}\n")

    tests = [
        test_health,
        test_list_models,
        test_model_info,
        test_embeddings,
        test_chat_non_streaming,
        test_chat_streaming,
    ]

    results: list[TestResult] = []
    for test_fn in tests:
        result = test_fn()
        print(result)
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    total_time = sum(r.duration for r in results)
    print(f"\n{passed}/{total} passed in {total_time:.1f}s")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
