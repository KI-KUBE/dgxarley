"""Integration tests for the standalone Ollama instance.

Tests the Ollama API directly (not via OpenWebUI) — covers health, model
availability, chat completions (streaming + non-streaming), and embeddings.
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

# Repo root is 2 levels above __file__ (integration/ -> dgxarley/ -> repo-root)
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Load .env from repo root (does not override existing env vars)
_env_files: list[Path] = [_REPO_ROOT / ".env", _REPO_ROOT / ".env.local"]
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

OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "https://ollama.example.com")
TIMEOUT: tuple[int, int] = (10, 120)

# Models expected to be loaded (from ollama_preload_models in defaults)
EXPECTED_MODELS: list[str] = ["bge-m3", "qwen3-coder", "qwen2.5-coder"]
EMBEDDING_MODEL: str = "bge-m3"
CHAT_MODEL: str = "qwen2.5-coder:latest"


class TestResult:
    """Result of a single integration test.

    Attributes:
        name: Short identifier for the test.
        passed: Whether the test succeeded.
        duration: Wall-clock time the test took, in seconds.
        detail: Optional human-readable detail string (error message or summary).
    """

    def __init__(self, name: str, passed: bool, duration: float, detail: str = "") -> None:
        """Initialise a TestResult.

        Args:
            name: Short identifier for the test.
            passed: Whether the test succeeded.
            duration: Wall-clock time the test took, in seconds.
            detail: Optional human-readable detail string.
        """
        self.name: str = name
        self.passed: bool = passed
        self.duration: float = duration
        self.detail: str = detail

    def __str__(self) -> str:
        """Return a colour-coded, human-readable summary line.

        Returns:
            A single line string suitable for printing to a terminal.
        """
        status = "\033[32mPASS\033[0m" if self.passed else "\033[31mFAIL\033[0m"
        line = f"  [{status}] {self.name} ({self.duration:.2f}s)"
        if self.detail:
            line += f" — {self.detail}"
        return line


def test_health() -> TestResult:
    """Verify that the Ollama server is reachable and reports itself as running.

    Sends GET / and checks for HTTP 200 with the expected response body text.

    Returns:
        A TestResult indicating whether the health check passed.
    """
    t0 = time.monotonic()
    try:
        resp = requests.get(OLLAMA_URL, timeout=TIMEOUT)
        ok = resp.status_code == 200 and "Ollama is running" in resp.text
        return TestResult("health", ok, time.monotonic() - t0)
    except Exception as e:
        return TestResult("health", False, time.monotonic() - t0, str(e))


def test_list_models() -> TestResult:
    """Verify that all expected models appear in the Ollama model list.

    Sends GET /api/tags and checks that every entry in EXPECTED_MODELS
    matches at least one returned model name (substring match).

    Returns:
        A TestResult indicating whether all expected models were found.
    """
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
    """Verify that model metadata can be retrieved for the chat model.

    Sends POST /api/show for CHAT_MODEL and checks that at least one of
    the expected metadata fields (modelfile, parameters, template) is present
    in the response.

    Returns:
        A TestResult indicating whether model metadata was returned successfully.
    """
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
    """Verify that the embedding endpoint returns well-formed vectors.

    Sends POST /api/embed with two input strings and checks that exactly two
    non-empty float vectors are returned with matching dimensions.

    Returns:
        A TestResult indicating whether the embeddings response was valid.
    """
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
    """Verify that a non-streaming chat completion returns a complete response.

    Sends POST /api/chat with stream=False and checks that the response
    contains non-empty content and the done flag is set.

    Returns:
        A TestResult indicating whether the non-streaming chat response was valid.
    """
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
    """Verify that a streaming chat completion delivers tokens incrementally.

    Sends POST /api/chat with stream=True and collects all tokens until the
    done flag is received. Checks that at least one non-empty chunk was
    delivered and that the assembled response is non-empty.

    Returns:
        A TestResult indicating whether the streaming chat response was valid.
    """
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
    """Run all Ollama integration tests and exit with an appropriate status code.

    Executes each test function in sequence, prints a colour-coded result for
    each, then prints a summary of passed/total tests and total elapsed time.
    Exits with code 0 if all tests passed, or 1 if any test failed.
    """
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
