"""OpenWebUI / SGLang integration tests.

Sampling Presets
~~~~~~~~~~~~~~~~
Recommended sampling parameters are loaded from the active model's
``recommended_sampling`` dict in ``roles/k8s_dgx/defaults/main.yml``
(the Ansible single source of truth for model profiles).

SGLang itself has no server-side default sampling flags — temperature,
top_p, top_k etc. are always per-request via the OpenAI-compatible API.
This script reads them from the Ansible defaults so every test call
automatically uses the model author's recommended values.

**Flat profiles** (most models, e.g. Qwen2.5-*):
  The ``recommended_sampling`` dict contains keys like ``temperature``,
  ``top_p``, ``top_k`` directly.  These are wrapped into a single
  preset called ``"default"``.

**Multi-mode profiles** (e.g. Qwen3.5-35B-A3B):
  The ``recommended_sampling`` dict contains sub-dicts keyed by mode:

  ============== ============================================= ====================
  Preset         When to use                                   Thinking?
  ============== ============================================= ====================
  thinking         General-purpose (default mode for Qwen3/3.5)  yes
  thinking_coding  Precise code generation (lower temperature)    yes
  non_thinking     Fast factual answers, no reasoning overhead    no (enable_thinking=False)
  non_thinking_reasoning  Complex reasoning without think blocks  no (enable_thinking=False)
  ============== ============================================= ====================

  Thinking vs. non-thinking is toggled per request:
  - API: ``extra_body={"chat_template_kwargs": {"enable_thinking": false}}``
  - Prompt shortcut: start message with ``/no_think`` or ``/think``

  The "general vs. coding vs. reasoning" distinction is purely about
  which temperature/top_p/top_k values to use — there is no API flag
  for it.  You pick the preset that matches your task type.

Architecture
~~~~~~~~~~~~
``LLMClient`` is the base class handling streaming, preset loading,
and response parsing. Two subclasses adapt to different backends:

- ``OpenWebUIClient``: Auth via API key, ``extra_body`` wrapper,
  supports ``features`` (web_search).
- ``SGLangClient``: No auth, flattens ``extra_body`` into top-level
  payload, no ``features`` support.

Usage::

    # Via OpenWebUI (default):
    python openwebui_integration_test.py thinking coding presets

    # Via direct SGLang:
    python sglang_integration_test.py thinking coding presets

    # All tests:
    python openwebui_integration_test.py all
"""

import json
import os
import time
from pathlib import Path

import requests
import yaml
from ascii_magic import AsciiArt
from PIL import Image

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


# ---------------------------------------------------------------------------
# Model profiles & sampling presets (from Ansible defaults)
# ---------------------------------------------------------------------------

_defaults_path = Path(__file__).resolve().parent / "roles" / "k8s_dgx" / "defaults" / "main.yml"
with open(_defaults_path) as _f:
    _dgx_defaults = yaml.safe_load(_f)
_MODEL_PROFILES: dict = _dgx_defaults.get("sglang_model_profiles", {})


def load_sampling_presets(model_id: str) -> dict[str, dict]:
    """Build sampling presets from a model's recommended_sampling in the Ansible defaults."""
    profile = _MODEL_PROFILES.get(model_id, {})
    raw = profile.get("recommended_sampling", {})
    if not raw:
        return {}

    # Flat preset (no sub-modes like thinking/non_thinking) — wrap it
    if any(k in raw for k in ("temperature", "top_p", "top_k")):
        raw = {"default": raw}

    presets: dict[str, dict] = {}
    for name, params in raw.items():
        preset: dict = {}
        for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "repetition_penalty"):
            if k in params:
                preset[k] = params[k]
        extra: dict = {}
        if "top_k" in params:
            extra["top_k"] = params["top_k"]
        if name.startswith("non_thinking"):
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        if extra:
            preset["extra_body"] = extra
        presets[name] = preset
    return presets


def pick_default_preset(presets: dict[str, dict]) -> str | None:
    """Pick a sensible default: prefer 'thinking', then 'default', then first available."""
    if "thinking" in presets:
        return "thinking"
    if "default" in presets:
        return "default"
    if presets:
        return list(presets)[0]
    return None


# ---------------------------------------------------------------------------
# LLMClient base class
# ---------------------------------------------------------------------------

class LLMClient:
    """Base class for streaming LLM chat completions with sampling presets."""

    def __init__(self, base_url: str, model_id: str, verbose: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.verbose = verbose
        self.presets = load_sampling_presets(model_id)
        self.default_preset = pick_default_preset(self.presets)

    # -- Subclass hooks --

    def _endpoint(self) -> str:
        """Return the full chat completions URL."""
        raise NotImplementedError

    def _headers(self) -> dict:
        """Return request headers."""
        return {"Content-Type": "application/json"}

    def _prepare_payload(self, payload: dict) -> dict:
        """Transform payload before sending (e.g. flatten extra_body for SGLang)."""
        return payload

    def _supports_features(self) -> bool:
        """Whether this backend supports OpenWebUI 'features' (web_search etc.)."""
        return False

    # -- Preset application --

    def apply_preset(
        self,
        payload: dict,
        preset: str | None = ...,
        allow_fallback: bool = True,
    ) -> dict:
        """Merge a named sampling preset into the request payload.

        If preset is ... (sentinel), uses self.default_preset.
        If preset is None, no sampling parameters are applied.
        """
        if preset is ...:
            preset = self.default_preset
        if preset is None:
            return payload
        if preset not in self.presets:
            if allow_fallback:
                available = list(self.presets) or ["(none)"]
                print(f"  [WARN] Preset '{preset}' not available for {self.model_id} "
                      f"(available: {available}), using '{self.default_preset}'")
                if self.default_preset is None:
                    return payload
                preset = self.default_preset
            else:
                raise ValueError(f"Unknown preset '{preset}'. Available: {list(self.presets)}")
        p = self.presets[preset]
        for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "repetition_penalty"):
            if k in p:
                payload[k] = p[k]
        if "extra_body" in p:
            payload.setdefault("extra_body", {})
            payload["extra_body"].update(p["extra_body"])

        # For non_thinking presets: prepend /no_think to the last user message.
        # This works both via OpenWebUI (which ignores chat_template_kwargs) and
        # direct SGLang (which honours both, belt-and-suspenders).
        if preset.startswith("non_thinking"):
            messages = payload.get("messages", [])
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str) and not content.startswith("/no_think"):
                        msg["content"] = f"/no_think\n{content}"
                    elif isinstance(content, list):
                        # Multimodal message — prepend to first text part
                        for part in content:
                            if part.get("type") == "text" and not part["text"].startswith("/no_think"):
                                part["text"] = f"/no_think\n{part['text']}"
                                break
                    break

        return payload

    # -- Streaming --

    def _niceprint_payload(self, payload: dict) -> None:
        """Print payload summary to stdout (excludes bulky fields like base64 images)."""
        display = {}
        for k, v in payload.items():
            if k == "messages":
                # Summarize messages — truncate image data
                msgs = []
                for m in v:
                    content = m.get("content", "")
                    if isinstance(content, list):
                        parts = []
                        for p in content:
                            if p.get("type") == "image_url":
                                parts.append({"type": "image_url", "image_url": "(base64 omitted)"})
                            else:
                                parts.append(p)
                        msgs.append({**m, "content": parts})
                    elif isinstance(content, str) and len(content) > 200:
                        msgs.append({**m, "content": content[:200] + "..."})
                    else:
                        msgs.append(m)
                display[k] = msgs
            else:
                display[k] = v
        print(f"\033[2m[payload] {json.dumps(display, indent=2, ensure_ascii=False)}\n[/payload]\033[0m")

    def stream_chat(self, payload: dict, print_thinking: bool = True) -> dict:
        """Stream a chat completion and print output. Returns usage stats."""
        payload = self._prepare_payload(payload)
        if self.verbose:
            self._niceprint_payload(payload)
        response = requests.post(
            self._endpoint(), headers=self._headers(), json=payload,
            stream=True, timeout=(10, 300),
        )
        response.raise_for_status()

        in_thinking = False
        usage = {}
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            decoded = raw_line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            data = decoded[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if "usage" in chunk:
                usage = chunk["usage"]
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            reasoning = delta.get("reasoning_content", "")
            content = delta.get("content", "")
            if reasoning:
                if print_thinking:
                    if not in_thinking:
                        print("\033[2m<think>", end="", flush=True)
                        in_thinking = True
                    print(reasoning, end="", flush=True)
            if content:
                if in_thinking:
                    if print_thinking:
                        print("</think>\033[0m\n", end="", flush=True)
                    in_thinking = False
                print(content, end="", flush=True)
        if in_thinking:
            print("</think>\033[0m", end="", flush=True)
        print()
        return usage

    # -- Convenience helpers --

    def chat(
        self,
        messages: list[dict],
        preset: str | None = ...,
        print_thinking: bool = True,
        stream: bool = True,
        **extra_payload,
    ) -> dict:
        """Build payload, apply preset, stream response."""
        payload = {"model": self.model_id, "messages": messages, "stream": stream, **extra_payload}
        payload = self.apply_preset(payload, preset)
        return self.stream_chat(payload, print_thinking=print_thinking)

    def explain_image(
        self,
        image: Image.Image,
        print_thinking: bool = True,
        preset: str | None = ...,
    ) -> None:
        import base64
        from io import BytesIO

        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        self.chat(
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Beschreibe dieses Bild. Was siehst du? Wenn es ein Comic ist, erkläre den Witz."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            preset=preset,
            print_thinking=print_thinking,
        )

    def get_daily_briefing(
        self,
        print_thinking: bool = True,
        preset: str | None = ...,
    ) -> None:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "Erstelle eine übersicht über die geschehnisse der nacht. so im sinne eines daily briefings"},
                {"role": "user", "content": "Erstelle mir bitte das Daily Briefing für heute."},
            ],
            "stream": True,
        }
        if self._supports_features():
            payload["features"] = {"web_search": True}
        payload = self.apply_preset(payload, preset)

        try:
            print("--- Daily Briefing ---")
            self.stream_chat(payload, print_thinking=print_thinking)
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")


# ---------------------------------------------------------------------------
# OpenWebUI client
# ---------------------------------------------------------------------------

class OpenWebUIClient(LLMClient):
    """LLM client via OpenWebUI (auth, extra_body passthrough, features support)."""

    def __init__(self, base_url: str, model_id: str, api_key: str, verbose: bool = False):
        super().__init__(base_url, model_id, verbose=verbose)
        self.api_key = api_key

    def _endpoint(self) -> str:
        return f"{self.base_url}/api/chat/completions"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _supports_features(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# SGLang direct client
# ---------------------------------------------------------------------------

class SGLangClient(LLMClient):
    """LLM client directly against SGLang (no auth, flattened extra_body)."""

    def _endpoint(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def _prepare_payload(self, payload: dict) -> dict:
        """Flatten extra_body into top-level — SGLang expects these as direct fields."""
        extra = payload.pop("extra_body", None)
        if extra:
            payload.update(extra)
        # SGLang doesn't support OpenWebUI features
        payload.pop("features", None)
        return payload


# ---------------------------------------------------------------------------
# XKCD helpers (backend-independent)
# ---------------------------------------------------------------------------

def get_random_xkcd_image_url() -> str:
    resp = requests.get("https://c.xkcd.com/random/comic/", timeout=10, allow_redirects=True)
    comic = requests.get(f"{resp.url}info.0.json", timeout=10).json()
    return comic["img"]


def get_random_xkcd_image(url: str) -> Image.Image:
    from io import BytesIO
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))


def print_ascii_representation_of_image(image: Image.Image) -> None:
    art = AsciiArt.from_pillow_image(image.convert("RGB"))
    art.to_terminal(columns=120, enhance_image=True)


# ---------------------------------------------------------------------------
# Test functions (operate on any LLMClient)
# ---------------------------------------------------------------------------

def test_thinking_mode(client: LLMClient, print_thinking: bool = True) -> None:
    print("\n=== Thinking Mode (default) ===")
    t0 = time.monotonic()
    usage = client.chat(
        messages=[{"role": "user", "content": "What is the sum of the first 20 prime numbers?"}],
        preset="thinking",
        print_thinking=print_thinking,
    )
    elapsed = time.monotonic() - t0
    print(f"  [{elapsed:.1f}s, {usage}]")


def test_non_thinking_mode(client: LLMClient) -> None:
    print("\n=== Non-Thinking Mode ===")
    t0 = time.monotonic()
    usage = client.chat(
        messages=[{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
        preset="non_thinking",
        print_thinking=True,
    )
    elapsed = time.monotonic() - t0
    print(f"  [{elapsed:.1f}s, {usage}]")


def test_thinking_coding(client: LLMClient) -> None:
    print("\n=== Thinking Mode (Coding Preset) ===")
    t0 = time.monotonic()
    usage = client.chat(
        messages=[{"role": "user", "content": "Write a Python function that checks if a string is a valid IPv4 address without using ipaddress module."}],
        preset="thinking_coding",
        print_thinking=True,
    )
    elapsed = time.monotonic() - t0
    print(f"  [{elapsed:.1f}s, {usage}]")


def test_sampling_params_passthrough(client: LLMClient) -> None:
    prompt = "Give me a single random word."
    print("\n=== Sampling Parameter Passthrough Test ===")

    for label, temp, top_p in [("low temp (0.1)", 0.1, 0.5), ("high temp (1.5)", 1.5, 1.0)]:
        print(f"\n  --- {label} ---")
        for i in range(3):
            payload = {
                "model": client.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "temperature": temp,
                "top_p": top_p,
                "max_tokens": 20,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            print(f"    Run {i+1}: ", end="")
            client.stream_chat(client._prepare_payload(payload), print_thinking=False)

    print("\n  (Low temp should produce similar/identical words, high temp should vary)")


def test_all_presets(client: LLMClient) -> None:
    prompt = "How would you approach debugging a memory leak in a Python web application?"
    print("\n" + "=" * 80)
    print("=== All Sampling Presets Comparison ===")
    print("=" * 80)

    for preset_name in client.presets:
        print(f"\n--- Preset: {preset_name} ---")
        t0 = time.monotonic()
        is_thinking = not preset_name.startswith("non_thinking")
        usage = client.chat(
            messages=[{"role": "user", "content": prompt}],
            preset=preset_name,
            print_thinking=False,
            max_tokens=512,
        )
        elapsed = time.monotonic() - t0
        print(f"  [{preset_name}: {elapsed:.1f}s, thinking={'yes' if is_thinking else 'no'}, {usage}]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def create_openwebui_client(verbose: bool = False) -> OpenWebUIClient:
    """Create an OpenWebUI client from environment variables."""
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-35B-A3B")
    owui_url = os.environ.get("OPEN_WEBUI_URL", "https://openwebui.example.com")
    api_key = os.environ.get("OPENWEBUI_API_KEY", os.environ.get("API_KEY", ""))
    if not api_key:
        raise ValueError(
            "Set OPENWEBUI_API_KEY environment variable. "
            "Generate at: OpenWebUI -> User -> Account -> API Keys"
        )
    print(f"[OpenWebUI] {owui_url} model={model_id}")
    return OpenWebUIClient(owui_url, model_id, api_key, verbose=verbose)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="OpenWebUI / SGLang integration tests")
    parser.add_argument(
        "tests",
        nargs="*",
        default=["xkcd", "briefing"],
        help="Tests to run: xkcd, xkcd_non_thinking, briefing, briefing_non_thinking, "
             "thinking, non_thinking, coding, sampling, presets, all "
             "(default: xkcd briefing)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print full payload JSON before each request",
    )
    args = parser.parse_args()

    client = create_openwebui_client(verbose=args.verbose)

    tests = set(args.tests)
    if "all" in tests:
        tests = {"xkcd", "briefing", "xkcd_non_thinking", "briefing_non_thinking",
                 "thinking", "non_thinking", "coding", "sampling", "presets"}

    if "xkcd" in tests:
        image = get_random_xkcd_image(get_random_xkcd_image_url())
        print_ascii_representation_of_image(image)
        client.explain_image(image, print_thinking=True, preset="thinking")

    if "xkcd_non_thinking" in tests:
        image = get_random_xkcd_image(get_random_xkcd_image_url())
        print_ascii_representation_of_image(image)
        client.explain_image(image, print_thinking=True, preset="non_thinking")

    if "briefing" in tests:
        print(f"\n{'*' * 80}")
        t0 = time.monotonic()
        client.get_daily_briefing(print_thinking=True, preset="thinking")
        elapsed = time.monotonic() - t0
        print(f"\n--- Daily Briefing completed in {elapsed:.1f}s ---")

    if "briefing_non_thinking" in tests:
        print(f"\n{'*' * 80}")
        t0 = time.monotonic()
        client.get_daily_briefing(
            print_thinking=True,
            preset="non_thinking"
        )
        elapsed = time.monotonic() - t0
        print(f"\n--- Daily Briefing completed in {elapsed:.1f}s ---")

    if "thinking" in tests:
        test_thinking_mode(client, print_thinking=True)

    if "non_thinking" in tests:
        test_non_thinking_mode(client)

    if "coding" in tests:
        test_thinking_coding(client)

    if "sampling" in tests:
        test_sampling_params_passthrough(client)

    if "presets" in tests:
        test_all_presets(client)


if __name__ == "__main__":
    main()
