"""Direct SGLang integration tests (bypasses OpenWebUI).

Uses the same LLMClient base class and test functions from
openwebui_integration_test.py, but with SGLangClient which:
- Hits the SGLang OpenAI-compatible API directly (/v1/chat/completions)
- No auth header required
- Flattens extra_body into top-level payload (SGLang expects top_k,
  chat_template_kwargs etc. as direct fields, not nested in extra_body)
- No OpenWebUI features (web_search etc.)

Usage::

    # Requires SGLANG_URL env var
    SGLANG_URL=https://sglang.dgx.example.com python sglang_integration_test.py all

    # Specific tests
    python sglang_integration_test.py thinking non_thinking coding presets
"""

import os
import time

from openwebui_integration_test import (
    SGLangClient,
    get_random_xkcd_image,
    get_random_xkcd_image_url,
    print_ascii_representation_of_image,
    test_all_presets,
    test_non_thinking_mode,
    test_sampling_params_passthrough,
    test_thinking_coding,
    test_thinking_mode,
)


def create_sglang_client(verbose: bool = False) -> SGLangClient:
    """Create a SGLang client from environment variables."""
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-35B-A3B")
    sglang_url = os.environ.get("SGLANG_URL", "")
    if not sglang_url:
        raise ValueError(
            "Set SGLANG_URL environment variable (e.g. https://sglang.dgx.example.com)"
        )
    print(f"[SGLang direct] {sglang_url} model={model_id}")
    return SGLangClient(sglang_url, model_id, verbose=verbose)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Direct SGLang integration tests")
    parser.add_argument(
        "tests",
        nargs="*",
        default=["thinking", "non_thinking"],
        help="Tests to run: xkcd, xkcd_non_thinking, briefing, briefing_non_thinking, "
             "thinking, non_thinking, coding, sampling, presets, all "
             "(default: thinking non_thinking)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show thinking/reasoning tokens and full payloads",
    )
    args = parser.parse_args()

    verbose = args.verbose
    client = create_sglang_client(verbose=verbose)

    tests = set(args.tests)
    if "all" in tests:
        tests = {"xkcd", "xkcd_non_thinking", "briefing", "briefing_non_thinking",
                 "thinking", "non_thinking", "coding", "sampling", "presets"}

    if "xkcd" in tests:
        image = get_random_xkcd_image(get_random_xkcd_image_url())
        print_ascii_representation_of_image(image)
        client.explain_image(image, print_thinking=verbose)

    if "xkcd_non_thinking" in tests:
        image = get_random_xkcd_image(get_random_xkcd_image_url())
        print_ascii_representation_of_image(image)
        client.explain_image(image, print_thinking=verbose, preset="non_thinking")

    if "briefing" in tests:
        print(f"\n{'*' * 80}")
        t0 = time.monotonic()
        client.get_daily_briefing(print_thinking=verbose, preset="thinking")
        elapsed = time.monotonic() - t0
        print(f"\n--- Daily Briefing completed in {elapsed:.1f}s ---")

    if "briefing_non_thinking" in tests:
        print(f"\n{'*' * 80}")
        t0 = time.monotonic()
        client.get_daily_briefing(print_thinking=verbose, preset="non_thinking")
        elapsed = time.monotonic() - t0
        print(f"\n--- Daily Briefing (non-thinking) completed in {elapsed:.1f}s ---")

    if "thinking" in tests:
        test_thinking_mode(client, print_thinking=verbose)

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
