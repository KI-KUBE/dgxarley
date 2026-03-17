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

    # Parallel load test (4 concurrent requests)
    python sglang_integration_test.py parallel -n 4

    # Parallel with custom prompt and 8 requests
    python sglang_integration_test.py parallel -n 8 --prompt "Erkläre Quantencomputing"
"""

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field

import aiohttp
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import requests as httplib

from openwebui_integration_test import (
    SGLangClient,
    _dgx_defaults,
    get_random_xkcd_image,
    get_random_xkcd_image_url,
    load_sampling_presets,
    pick_default_preset,
    print_ascii_representation_of_image,
    test_all_presets,
    test_non_thinking_mode,
    test_sampling_params_passthrough,
    test_thinking_coding,
    test_thinking_mode,
)

# Default model from Ansible defaults (the model currently deployed to SGLang)
_CONFIGURED_MODEL: str = _dgx_defaults.get("sglang_model", "")

# Default prompts for parallel load testing — varied to avoid prefix cache hits
PARALLEL_PROMPTS = [
    "What are the main differences between TCP and UDP? Explain with examples.",
    "Write a Python function that finds all prime numbers up to N using the Sieve of Eratosthenes.",
    "Explain the concept of quantum entanglement to a 12-year-old.",
    "What were the key causes and consequences of the French Revolution?",
    "Design a REST API for a todo-list application. Include endpoints, methods, and example payloads.",
    "Compare and contrast functional programming and object-oriented programming.",
    "Explain how a transformer neural network works, step by step.",
    "What is the significance of Gödel's incompleteness theorems?",
    "Write a bash script that monitors disk usage and sends an alert when any partition exceeds 90%.",
    "Explain the difference between symmetric and asymmetric encryption with real-world examples.",
    "How does garbage collection work in Java vs Python vs Rust?",
    "What is the CAP theorem and why does it matter for distributed systems?",
    "Explain the Monty Hall problem and why switching doors is optimal.",
    "Write a SQL query to find the top 3 customers by total spend per month for the last year.",
    "What are the pros and cons of microservices vs monolithic architecture?",
    "Explain how DNS resolution works from typing a URL to loading a webpage.",
]


def validate_model(sglang_url: str, model_id: str) -> None:
    """Query /v1/models and warn if the running model doesn't match MODEL_ID."""
    try:
        resp = httplib.get(f"{sglang_url.rstrip('/')}/v1/models", timeout=5)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        served_ids = [m.get("id", "") for m in data]
    except Exception as e:
        print(f"\033[33m⚠ Could not query /v1/models: {e}\033[0m")
        return

    if not served_ids:
        print("\033[33m⚠ /v1/models returned no models\033[0m")
        return

    if model_id in served_ids:
        print(f"✓ Model confirmed: {model_id}")
    else:
        print(f"\033[31m✗ MODEL MISMATCH — requested '{model_id}' but server serves: {served_ids}\033[0m")
        print(f"\033[31m  Hint: set MODEL_ID={served_ids[0]} or omit to use Ansible default\033[0m")
        raise SystemExit(1)


def resolve_model_id() -> str:
    """Resolve MODEL_ID: env var > Ansible defaults > error."""
    model_id = os.environ.get("MODEL_ID", "")
    if model_id:
        return model_id
    if _CONFIGURED_MODEL:
        return _CONFIGURED_MODEL
    raise ValueError("No MODEL_ID env var and no sglang_model in Ansible defaults")


def create_sglang_client(verbose: bool = False) -> SGLangClient:
    """Create a SGLang client from environment variables."""
    model_id = resolve_model_id()
    sglang_url = os.environ.get("SGLANG_URL", "")
    if not sglang_url:
        raise ValueError(
            "Set SGLANG_URL environment variable (e.g. https://sglang.dgx.example.com)"
        )
    validate_model(sglang_url, model_id)
    print(f"[SGLang direct] {sglang_url} model={model_id}")
    return SGLangClient(sglang_url, model_id, verbose=verbose)


# ---------------------------------------------------------------------------
# Parallel load test
# ---------------------------------------------------------------------------

@dataclass
class RequestStats:
    """Tracks stats for a single parallel request."""
    request_id: int
    prompt: str
    status: str = "pending"  # pending, streaming, done, error
    output: str = ""
    ttft: float = 0.0  # time to first token
    total_time: float = 0.0
    output_tokens: int = 0
    prompt_tokens: int = 0
    error: str = ""
    _start: float = field(default=0.0, repr=False)
    _first_token: bool = field(default=False, repr=False)

    @property
    def tokens_per_sec(self) -> float:
        if self.status == "done" or self.status == "error":
            t = self.total_time
        else:
            t = (time.monotonic() - self._start) if self._start else 0
        tokens = self.output_tokens if self.output_tokens > 0 else len(self.output) // 4
        if t > 0 and tokens > 0:
            return tokens / t
        return 0.0

    def output_tail(self, max_chars: int = 1200) -> str:
        """Last N chars of output for display."""
        if len(self.output) <= max_chars:
            return self.output
        return "..." + self.output[-max_chars:]


def _detect_repetition(text: str, min_output: int = 800, max_ratio: float = 0.20) -> bool:
    """Detect degenerate repetition via compression ratio.

    Repetitive text compresses extremely well.  Normal prose/code compresses
    to ~30-50% of its original size (zlib); degenerate loops compress to <20%.
    This catches all repetition patterns regardless of exact/near-match and
    block alignment — subword loops, sentence repetition, paragraph repetition.

    Only checks the tail (last 1500 chars) to detect when the model *starts*
    looping, not penalise early legitimate structure.
    """
    import zlib
    if len(text) < min_output:
        return False
    tail = text[-1500:] if len(text) > 1500 else text
    compressed = zlib.compress(tail.encode())
    ratio = len(compressed) / len(tail.encode())
    return ratio < max_ratio


async def stream_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    stats: RequestStats,
) -> None:
    """Stream a single chat completion and update stats in-place."""
    stats.status = "streaming"
    stats._start = time.monotonic()
    try:
        async with session.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=600, connect=10),
        ) as resp:
            if resp.status != 200:
                stats.status = "error"
                stats.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                stats.total_time = time.monotonic() - stats._start
                return

            async for raw_line in resp.content:
                decoded = raw_line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                data = decoded[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                u = chunk.get("usage")
                if u:
                    stats.output_tokens = u.get("completion_tokens", 0)
                    stats.prompt_tokens = u.get("prompt_tokens", 0)
                choice = (chunk.get("choices") or [None])[0]
                delta = (choice or {}).get("delta", {})
                content = delta.get("content", "")
                if content:
                    if not stats._first_token:
                        stats._first_token = True
                        stats.ttft = time.monotonic() - stats._start
                    stats.output += content
                    # Abort on degenerate repetition loops
                    if _detect_repetition(stats.output):
                        stats.status = "error"
                        stats.error = "aborted: degenerate repetition detected"
                        return

    except Exception as e:
        import traceback
        stats.status = "error"
        stats.error = f"{e}\n{traceback.format_exc()}"

    stats.total_time = time.monotonic() - stats._start
    if stats.status != "error":
        stats.status = "done"
    # Estimate tokens from output length if usage wasn't reported
    if stats.output_tokens == 0 and stats.output:
        stats.output_tokens = len(stats.output) // 4  # rough estimate


def build_live_display(all_stats: list[RequestStats]) -> Table:
    """Build a rich Table showing all parallel request states."""
    # Summary stats at the top
    done = [s for s in all_stats if s.status == "done"]
    streaming = [s for s in all_stats if s.status == "streaming"]
    errors = [s for s in all_stats if s.status == "error"]
    pending = [s for s in all_stats if s.status == "pending"]

    summary = Table.grid(padding=(0, 2))
    summary.add_row(
        f"[bold]Requests:[/] {len(all_stats)}",
        f"[green]Done:[/] {len(done)}",
        f"[yellow]Streaming:[/] {len(streaming)}",
        f"[dim]Pending:[/] {len(pending)}",
        f"[red]Errors:[/] {len(errors)}",
    )
    if done:
        agg_tokens = sum(s.output_tokens for s in done)
        agg_time = max(s.total_time for s in done) if done else 0
        avg_ttft = sum(s.ttft for s in done) / len(done)
        avg_tps = sum(s.tokens_per_sec for s in done) / len(done)
        summary.add_row(
            f"[bold]Aggregate:[/] {agg_tokens} tokens",
            f"[bold]Elapsed:[/] {agg_time:.1f}s",
            f"[bold]Avg TTFT:[/] {avg_ttft:.2f}s",
            f"[bold]Avg tok/s:[/] {avg_tps:.1f}",
            f"[bold]Total tok/s:[/] {agg_tokens / agg_time:.1f}" if agg_time > 0 else "",
        )

    # Per-request panels — size to fill terminal
    console_width = Console().width
    console_height = Console().height
    col_width = (console_width - 1) // 2
    n_rows = (len(all_stats) + 1) // 2
    # Reserve 4 lines for summary header, split remaining height across panel rows
    panel_height = max(8, (console_height - 4) // n_rows) if n_rows > 0 else 16
    # Usable lines/width inside panel (subtract borders + padding)
    inner_width = col_width - 4
    inner_lines = panel_height - 2
    # Conservative: assume avg ~45% line fill due to word wrap and short lines
    max_chars = int(inner_width * inner_lines * 0.45)

    panels = []
    for s in all_stats:
        if s.status == "pending":
            style = "dim"
            header = f"[dim]#{s.request_id} pending[/]"
            body = Text(s.prompt[:80] + "...", style="dim")
        elif s.status == "streaming":
            elapsed = time.monotonic() - s._start
            style = "yellow"
            tps = f" {s.tokens_per_sec:.1f} t/s" if s._first_token else ""
            header = f"[yellow]#{s.request_id} streaming {elapsed:.1f}s{tps}[/]"
            body = Text(s.output_tail(max_chars), style="white")
        elif s.status == "done":
            style = "green"
            header = (
                f"[green]#{s.request_id} done[/] "
                f"TTFT={s.ttft:.2f}s | {s.total_time:.1f}s | "
                f"{s.output_tokens} tok | {s.tokens_per_sec:.1f} t/s"
            )
            body = Text(s.output_tail(max_chars), style="white")
        else:
            style = "red"
            header = f"[red]#{s.request_id} ERROR[/]"
            body = Text(s.error[:max_chars], style="red")

        panels.append(Panel(body, title=header, border_style=style, height=panel_height))
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=col_width)
    grid.add_column(width=col_width)
    for i in range(0, len(panels), 2):
        left = panels[i]
        right = panels[i + 1] if i + 1 < len(panels) else ""
        grid.add_row(left, right)

    outer = Table.grid()
    outer.add_row(summary)
    outer.add_row(grid)
    return outer


def print_final_summary(all_stats: list[RequestStats], wall_time: float) -> None:
    """Print a final results table after all requests complete."""
    console = Console()
    console.print()

    table = Table(title="Parallel Request Results", show_lines=True)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("TTFT", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Prompt tok", justify="right")
    table.add_column("Output tok", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("Prompt", max_width=40)

    for s in all_stats:
        status = "[green]OK[/]" if s.status == "done" else f"[red]{s.status}[/]"
        table.add_row(
            str(s.request_id),
            status,
            f"{s.ttft:.2f}s" if s.ttft > 0 else "-",
            f"{s.total_time:.1f}s",
            str(s.prompt_tokens),
            str(s.output_tokens),
            f"{s.tokens_per_sec:.1f}" if s.tokens_per_sec > 0 else "-",
            s.prompt[:40] + ("..." if len(s.prompt) > 40 else ""),
        )

    console.print(table)

    done = [s for s in all_stats if s.status == "done"]
    if done:
        total_out = sum(s.output_tokens for s in done)
        total_prompt = sum(s.prompt_tokens for s in done)
        avg_ttft = sum(s.ttft for s in done) / len(done)
        avg_tps = sum(s.tokens_per_sec for s in done) / len(done)
        p50_ttft = sorted(s.ttft for s in done)[len(done) // 2]
        p50_tps = sorted(s.tokens_per_sec for s in done)[len(done) // 2]

        agg = Table(title="Aggregate Stats", show_lines=True)
        agg.add_column("Metric", style="bold")
        agg.add_column("Value", justify="right")
        agg.add_row("Wall time", f"{wall_time:.1f}s")
        agg.add_row("Successful requests", str(len(done)))
        agg.add_row("Failed requests", str(len(all_stats) - len(done)))
        agg.add_row("Total prompt tokens", str(total_prompt))
        agg.add_row("Total output tokens", str(total_out))
        agg.add_row("Aggregate throughput", f"{total_out / wall_time:.1f} tok/s")
        agg.add_row("Avg TTFT", f"{avg_ttft:.2f}s")
        agg.add_row("P50 TTFT", f"{p50_ttft:.2f}s")
        agg.add_row("Avg per-request tok/s", f"{avg_tps:.1f}")
        agg.add_row("P50 per-request tok/s", f"{p50_tps:.1f}")
        console.print(agg)

    # Full output log for each request
    console.print()
    for s in all_stats:
        if s.output:
            console.print(Panel(
                Text(s.output),
                title=f"[bold]#{s.request_id} full output[/] ({s.status})",
                border_style="green" if s.status == "done" else "red",
                expand=True,
            ))
        elif s.error:
            console.print(Panel(
                Text(s.error, style="red"),
                title=f"[bold red]#{s.request_id} error[/]",
                border_style="red",
                expand=True,
            ))


async def run_parallel_test(
    n: int,
    sglang_url: str,
    model_id: str,
    preset: str | None,
    prompts: list[str],
    max_tokens: int|None,
) -> None:
    """Run n parallel streaming requests with live display."""
    # Build payload template
    presets = load_sampling_presets(model_id)
    default_preset = pick_default_preset(presets)
    if preset is None:
        preset = default_preset

    all_stats: list[RequestStats] = []
    for i in range(n):
        prompt = prompts[i % len(prompts)]
        all_stats.append(RequestStats(request_id=i + 1, prompt=prompt))

    # Build payloads
    payloads = []
    for s in all_stats:
        messages = [{"role": "user", "content": s.prompt}]
        payload: dict = {
            "model": model_id,
            "messages": messages,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # Apply preset sampling params
        if preset and preset in presets:
            p = presets[preset]
            for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "repetition_penalty"):
                if k in p:
                    payload[k] = p[k]
            extra = p.get("extra_body", {})
            payload.update(extra)
            # Non-thinking prefix
            if preset.startswith("non_thinking"):
                messages[0]["content"] = f"/no_think\n{messages[0]['content']}"
        payloads.append(payload)

    url = f"{sglang_url.rstrip('/')}/v1/chat/completions"
    console = Console()
    console.print(f"[bold]Starting {n} parallel requests to {url}[/]")
    console.print(f"[dim]Model: {model_id} | Preset: {preset} | Max tokens: {max_tokens}[/]\n")

    wall_start = time.monotonic()

    async with aiohttp.ClientSession() as session:
        tasks = [
            stream_request(session, url, payloads[i], all_stats[i])
            for i in range(n)
        ]

        # Live display updates while requests stream
        with Live(build_live_display(all_stats), console=console, refresh_per_second=4) as live:
            # Start all tasks
            pending = set()
            for t in tasks:
                pending.add(asyncio.ensure_future(t))

            while pending:
                done_tasks, pending = await asyncio.wait(
                    pending, timeout=0.25, return_when=asyncio.FIRST_COMPLETED
                )
                live.update(build_live_display(all_stats))

            # Final update
            live.update(build_live_display(all_stats))

    wall_time = time.monotonic() - wall_start
    print_final_summary(all_stats, wall_time)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Direct SGLang integration tests")

    # Default: run named tests
    parser.add_argument(
        "tests",
        nargs="*",
        default=["thinking", "non_thinking"],
        help="Tests to run: xkcd, xkcd_non_thinking, briefing, briefing_non_thinking, "
             "thinking, non_thinking, coding, sampling, presets, parallel, all "
             "(default: thinking non_thinking)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show thinking/reasoning tokens and full payloads",
    )
    parser.add_argument(
        "-n", "--num-requests",
        type=int,
        default=4,
        help="Number of parallel requests (for 'parallel' test, default: 4)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Sampling preset for parallel test (default: model's default)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for all parallel requests (default: varied prompts)",
    )
    parser.add_argument(
        "--max-tokens",
        type=lambda v: None if v.lower() == "none" else int(v),
        default=8192,
        help="Max output tokens per request for parallel test (default: 1024, 'none' for model default)",
    )
    args = parser.parse_args()

    verbose = args.verbose
    tests = set(args.tests)
    if "all" in tests:
        tests = {"xkcd", "xkcd_non_thinking", "briefing", "briefing_non_thinking",
                 "thinking", "non_thinking", "coding", "sampling", "presets"}

    # Handle parallel test separately (async)
    if "parallel" in tests:
        tests.discard("parallel")
        model_id = resolve_model_id()
        sglang_url = os.environ.get("SGLANG_URL", "")
        if not sglang_url:
            raise ValueError("Set SGLANG_URL environment variable")
        validate_model(sglang_url, model_id)
        prompts = [args.prompt] * args.num_requests if args.prompt else random.sample(PARALLEL_PROMPTS, len(PARALLEL_PROMPTS))
        asyncio.run(run_parallel_test(
            n=args.num_requests,
            sglang_url=sglang_url,
            model_id=model_id,
            preset=args.preset,
            prompts=prompts,
            max_tokens=args.max_tokens,
        ))

    # Sequential tests
    if tests:
        client = create_sglang_client(verbose=verbose)

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
