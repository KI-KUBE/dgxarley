"""GSM8K accuracy/throughput harness for reasoning models behind an OpenAI-compatible chat API.

The stock sglang GSM8K harness needs Python 3.12 (`urlopen(cafile=)`) and drives the raw completion
endpoint, which a reasoning model answers with its trace and ~1 token; this one uses /v1/chat/completions,
asks for a `#### <number>` last line, scores numerically and counts budget exhaustion (empty content or
finish_reason length) as truncation, never as a wrong answer. Requests stream by default so `analyze` can
report windowed throughput (`sustained_10s_tok_s`, `full_load_tok_s`) next to the start/end based peak; with
`continuous_usage_stats` (SGLang >= 0.5.21) the timeline counts tokens, otherwise it falls back to characters.

Usage:
  kubectl --context=ht@dgxarley -n sglang port-forward svc/sglang 38080:8000 &
  sglang-gsm8k fetch --data /tmp/gsm8k/test.jsonl
  sglang-gsm8k run --data /tmp/gsm8k/test.jsonl --out /tmp/gsm8k/run1.jsonl \\
      --model vroomfondel/GLM-5.3-Flash-NVFP4-W4A4 --minutes 15 --concurrency 8
  sglang-gsm8k analyze /tmp/gsm8k/run1.jsonl [/tmp/gsm8k/run2.jsonl]
"""

import argparse
import asyncio
import concurrent.futures
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

GSM8K_TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
PROMPT_SUFFIX = (
    "\n\nSolve step by step. The final line of your answer must be exactly `#### <number>` (just the number, no units)."
)
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
FINAL_RE = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
CHAR_RUN_RE = re.compile(r"(.)\1{19,}")
TIMELINE_BIN_S = 0.5
SUSTAINED_WINDOW_S = 10.0
FULL_LOAD_FRAC = 0.9
NON_MATH_NONASCII_RE = re.compile(r"[^\x00-\x7f‘-”–— °×÷€£≈→≤≥−½²]")


def to_number(text: str) -> float | None:
    text = text.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(text)
    except ValueError:
        return None


def parse_prediction(content: str) -> float | None:
    final = FINAL_RE.findall(content)
    if final:
        return to_number(final[-1])
    numbers = NUMBER_RE.findall(content)
    return to_number(numbers[-1]) if numbers else None


def is_correct(pred: float | None, gt: float) -> bool:
    if pred is None:
        return False
    return abs(pred - gt) <= 1e-6 or abs(pred - gt) <= 1e-2 * abs(gt)


def fetch(args: argparse.Namespace) -> None:
    target = Path(args.data)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.url, timeout=120) as resp:
        target.write_bytes(resp.read())
    print(f"wrote {target} ({sum(1 for _ in target.open())} questions)")


def chat_call(
    url: str,
    model: str,
    question: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    t0: float = 0.0,
    stream: bool = True,
) -> tuple[dict[str, Any], list[list[float]]]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": question + PROMPT_SUFFIX}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stream:
        body.update(stream=True, stream_options={"include_usage": True, "continuous_usage_stats": True})
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if not stream:
            data: dict[str, Any] = json.loads(resp.read())
            return data, []
        content: list[str] = []
        reasoning: list[str] = []
        finish: str | None = None
        usage: dict[str, Any] = {}
        char_bins: dict[float, int] = {}
        token_bins: dict[float, int] = {}
        last_tokens = 0
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if "error" in chunk:
                raise RuntimeError(f"stream error: {chunk['error']}")
            key = math.floor((time.time() - t0) / TIMELINE_BIN_S) * TIMELINE_BIN_S
            if chunk.get("usage"):
                usage = chunk["usage"]
                tokens = usage.get("completion_tokens")
                if isinstance(tokens, int) and tokens > last_tokens:
                    token_bins[key] = token_bins.get(key, 0) + tokens - last_tokens
                    last_tokens = tokens
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                think = delta.get("reasoning_content") or delta.get("reasoning") or ""
                content.append(text)
                reasoning.append(think)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                if text or think:
                    char_bins[key] = char_bins.get(key, 0) + len(text) + len(think)
    # One usage chunk at the end only means the server ignored continuous_usage_stats.
    exact = len(token_bins) > 1 or (bool(token_bins) and not char_bins)
    bins = token_bins if exact else char_bins
    message = {"content": "".join(content), "reasoning_content": "".join(reasoning)}
    data = {
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": usage,
        "timeline_unit": "tokens" if exact else "chars",
    }
    return data, [[t, float(c)] for t, c in sorted(bins.items())]


async def run(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.data).open() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    t0 = time.time()
    dispatch_stop = t0 + args.minutes * 60 * args.dispatch_stop_frac
    hard_stop = t0 + args.minutes * 60
    out = Path(args.out).open("w")
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(len(rows)):
        queue.put_nowait(i)

    async def worker() -> None:
        while time.time() < dispatch_stop:
            try:
                i = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            row = rows[i]
            gt = to_number(row["answer"].split("####")[-1])
            rec: dict[str, Any] = {"id": i, "gt": gt, "start": time.time() - t0}
            try:
                data, timeline = await asyncio.to_thread(
                    chat_call,
                    args.url,
                    args.model,
                    row["question"],
                    args.max_tokens,
                    args.temperature,
                    args.timeout,
                    t0,
                    not args.no_stream,
                )
                if timeline:
                    rec["tl"] = timeline
                    rec["tl_unit"] = data.get("timeline_unit", "chars")
                choice = data["choices"][0]
                msg = choice["message"]
                content = msg.get("content") or ""
                usage = data.get("usage") or {}
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                rec.update(
                    content=content,
                    reasoning_len=len(reasoning),
                    finish=choice["finish_reason"],
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                )
                if not content.strip() or choice["finish_reason"] == "length":
                    rec["status"] = "truncated"
                elif gt is None:
                    rec["status"] = "error"
                    rec["err"] = "unparseable ground truth"
                else:
                    pred = parse_prediction(content)
                    rec["pred"] = pred
                    rec["status"] = "correct" if is_correct(pred, gt) else "wrong"
                    rec["exact"] = pred is not None and pred == gt
            except Exception as exc:  # noqa: BLE001 - every failure mode is one JSONL row
                rec.update(status="error", err=repr(exc))
            rec["end"] = time.time() - t0
            rec["latency"] = rec["end"] - rec["start"]
            out.write(json.dumps(rec) + "\n")
            out.flush()

    # Default executor is min(32, cpu_count + 4) threads and would silently cap in-flight requests.
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency + 2, thread_name_prefix="gsm8k")
    )
    workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
    _done, pending = await asyncio.wait(workers, timeout=hard_stop - time.time())
    for task in pending:
        task.cancel()
    out.close()
    print(json.dumps({"wall": round(time.time() - t0, 1), "unfinished_workers": len(pending), "out": args.out}))


def windowed_throughput(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens: dict[float, float] = {}
    units: set[str] = set()
    for r in rows:
        tl: list[list[float]] = r.get("tl") or []
        total = sum(c for _, c in tl)
        if not total or not r.get("completion_tokens"):
            continue
        unit = r.get("tl_unit", "chars")
        units.add(unit)
        scale = 1.0 if unit == "tokens" else r["completion_tokens"] / total
        for t, c in tl:
            tokens[t] = tokens.get(t, 0.0) + c * scale
    if not tokens:
        return {
            "sustained_10s_tok_s": None,
            "full_load_tok_s": None,
            "full_load_seconds": 0.0,
            "full_load_min_active": 0,
            "timeline_unit": None,
        }
    n_bins = int(max(tokens) / TIMELINE_BIN_S) + 1
    series = [tokens.get(i * TIMELINE_BIN_S, 0.0) for i in range(n_bins)]
    window = max(1, min(n_bins, int(SUSTAINED_WINDOW_S / TIMELINE_BIN_S)))
    best = running = sum(series[:window])
    for i in range(window, n_bins):
        running += series[i] - series[i - window]
        best = max(best, running)
    active = [sum(1 for r in rows if r["start"] <= (i + 0.5) * TIMELINE_BIN_S < r["end"]) for i in range(n_bins)]
    threshold = math.ceil(FULL_LOAD_FRAC * max(active))
    full = [i for i in range(n_bins) if active[i] >= threshold]
    full_seconds = len(full) * TIMELINE_BIN_S
    return {
        "sustained_10s_tok_s": round(best / (window * TIMELINE_BIN_S), 1),
        "full_load_tok_s": round(sum(series[i] for i in full) / full_seconds, 1) if full_seconds else None,
        "full_load_seconds": full_seconds,
        "full_load_min_active": threshold,
        "timeline_unit": units.pop() if len(units) == 1 else "mixed",
    }


def summarize(path: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(path).open() if line.strip()]
    status: dict[str, int] = {}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
    scored = status.get("correct", 0) + status.get("wrong", 0)
    latencies = sorted(r["latency"] for r in rows)

    def pct(p: float) -> float:
        return latencies[min(len(latencies) - 1, int(p * len(latencies)))] if latencies else 0.0

    completion_tokens = sum(r.get("completion_tokens") or 0 for r in rows)
    wall = max((r["end"] for r in rows), default=0.0)
    events: list[tuple[float, float]] = []
    for r in rows:
        if r.get("completion_tokens") and r["latency"] > 0:
            rate = r["completion_tokens"] / r["latency"]
            events += [(r["start"], rate), (r["end"], -rate)]
    events.sort()
    current = peak = 0.0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    char_runs = [r["id"] for r in rows if CHAR_RUN_RE.search(r.get("content") or "")]
    odd_nonascii = [r["id"] for r in rows if NON_MATH_NONASCII_RE.search(r.get("content") or "")]
    return {
        "file": path,
        "n": len(rows),
        "status": status,
        "accuracy_scored": round(status.get("correct", 0) / scored, 4) if scored else None,
        "exact": sum(1 for r in rows if r.get("exact")),
        "latency_p50": round(pct(0.5), 1),
        "latency_p95": round(pct(0.95), 1),
        "latency_max": round(latencies[-1], 1) if latencies else 0.0,
        "completion_tokens": completion_tokens,
        "wall": round(wall, 1),
        "aggregate_tok_s": round(completion_tokens / wall, 1) if wall else 0.0,
        "concurrent_peak_tok_s": round(peak, 1),
        **windowed_throughput(rows),
        "avg_completion_tokens": round(completion_tokens / len(rows), 1) if rows else 0.0,
        "char_runs": char_runs,
        "odd_nonascii": odd_nonascii,
        "failures": [
            {
                "id": r["id"],
                "status": r["status"],
                "gt": r["gt"],
                "pred": r.get("pred"),
                "finish": r.get("finish"),
                "tail": (r.get("content") or "")[-120:],
            }
            for r in rows
            if r["status"] != "correct"
        ][:12],
    }


def analyze(args: argparse.Namespace) -> None:
    reports = [summarize(p) for p in args.files]
    for rep in reports:
        print(json.dumps(rep, indent=1, ensure_ascii=False))
    if len(reports) == 2:
        a = {json.loads(l)["id"]: json.loads(l) for l in Path(args.files[0]).open() if l.strip()}
        b = {json.loads(l)["id"]: json.loads(l) for l in Path(args.files[1]).open() if l.strip()}
        shared = sorted(set(a) & set(b))
        ca = sum(1 for i in shared if a[i]["status"] == "correct")
        cb = sum(1 for i in shared if b[i]["status"] == "correct")
        both_wrong = [i for i in shared if a[i]["status"] == "wrong" and b[i]["status"] == "wrong"]
        print(json.dumps({"shared_ids": len(shared), "correct_a": ca, "correct_b": cb, "wrong_in_both": both_wrong}))


def cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="download the GSM8K test split as JSONL")
    p_fetch.add_argument("--data", required=True, help="target path for test.jsonl")
    p_fetch.add_argument("--url", default=GSM8K_TEST_URL)
    p_fetch.set_defaults(func=fetch)

    p_run = sub.add_parser("run", help="run questions against the chat API for a wall-clock budget")
    p_run.add_argument("--data", required=True, help="GSM8K test.jsonl (question/answer rows)")
    p_run.add_argument("--out", required=True, help="per-request results JSONL")
    p_run.add_argument("--url", default="http://127.0.0.1:38080/v1/chat/completions")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--minutes", type=float, default=15.0, help="hard wall-clock budget")
    p_run.add_argument("--dispatch-stop-frac", type=float, default=0.9, help="stop dispatching at this fraction")
    p_run.add_argument("--concurrency", type=int, default=8)
    p_run.add_argument("--max-tokens", type=int, default=8192)
    p_run.add_argument("--temperature", type=float, default=0.0)
    p_run.add_argument("--timeout", type=int, default=600, help="per-request HTTP timeout in seconds")
    p_run.add_argument("--limit", type=int, default=0, help="only the first N questions (0 = all)")
    p_run.add_argument("--no-stream", action="store_true", help="plain responses, no windowed throughput")
    p_run.set_defaults(func=lambda a: asyncio.run(run(a)))

    p_an = sub.add_parser("analyze", help="summarize one results JSONL, or compare two")
    p_an.add_argument("files", nargs="+")
    p_an.set_defaults(func=analyze)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


def main() -> None:
    sys.exit(cli(sys.argv[1:]))


if __name__ == "__main__":
    main()
