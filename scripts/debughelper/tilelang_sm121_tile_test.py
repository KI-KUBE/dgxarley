"""Standalone GB10/SM121 test of SGLang's TileLang DSA sparse-attention kernels with retuned tiles.

Each (kernel, tile, heads) combination runs in its own subprocess so a failed compile/launch cannot poison the rest.
"""

import json
import re
import subprocess
import sys
import time
from typing import Any

TOPK = 2048
BATCH = 4
PAGES = 8192
D_V = 512
D_TAIL_V2 = 64
SHAPES = {"decode": 1, "verify": 8}

# (kernel, block_I, num_stages, threads); v2 has only block_I (threads fixed at 384, no num_stages)
CONFIGS = [
    ("v1", 64, 2, 256),
    ("v1", 32, 1, 128),
    ("v1", 16, 1, 128),
    ("v1", 32, 1, 256),
    ("v1", 32, 2, 128),
    ("v2", 64, 0, 384),
    ("v2", 32, 0, 384),
]
HEADS = [16, 32]


def reference(q: Any, kv: Any, idx: Any, scale: float, d_v: int) -> Any:
    import torch

    qf = q.float()
    kvf = kv[:, 0, :].float()
    safe = idx[:, 0, :].clamp(min=0).long()
    g = kvf[safe]
    s = torch.einsum("shd,skd->shk", qf, g) * scale
    s = s.masked_fill((idx[:, 0, :] < 0).unsqueeze(1), float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("shk,skd->shd", p, g[..., :d_v])


def smem_from_source(src: str) -> list[str]:
    hits = re.findall(r"[^\n]*(?:MaxDynamicSharedMemorySize|dyn_shmem|dynamic_shared)[^\n]*", src)
    return [h.strip() for h in hits[:6]]


def run_one(kern: str, block_i: int, stages: int, threads: int, heads: int) -> dict[str, Any]:
    import torch
    import tilelang  # type: ignore[import-not-found]
    from sglang.kernels.ops.attention.dsa import tilelang_kernel as tk

    out: dict[str, Any] = {"kernel": kern, "block_I": block_i, "num_stages": stages, "threads": threads, "H": heads}
    tail = D_TAIL_V2 if kern == "v2" else 0
    dim = D_V + tail
    scale = dim**-0.5
    t0 = time.time()
    try:
        if kern == "v1":
            fn = tk.sparse_attention_fwd_kernel_v1(
                heads, D_V, 0, TOPK, sm_scale=scale, block_I=block_i, num_stages=stages, threads=threads
            )
        else:
            fn = tk.sparse_attention_fwd_kernel_v2(heads, D_V, tail, TOPK, sm_scale=scale, block_I=block_i)
    except Exception as e:  # noqa: BLE001
        out["compile_error"] = f"{type(e).__name__}: {str(e)[-1500:]}"
        return out
    out["compile_s"] = round(time.time() - t0, 1)
    try:
        out["smem_hits"] = smem_from_source(fn.get_kernel_source())
    except Exception as e:  # noqa: BLE001
        out["smem_hits_error"] = repr(e)

    torch.manual_seed(0)
    dev = "cuda"
    kv_full = (torch.randn(PAGES + 1, 1, dim, device=dev, dtype=torch.float32) * 0.5).bfloat16()
    kv = kv_full[1:]
    out["shapes"] = {}
    for name, ntpr in SHAPES.items():
        rec: dict[str, Any] = {}
        out["shapes"][name] = rec
        s = BATCH * ntpr
        q = (torch.randn(s, heads, dim, device=dev, dtype=torch.float32) * 0.5).bfloat16()
        idx = torch.stack([torch.randperm(PAGES, device=dev)[:TOPK] for _ in range(s)]).int().unsqueeze(1)
        idx[0, 0, 1500:] = -1
        idx[s - 1, 0, 1900:] = -1
        try:
            o = fn(q.unsqueeze(0), kv.unsqueeze(0), idx.unsqueeze(0))
            torch.cuda.synchronize()
            o = o[0] if o.dim() == 4 else o
            ref = reference(q, kv, idx, scale, D_V)
            diff = (o.float() - ref).abs()
            rec["max_abs"] = float(diff.max())
            rec["rel_err"] = float(diff.norm() / ref.norm())
            rec["nan"] = bool(torch.isnan(o).any())
            for _ in range(5):
                fn(q.unsqueeze(0), kv.unsqueeze(0), idx.unsqueeze(0))
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            a.record()
            for _ in range(20):
                fn(q.unsqueeze(0), kv.unsqueeze(0), idx.unsqueeze(0))
            b.record()
            torch.cuda.synchronize()
            rec["ms"] = round(a.elapsed_time(b) / 20, 4)
        except Exception as e:  # noqa: BLE001
            rec["launch_error"] = f"{type(e).__name__}: {str(e)[-1500:]}"
            break
    return out


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        kern, bi, st, th, h = sys.argv[2], *map(int, sys.argv[3:7])
        res = run_one(kern, bi, st, th, h)
        print("RESULT_JSON " + json.dumps(res))
        return

    import torch

    p = torch.cuda.get_device_properties(0)
    import tilelang

    print(
        f"tilelang {tilelang.__version__}  device {p.name} sm_{p.major}{p.minor}  smem_optin {getattr(p, "shared_memory_per_block_optin")}"
    )
    results = []
    for h in HEADS:
        for kern, bi, st, th in CONFIGS:
            tag = f"{kern} block_I={bi} stages={st} threads={th} H={h}"
            print(f"\n=== {tag}", flush=True)
            try:
                r = subprocess.run(
                    [sys.executable, __file__, "--one", kern, str(bi), str(st), str(th), str(h)],
                    capture_output=True,
                    text=True,
                    timeout=2400,
                )
            except subprocess.TimeoutExpired:
                print("TIMEOUT", flush=True)
                continue
            line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT_JSON ")), None)
            if line is None:
                print(
                    f"NO RESULT rc={r.returncode}\nstdout tail: {r.stdout[-1500:]}\nstderr tail: {r.stderr[-2500:]}",
                    flush=True,
                )
                continue
            res = json.loads(line[len("RESULT_JSON ") :])
            results.append(res)
            print(json.dumps(res, indent=1), flush=True)
            if r.stderr.strip():
                print("stderr tail:", r.stderr[-1200:], flush=True)
    print("\n\n##### SUMMARY")
    for res in results:
        head = f"{res['kernel']} bI={res['block_I']} st={res['num_stages']} th={res['threads']} H={res['H']}"
        if "compile_error" in res:
            print(f"{head}: COMPILE FAIL {res['compile_error'][-200:]!r}")
            continue
        for sn, rec in res.get("shapes", {}).items():
            if "launch_error" in rec:
                print(f"{head} {sn}: LAUNCH FAIL {rec['launch_error'][-300:]!r}")
            else:
                print(
                    f"{head} {sn}: rel_err={rec['rel_err']:.2e} max_abs={rec['max_abs']:.3e} ms={rec['ms']} nan={rec['nan']}"
                )


if __name__ == "__main__":
    main()
