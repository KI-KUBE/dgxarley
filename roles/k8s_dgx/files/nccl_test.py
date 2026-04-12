#!/usr/bin/env python3
"""NCCL connectivity + bandwidth test across 4 DGX Spark GB10 nodes.

Launched as a ConfigMap-mounted script inside K8s pods, one pod per rank.
Uses torch.distributed with NCCL backend. Env vars control transport mode
(socket / RoCE / host-network) — set by the Ansible task via the pod spec.

Required env vars:
  RANK            — 0..3
  WORLD_SIZE      — 4
  MASTER_ADDR     — rendezvous IP (rank 0's QSFP or host IP)
  MASTER_PORT     — rendezvous port (default 29500)
  NCCL_*          — transport-specific NCCL env vars (set by pod spec)

Output: prints per-size all_reduce bandwidth + latency, then a summary.
"""

import os
import sys
import time

import torch
import torch.distributed as dist


def human_bytes(n: int) -> str:
    v: float = n
    for unit in ("B", "KB", "MB", "GB"):
        if v < 1024:
            return f"{v:.0f} {unit}" if v == int(v) else f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} TB"


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master = os.environ["MASTER_ADDR"]
    port = os.environ.get("MASTER_PORT", "29500")

    transport = "unknown"
    if os.environ.get("NCCL_NET") == "Socket":
        transport = "socket"
    elif os.environ.get("NCCL_IB_HCA"):
        transport = f"roce ({os.environ['NCCL_IB_HCA']})"
    elif os.environ.get("HOST_NETWORK_MODE"):
        transport = "host-network"

    if rank == 0:
        print(f"=== NCCL connectivity test ===")
        print(f"Transport:  {transport}")
        print(f"World size: {world_size}")
        print(f"Master:     {master}:{port}")
        print(f"NCCL env:   " + ", ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("NCCL_")))
        print()

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master}:{port}",
        rank=rank,
        world_size=world_size,
    )

    if rank == 0:
        print(f"init_process_group OK (backend=nccl)")
        print()

    # Warmup
    x = torch.ones(1024, device="cuda")
    dist.all_reduce(x)
    torch.cuda.synchronize()

    # Test sizes: 4 KB to 256 MB (powers of 4)
    sizes = []
    s = 1024  # 4 KB in float32 (1024 elements * 4 bytes)
    while s <= 64 * 1024 * 1024:  # 256 MB in float32
        sizes.append(s)
        s *= 4

    if rank == 0:
        print(f"{'Size':>12}  {'Latency':>10}  {'Bandwidth':>12}  {'Status'}")
        print(f"{'-' * 12}  {'-' * 10}  {'-' * 12}  {'-' * 8}")

    results = []
    for numel in sizes:
        nbytes = numel * 4  # float32
        x = torch.randn(numel, device="cuda")
        torch.cuda.synchronize()
        dist.barrier()

        # Multiple iterations for stable measurement
        iters = max(1, min(50, 128 * 1024 * 1024 // nbytes))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        latency_us = elapsed / iters * 1e6
        # all_reduce bus bandwidth: 2 * (n-1)/n * size / time
        algbw = nbytes / (elapsed / iters)
        busbw = algbw * 2 * (world_size - 1) / world_size

        results.append((nbytes, latency_us, busbw))

        if rank == 0:
            print(f"{human_bytes(nbytes):>12}  " f"{latency_us:>8.0f}us  " f"{busbw / 1e9:>9.2f} GB/s  " f"OK")

    dist.barrier()

    if rank == 0:
        print()
        peak_bw = max(r[2] for r in results)
        print(f"Peak bus bandwidth: {peak_bw / 1e9:.2f} GB/s")
        print(f"Transport:          {transport}")
        print(f"=== NCCL test complete ===")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
