# SGLang Test Log — MiniMax M2.5 NVFP4, 3 Nodes, v0.5.10rc0

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (SM121/Blackwell), 128 GB per node |
| Driver | 580.142 |
| CUDA | 13.0 |
| Kernel | 6.17.0-1014-nvidia |
| OS | Ubuntu 24.04.4 LTS (aarch64) |
| K3s | v1.35.3+k3s1 |
| Nodes | spark1, spark2, spark3 (1 GPU each) |
| Image | `scitrera/dgx-spark-sglang:0.5.10rc0` |
| Model | `nvidia/MiniMax-M2.5-NVFP4` |

Previous test series with `0.5.9-dev2-acab24a7-t5`: see `TESTLOG_nv580.142_sglang-0.5.9-dev2_minimax-m2.5-nvfp4_3n.md`.

---

## Baseline: Winner config from 0.5.9-dev2 series (Test 13)

```
tp_size=1, pp_size=3, ep_size=1
moe_runner_backend=triton
attention_backend=flashinfer
fp4_gemm_backend=flashinfer_cutlass
disable_cuda_graph=false
disable_piecewise_cuda_graph=true
pp_async_batch_depth=0
cuda_graph_max_bs=8
disable_deep_gemm=true
quantization=modelopt_fp4
kv_cache_dtype=fp8_e4m3
mem_fraction_static=0.80
context_length=196608
max_running_requests=32
```

Throughput on 0.5.9-dev2: 16.1 tok/s (1∥), 31.5 tok/s (4∥ avg), 50.6 tok/s (4∥ peak).

---

## 2026-04-01: v0.5.10rc0 — initial test

### Test 1: Winner config from 0.5.9-dev2

- **Config:** Same as baseline above. No changes except image `0.5.10rc0`.
- **Key question:** Does the FlashInfer CUTLASS MoE Xid 13 bug (`0x1c81fb60:0x1174`) still exist in 0.5.10rc0? If fixed, `moe_runner_backend=flashinfer_cutlass` could be restored for better performance.
- **Result:** *pending*

---

## Configuration Matrix

All tests use: `tp=1, pp=3, ep=1, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.80, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

| # | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ avg | 4∥ peak | 8∥ avg | 8∥ peak |
|---|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|--------|---------|--------|---------|
| 1 | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | *pending* | — | — | — | — | — |

### Column Legend

| Column | Description |
|--------|-------------|
| moe_runner | `moe_runner_backend` — MoE expert dispatch kernel (`fi_cutlass` = flashinfer_cutlass, `triton` = triton→cutlass_moe_fp4 fallback for NVFP4) |
| attention | `attention_backend` — attention kernel |
| fp4_gemm | `fp4_gemm_backend` — FP4 dense GEMM kernel |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs |
| dis_piecewise | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs |
| pp_async | `pp_async_batch_depth` — async micro-batches in PP pipeline (0 = synchronous) |
| cuda_graph_max_bs | `cuda_graph_max_bs` — largest batch size to capture |
| 1∥ tok/s | Aggregate throughput with 1 sequential request |
| 4∥ avg | Aggregate throughput with 4 parallel requests (total output tokens / wall time) |
| 4∥ peak | Peak concurrent throughput at 4∥ = sum of per-request tok/s while all requests active |
| 8∥ avg | Aggregate throughput with 8 parallel requests |
| 8∥ peak | Peak concurrent throughput at 8∥ |
