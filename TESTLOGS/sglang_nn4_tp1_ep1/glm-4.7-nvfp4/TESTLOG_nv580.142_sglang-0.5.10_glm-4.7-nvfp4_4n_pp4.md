# SGLang Test Log — GLM 4.7 NVFP4, 4 Nodes PP=4, v0.5.10

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (SM121/Blackwell), 128 GB per node |
| Driver | 580.142 |
| CUDA | 13.0 |
| Kernel | 6.17.0-1014-nvidia |
| OS | Ubuntu 24.04.4 LTS (aarch64) |
| K3s | v1.35.3+k3s1 |
| Nodes | spark1, spark2, spark3, spark4 (1 GPU each) |
| Image | `scitrera/dgx-spark-sglang:0.5.10` |
| Model | `nvidia/GLM-4.7-NVFP4` |

Previous test series: v0.5.10rc0 (`TESTLOG_nv580.142_sglang-0.5.10rc0_glm-4.7-nvfp4_4n_pp4.md`).

---

## Configuration Matrix

All tests use: `tp=1, pp=4, ep=1, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.60, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

**No test is stable at all three concurrency levels (n=1, n=4, n=8).** The best results are tests 1, 3, 25, 27 which complete n=1 at ~5.5 tok/s but fail at n=4+.

| # | nccl_transport | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ tok/s | 8∥ tok/s |
|---|----------------|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|---------|---------|
| 1 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 16 | partial | 5.4 | — | — |
| 2 | socket | triton | flashinfer | fi_cutlass | true | true | 0 | — | partial | 3.7 | — | — |
| 3 | socket | triton | flashinfer | fi_cutlass | false | false | 0 | 16 | partial | 5.5 | — | — |
| 4 | socket | triton | triton | fi_cutlass | false | true | 0 | 16 | CRASH | — | — | — |
| 5 | socket | triton | triton | fi_cutlass | true | true | 0 | — | CRASH | — | — | — |
| 6 | socket | triton | triton | fi_cutlass | false | false | 0 | 16 | CRASH | — | — | — |
| 7 | socket | triton | flashinfer | fi_cudnn | false | true | 0 | 16 | CRASH | — | — | — |
| 8 | socket | triton | flashinfer | fi_cudnn | true | true | 0 | — | FAIL | — | — | — |
| 9 | socket | triton | flashinfer | fi_cudnn | false | false | 0 | 16 | CRASH | — | — | — |
| 10 | socket | triton | triton | fi_cudnn | false | true | 0 | 16 | CRASH | — | — | — |
| 11 | socket | triton | triton | fi_cudnn | true | true | 0 | — | CRASH | — | — | — |
| 12 | socket | triton | triton | fi_cudnn | false | false | 0 | 16 | CRASH | — | — | — |
| 13 | socket | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 16 | CRASH | — | — | — |
| 14 | socket | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | FAIL | — | — | — |
| 15 | socket | fi_cutlass | flashinfer | fi_cutlass | false | false | 0 | 16 | CRASH | — | — | — |
| 16 | socket | fi_cutlass | triton | fi_cutlass | false | true | 0 | 16 | CRASH | — | — | — |
| 17 | socket | fi_cutlass | triton | fi_cutlass | true | true | 0 | — | CRASH | — | — | — |
| 18 | socket | fi_cutlass | triton | fi_cutlass | false | false | 0 | 16 | CRASH | — | — | — |
| 19 | socket | fi_cutlass | flashinfer | fi_cudnn | false | true | 0 | 16 | CRASH | — | — | — |
| 20 | socket | fi_cutlass | flashinfer | fi_cudnn | true | true | 0 | — | FAIL | — | — | — |
| 21 | socket | fi_cutlass | flashinfer | fi_cudnn | false | false | 0 | 16 | CRASH | — | — | — |
| 22 | socket | fi_cutlass | triton | fi_cudnn | false | true | 0 | 16 | CRASH | — | — | — |
| 23 | socket | fi_cutlass | triton | fi_cudnn | true | true | 0 | — | CRASH | — | — | — |
| 24 | socket | fi_cutlass | triton | fi_cudnn | false | false | 0 | 16 | CRASH | — | — | — |
| 25 | socket | cutlass | flashinfer | fi_cutlass | false | true | 0 | 16 | partial | 5.5 | — | — |
| 26 | socket | cutlass | flashinfer | fi_cutlass | true | true | 0 | — | partial | 3.7 | — | — |
| 27 | socket | cutlass | flashinfer | fi_cutlass | false | false | 0 | 16 | partial | 5.5 | — | — |
| 28 | socket | cutlass | triton | fi_cutlass | false | true | 0 | 16 | CRASH | — | — | — |
| 29 | socket | cutlass | triton | fi_cutlass | true | true | 0 | — | CRASH | — | — | — |
| 30 | socket | cutlass | triton | fi_cutlass | false | false | 0 | 16 | CRASH | — | — | — |
| 31 | socket | cutlass | flashinfer | fi_cudnn | false | true | 0 | 16 | CRASH | — | — | — |
| 32 | socket | cutlass | flashinfer | fi_cudnn | true | true | 0 | — | FAIL | — | — | — |
| 33 | socket | cutlass | flashinfer | fi_cudnn | false | false | 0 | 16 | CRASH | — | — | — |
| 34 | socket | cutlass | triton | fi_cudnn | false | true | 0 | 16 | CRASH | — | — | — |
| 35 | socket | cutlass | triton | fi_cudnn | true | true | 0 | — | CRASH | — | — | — |
| 36 | socket | cutlass | triton | fi_cudnn | false | false | 0 | 16 | CRASH | — | — | — |
| 37 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 16 | CRASH | — | — | — |

> **#37** = #1 winner config + MTP speculative decoding (NEXTN, 3 steps, 4 draft tokens)

### Column Legend

| Column | Description |
|--------|-------------|
| nccl_transport | `sglang_nccl_transport` — NCCL inter-node transport (`socket` = TCP/IP, `roce` = RDMA/RoCE via IBext) |
| moe_runner | `moe_runner_backend` — MoE expert dispatch kernel (`fi_cutlass` = flashinfer_cutlass, `triton` = triton→cutlass_moe_fp4 fallback for NVFP4, `cutlass` = cutlass direct) |
| attention | `attention_backend` — attention kernel (`flashinfer` = FlashInfer, `triton` = Triton) |
| fp4_gemm | `fp4_gemm_backend` — FP4 dense GEMM kernel (`fi_cutlass` = flashinfer_cutlass, `fi_cudnn` = flashinfer_cudnn; valid choices: auto, flashinfer_cudnn, flashinfer_cutlass, flashinfer_trtllm) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs |
| dis_piecewise | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |
| pp_async | `pp_async_batch_depth` — async micro-batches in PP pipeline (0 = synchronous) |
| cuda_graph_max_bs | `cuda_graph_max_bs` — largest batch size to capture (— = N/A when graphs disabled) |
| 1∥ tok/s | Throughput with 1 sequential request (= per-request tok/s) |
| 4∥ tok/s | Peak concurrent throughput at 4∥ (sum of per-request tok/s) |
| 8∥ tok/s | Peak concurrent throughput at 8∥ (sum of per-request tok/s) |

---

## Failure Patterns

### Pattern 1: `triton` attn crashes on PP=4 (tests 4–6, 10–12, 28–30, 34–36)

All configurations using `attention_backend=triton` with PP=4 crash at startup. Every triton-attn test (across all MoE and fp4 backends) is a startup_crash. `flashinfer` attention is required for PP=4 on GLM-4.7.

### Pattern 2: `fi_cudnn` fp4_gemm is broken in v0.5.10 (tests 8, 14, 20, 32)

All `flashinfer_cudnn` fp4_gemm tests that reach inference return 0 tokens (infer_error). This is the same regression observed in the TP=4 EP=4 matrix — `fi_cudnn` fp4_gemm is completely non-functional in v0.5.10 on SM121. Tests with CUDA graphs + fi_cudnn crash at startup (7, 9, 19, 21, 31, 33).

### Pattern 3: `fi_cutlass` MoE crashes on PP=4 (tests 13–24)

All `flashinfer_cutlass` MoE tests crash or fail. CUDA graph variants crash at startup. Eager variants (14, 17, 20, 23) either return 0 tokens or crash. The `fi_cutlass` MoE kernel is not compatible with PP=4 on this model.

### Pattern 4: n=1 works but n=4+ fails (tests 1, 2, 3, 25, 26, 27)

Six tests complete n=1 successfully but fail at n=4 and n=8 with `error` (0 tokens). These all use `flashinfer` attn + `fi_cutlass` fp4 with either `triton` or `cutlass` MoE. The server stays alive but cannot handle concurrent requests.

| Test | MoE | CUDA graphs | n=1 tok/s | n=1 TTFT |
|------|-----|-------------|-----------|----------|
| 1 | triton | yes (fixed-BS) | 5.4 | 5.9s |
| 3 | triton | yes (piecewise) | 5.5 | 4.7s |
| 25 | cutlass | yes (fixed-BS) | 5.5 | 1.4s |
| 27 | cutlass | yes (piecewise) | 5.5 | 1.3s |
| 2 | triton | no (eager) | 3.7 | **257.4s** |
| 26 | cutlass | no (eager) | 3.7 | **251.8s** |

**Note:** Eager mode (tests 2, 26) has catastrophic TTFT (~4 min) — CUDA graphs are essential for acceptable latency on PP=4. The cutlass MoE variants (25, 27) have significantly lower TTFT (1.3–1.4s vs 5.9s) than triton MoE with CUDA graphs.

### Test 14 — fi_cutlass MoE / flashinfer attn / fi_cutlass fp4 / no-cuda-graph

n=1 generated 16 tokens at 0.24 tok/s with TTFT 49.8s before erroring. Very slow — likely JIT compilation overhead.

### Test 15 — fi_cutlass MoE / flashinfer attn / fi_cutlass fp4 / piecewise

n=1 aborted after generating tokens (4.4 tok/s, TTFT 8.6s), Worker-1 restart.

### Test 37 — #1 winner + MTP speculative decoding (NEXTN)

Startup crash. PP + speculative decoding is still incompatible in v0.5.10.

---

## Summary — v0.5.10 PP=4 GLM-4.7

**No fully stable configuration exists.** All 37 tests fail at n=4 or higher.

| Finding | Detail |
|---------|--------|
| **Best n=1 only** | Tests 25, 27 (cutlass MoE + flashinfer attn + fi_cutlass fp4 + CUDA graphs): 5.5 tok/s, TTFT 1.3s |
| **`triton` attn** | Always crashes on PP=4 — use `flashinfer` attn only |
| **`fi_cudnn` fp4** | Broken in v0.5.10 (0 tokens) — same regression as TP=4 EP=4 |
| **`fi_cutlass` MoE** | Crashes on PP=4 — use `triton` or `cutlass` MoE |
| **Eager mode** | TTFT ~250s — unusable without CUDA graphs |
| **Speculative** | PP + NEXTN still incompatible |

### Cross-matrix comparison — GLM-4.7 best configs across versions

| Setup | Config | n=1 | n=4 | n=8 | Verdict |
|-------|--------|-----|-----|-----|---------|
| **rc0 TP=4 EP=4 #23** | fi_cutlass MoE + triton attn + fi_cudnn fp4 + eager | **8.06** | **21.94** | **30.01** | **Only fully stable config across all versions** |
| v0.5.10 TP=4 EP=4 #17 | fi_cutlass MoE + triton attn + fi_cutlass fp4 + eager | 8.4 | 20.8 | crashed | n=8 unstable |
| v0.5.10 PP=4 #25/#27 | cutlass MoE + flashinfer attn + fi_cutlass fp4 + CUDA graphs | 5.5 | 0 | 0 | n=1 only |
| v0.5.10 TP=4 EP=4 #23 | fi_cutlass MoE + triton attn + fi_cudnn fp4 + eager | 0 | 0 | 0 | **`fi_cudnn` regression** |
| rc0 PP=4 | all configs | — | — | — | all-fail on rc0 too |

### Recommendation

**Stay on v0.5.10rc0 with TP=4 EP=4 for GLM-4.7.** The rc0 winner config (test 23: fi_cutlass MoE + triton attn + fi_cudnn fp4 + eager, 8.06/21.94/30.01 tok/s) is the only configuration that is fully stable at n=1, n=4, and n=8 across all tested versions. v0.5.10 introduced a `flashinfer_cudnn` fp4_gemm regression that breaks this config, and no alternative config in v0.5.10 achieves full n=8 stability — neither with TP=4 EP=4 nor PP=4.
