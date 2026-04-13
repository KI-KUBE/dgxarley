# SGLang Test Log — Qwen3.5 397B-A17B NVFP4, 4 Nodes, TP=4 EP=4, v0.5.10

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (SM121/Blackwell), 128 GB per node |
| Driver | 580.142 |
| CUDA | 13.2 |
| Kernel | 6.19.11-custom |
| OS | Ubuntu 24.04 LTS (aarch64) |
| K3s | v1.35.3+k3s1 |
| Nodes | spark1, spark2, spark3, spark4 (1 GPU each) |
| Image | `scitrera/dgx-spark-sglang:0.5.10` |
| Model | `nvidia/Qwen3.5-397B-A17B-NVFP4` |
| NCCL | 2.29.7+cuda13.2 (`dgxspark-3node-ring` build tag from scitrera image — functionally unrelated to our 4-node setup) |
| Transport | **RoCE** via SR-IOV VF (9.78 GB/s measured bus BW) |

---

## Model Notes

- 397B total / 17B active MoE (512 experts, top-10, softmax routing), NVFP4 quantized (~234 GB).
- Hybrid attention: 15 full GQA layers + 45 linear attention layers (every 4th layer is full attention). 60 layers total.
- 1 shared expert + 512 routed experts per MoE layer. Multimodal (text+image+video).
- Has MTP head (1 layer) for speculative decoding (NEXTN).
- `num_attention_heads=32, num_key_value_heads=2` — TP=4 per model card.
- NVFP4: only routed expert MoE FFN weights are FP4; attention, shared experts, vision encoder, lm_head, and MTP layer remain BF16.
- ~234 GB / 4 GPUs ≈ ~59 GB/GPU — fits on 4× DGX Spark.

## Key difference from the EP=1 test

- **EP=4 TP=4** — 128 of 512 experts per GPU (sharded), full intermediate dimension (not TP-sharded within MoE). Better GEMM efficiency per expert vs EP=1, but requires per-layer EP all-reduce through the `StandardDispatcher` combine path.
- **RoCE transport** — same as EP=1 (9.78 GB/s NCCL bus bandwidth).
- **Known risks at EP=4 on NVFP4:**
  - `triton` and `cutlass` direct MoE backends go through `cutlass_moe_fp4` which has the `StandardDispatcher` EP combine bug (see `SGLANG_NVFP4_SHUFFLE_ROWS_OOB_UPSTREAM_BUG.md`). Our monkey-patches in `sglang_launch.sh` (`torch.zeros` for a_map/c_map, `topk_weights.masked_fill(topk_ids < 0, 0)`) eliminate the crash but produce garbage output (the `apply_shuffle_mul_sum` path is still broken).
  - `flashinfer_cutlass` MoE has its own EP all-to-all routing and bypasses the broken codepath. This is the only MoE backend that works correctly at EP>1 for NVFP4.
  - The previous v0.5.10rc0 EP=4 matrix for this model had 100% crash rate across all 36 tests — socket transport + no monkey-patches + older sglang release. This current v0.5.10 run is the first EP=4 attempt with all fixes in place.
- **Runtime patches from `sglang_launch.sh` active:** `cute/mma.py` sm_120a/sm_121a admissible_archs, modelopt_quant.py EP-aware input_scale slicing + num_local_experts, cutlass_moe.py a_map/c_map zero-init + topk_weights mask, moe_wna16 qzeros EP remapping.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=4, nccl_transport=roce, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.70, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

| # | nccl | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|---|------|------------|-----------|----------|----------------|---------------|--------|-----------|----------|----------|
| 1 | roce | triton | fi | fi_cutlass | false | true | **STABLE** | 20.65 | 64.1 | 94.43 |
| 2 | roce | triton | fi | fi_cutlass | true | true | pending | — | — | — |
| 3 | roce | triton | fi | fi_cutlass | false | false | pending | — | — | — |
| 4 | roce | triton | triton | fi_cutlass | false | true | pending | — | — | — |
| 5 | roce | triton | triton | fi_cutlass | true | true | pending | — | — | — |
| 6 | roce | triton | triton | fi_cutlass | false | false | pending | — | — | — |
| 7 | roce | triton | fi | fi_cudnn | false | true | pending | — | — | — |
| 8 | roce | triton | fi | fi_cudnn | true | true | pending | — | — | — |
| 9 | roce | triton | fi | fi_cudnn | false | false | pending | — | — | — |
| 10 | roce | triton | triton | fi_cudnn | false | true | pending | — | — | — |
| 11 | roce | triton | triton | fi_cudnn | true | true | pending | — | — | — |
| 12 | roce | triton | triton | fi_cudnn | false | false | pending | — | — | — |
| 13 | roce | fi_cutlass | fi | fi_cutlass | false | true | pending | — | — | — |
| 14 | roce | fi_cutlass | fi | fi_cutlass | true | true | pending | — | — | — |
| 15 | roce | fi_cutlass | fi | fi_cutlass | false | false | pending | — | — | — |
| 16 | roce | fi_cutlass | triton | fi_cutlass | false | true | pending | — | — | — |
| 17 | roce | fi_cutlass | triton | fi_cutlass | true | true | pending | — | — | — |
| 18 | roce | fi_cutlass | triton | fi_cutlass | false | false | pending | — | — | — |
| 19 | roce | fi_cutlass | fi | fi_cudnn | false | true | pending | — | — | — |
| 20 | roce | fi_cutlass | fi | fi_cudnn | true | true | pending | — | — | — |
| 21 | roce | fi_cutlass | fi | fi_cudnn | false | false | pending | — | — | — |
| 22 | roce | fi_cutlass | triton | fi_cudnn | false | true | pending | — | — | — |
| 23 | roce | fi_cutlass | triton | fi_cudnn | true | true | pending | — | — | — |
| 24 | roce | fi_cutlass | triton | fi_cudnn | false | false | pending | — | — | — |
| 25 | roce | cutlass | fi | fi_cutlass | false | true | pending | — | — | — |
| 26 | roce | cutlass | fi | fi_cutlass | true | true | pending | — | — | — |
| 27 | roce | cutlass | fi | fi_cutlass | false | false | pending | — | — | — |
| 28 | roce | cutlass | triton | fi_cutlass | false | true | pending | — | — | — |
| 29 | roce | cutlass | triton | fi_cutlass | true | true | pending | — | — | — |
| 30 | roce | cutlass | triton | fi_cutlass | false | false | pending | — | — | — |
| 31 | roce | cutlass | fi | fi_cudnn | false | true | pending | — | — | — |
| 32 | roce | cutlass | fi | fi_cudnn | true | true | pending | — | — | — |
| 33 | roce | cutlass | fi | fi_cudnn | false | false | pending | — | — | — |
| 34 | roce | cutlass | triton | fi_cudnn | false | true | pending | — | — | — |
| 35 | roce | cutlass | triton | fi_cudnn | true | true | pending | — | — | — |
| 36 | roce | cutlass | triton | fi_cudnn | false | false | pending | — | — | — |

### Column Legend

| Column | Description |
|--------|-------------|
| nccl | `nccl_transport` — NCCL inter-node transport (`socket` = TCP/IP, `roce` = RDMA/RoCE via SR-IOV VF) |
| moe_runner | `moe_runner_backend` — MoE expert dispatch kernel (`fi_cutlass` = flashinfer_cutlass, `triton` = triton→cutlass_moe_fp4 fallback for NVFP4, `cutlass` = cutlass direct) |
| attention | `attention_backend` — attention kernel (`fi` = FlashInfer, `triton` = Triton) |
| fp4_gemm | `fp4_gemm_backend` — FP4 dense GEMM kernel (`fi_cutlass` = flashinfer_cutlass, `fi_cudnn` = flashinfer_cudnn) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs |
| dis_piecewise | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |
| n=1 tok/s | Per-request throughput at concurrency 1 |
| n=4 peak | Sum of per-request tok/s at concurrency 4 |
| n=8 peak | Sum of per-request tok/s at concurrency 8 |

---

## Results (matrix in progress)

### Legend

`STABLE` = all 3 concurrency levels (n=1, n=4, n=8) completed cleanly.
`FAIL` = matrix harness marked the test as failed (startup crash, watchdog timeout, or bench error).
`FAIL†` = partial results collected before failure (e.g. n=1 OK but n=4 or n=8 crashed).

### Expected patterns (based on EP=1 matrix + what we know)

- **Tests 1–12 (triton MoE):** expected to crash or produce garbage — `cutlass_moe_fp4` EP combine bug. Our monkey-patches eliminate the crash but the apply_shuffle_mul_sum path is still broken. If any stable row appears here, it is a surprise.
- **Tests 13–24 (flashinfer_cutlass MoE):** this is the expected winner region — fi_cutlass MoE has its own EP all-to-all routing. At EP=1 most of these FAILED, but EP=1 was outside the fi_cutlass "normal" operating range. EP=4 is where fi_cutlass MoE was designed to work.
- **Tests 25–36 (cutlass direct MoE):** same `cutlass_moe_fp4` codepath as triton MoE → same EP combine bug → expected to fail the same way.

### Comparison with EP=1 matrix winner

EP=1 Test 28 (cutlass direct MoE, triton attn, fi_cutlass fp4, CUDA graphs on):
- n=1: 21.5 tok/s
- n=4: 67.8 tok/s
- n=8: **102.0 tok/s**

Target for EP=4: match or exceed 102.0 tok/s at n=8. EP=4 has better GEMM efficiency per expert (full intermediate dimension per GPU) but adds per-layer EP all-reduce overhead. Net direction is hard to predict.

---

## Tests 1–36: matrix in progress

### Test 1 — `triton` MoE + `fi` attn + `fi_cutlass` fp4 (CUDA graphs on, piecewise off) — **STABLE** (surprise)

- n=1: 20.65 tok/s (ttft 2.80 s)
- n=4: 64.1 agg (16.1 per-request, ttft 0.85 s)
- n=8: 94.43 agg (12.02 per-request, ttft 1.15 s), 8/8 successful, 24,153 tokens in 255.78 s

Contrary to the expected "triton MoE crashes or garbage" prediction, this row is stable at EP=4. The `cutlass_moe_fp4` EP combine monkey-patches (`a_map/c_map` zero-init + `topk_weights` mask) are holding. Output quality not yet spot-checked — a passing bench only means no exceptions, not correct generations.

At n=8 this is still ~7% below the EP=1 winner (102.0 tok/s). More rows pending.

Results will continue to be filled in as the kikube-bench matrix progresses.
