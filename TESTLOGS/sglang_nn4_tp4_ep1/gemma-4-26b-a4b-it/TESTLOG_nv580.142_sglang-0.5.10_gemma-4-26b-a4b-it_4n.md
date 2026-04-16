# SGLang Test Log — Gemma-4 26B-A4B-it (BF16), 4 Nodes, TP=4 EP=1

## Environment

| Component | Value                                          |
|-----------|------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node |
| Driver    | 580.142                                        |
| CUDA      | 13.2                                           |
| Kernel    | 6.19.11-custom                                 |
| OS        | Ubuntu 24.04 LTS (aarch64)                     |
| K3s       | v1.35.3+k3s1                                   |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)    |
| Image     | `xomoxcc/dgx-spark-sglang:main-gemma4-sm121`   |
| Model     | `google/gemma-4-26B-A4B-it`                    |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)          |
| Transport | **RoCE** via SR-IOV VF                         |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it/nv580.142_sglang-0.5.10_gemma-4-26b-a4b-it_n4_ep1.yaml`

---

## Model Notes

- 26B total / ~3.8B active MoE (128 experts, top_k=8, softmax routing), **native BF16** (no quantization).
- Architecture: Gemma-4 MoE — 30 layers, hybrid attention 5:1 sliding-to-full (`sliding_window=1024`).
- `num_attention_heads=16, num_key_value_heads=8` (2:1 GQA), `head_dim=256`, `global_head_dim=512`.
- MoE: `num_experts=128, top_k_experts=8, moe_intermediate_size=704`.
- BF16 weights ~52 GB total → TP=4 → ~13 GB/GPU, huge headroom.
- No MTP head, no public EAGLE3 draft → speculative decoding not available.

## Image requirement

**Requires `xomoxcc/dgx-spark-sglang:main-gemma4-sm121`** (SGLang main branch,
pinned to PR #22079 merge commit). The upstream `scitrera/dgx-spark-sglang:0.5.10`
image does not include native Gemma-4 support (PR #21952, merged after v0.5.10
was cut). On v0.5.10, all Gemma-4 variants crash in the Transformers fallback
backend due to the dual head_dim architecture (`head_dim=256` sliding vs
`global_head_dim=512` full attention). See `SGLANG_GEMMA4_UPSTREAM_BUG.md`.

Unlike the NVFP4 variant, BF16 does **not** need the modelopt_quant padding
workaround — there are no FP4 weight scales to pad. BF16 MoE weights are loaded
through the standard triton/flashinfer_cutlass MoE path, which handles
`moe_intermediate_size=704` natively.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.85, context_length=262144` unless noted. No `fp4_gemm_backend` (BF16 model). No `cutlass` MoE runner (requires FP4 tensors).

| #  | nccl | moe_runner | attention | dis_cuda_graph | dis_piecewise | Status            | n=1 tok/s | n=4 peak | n=8 peak |
|----|------|------------|-----------|----------------|---------------|-------------------|-----------|----------|----------|
| 1  | roce | triton     | fi        | false          | true          | **startup_crash** | —         | —        | —        |
| 2  | roce | triton     | fi        | true           | true          | **bench_crash**   | —         | —        | —        |
| 3  | roce | triton     | fi        | false          | false         | **startup_crash** | —         | —        | —        |
| 4  | roce | triton     | triton    | false          | true          | **STABLE**        | 40.1      | 114.8    | 163.9    |
| 5  | roce | triton     | triton    | true           | true          | **STABLE**        | 20.5      | 104.9    | 159.8    |
| 6  | roce | triton     | triton    | false          | false         | *pending*         | —         | —        | —        |
| 7  | roce | fi_cutlass | fi        | false          | true          | *pending*         | —         | —        | —        |
| 8  | roce | fi_cutlass | fi        | true           | true          | *pending*         | —         | —        | —        |
| 9  | roce | fi_cutlass | fi        | false          | false         | *pending*         | —         | —        | —        |
| 10 | roce | fi_cutlass | triton    | false          | true          | *pending*         | —         | —        | —        |
| 11 | roce | fi_cutlass | triton    | true           | true          | *pending*         | —         | —        | —        |
| 12 | roce | fi_cutlass | triton    | false          | false         | *pending*         | —         | —        | —        |

Tests 1, 2, 3, 7, 8, 9 use `attention_backend=flashinfer` — expected to crash with the FlashInfer `head_dim=512` dispatch bug (see Test 1 below). Tests 4–6, 10–12 use `attention_backend=triton` and should avoid it.

### Column Legend

| Column         | Description                                                                                                           |
|----------------|-----------------------------------------------------------------------------------------------------------------------|
| nccl           | `nccl_transport` — NCCL inter-node transport (`roce` = RDMA/RoCE via SR-IOV VF)                                       |
| moe_runner     | `moe_runner_backend` — MoE expert dispatch kernel (`fi_cutlass` = flashinfer_cutlass, `triton` = standard triton MoE) |
| attention      | `attention_backend` — attention kernel (`fi` = FlashInfer, `triton` = Triton)                                         |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs                                                 |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs                |

---

## Results

### Test 1 — triton MoE + flashinfer attn, CUDA graphs on

- **startup_crash** — FlashInfer attention kernel dispatch fails during CUDA graph capture.
- Model loaded successfully on `main-gemma4-sm121` image (native `gemma4_causal.py`, no Transformers fallback): 43s weight load, 13.68 GB, FP8 KV cache, sliding window memory pool initialized.
- CUDA graph capture started for bs [1, 2, 4, 8], crashed at bs=8 after ~80s:

```
FlashInfer Internal Error: Invalid configuration :
  NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=1 NUM_WARPS_Q=1 NUM_WARPS_KV=4
  at flashinfer/data/include/flashinfer/attention/prefill.cuh:2615
```

- **Root cause:** Gemma-4's global attention layers use `global_head_dim=512`. FlashInfer's `BatchPrefillWithPagedKVCacheDispatched` computes `NUM_MMA_D_QK = head_dim/16 = 512/16 = 32` — this MMA configuration is not in FlashInfer 0.6.7.post3's dispatch table. The kernel refuses to launch with "Invalid configuration".
- All 4 ranks (TP0–TP3) hit the exact same error simultaneously — this is a deterministic config-check failure, not a race or memory issue.
- **This is the same class of bug that PR #22079 fixed for Triton attention on GB200** (PTX register exhaustion with `head_dim=512`). FlashInfer attention was not addressed in that PR.
- **Workaround: `attention_backend=triton`** — Triton attention handles `head_dim=512` correctly (PR #22079 added SM120/121-specific block sizes). Tests 4–6 and 10–12 use triton attention and should avoid this crash.
- Also observed: `CUTE_DSL WARNING: Unexpected error during package walk: cutlass.cute.experimental` on all ranks — non-fatal, likely a missing CUTLASS DSL dependency in the image. Did not cause the crash.
- See `FLASHINFER_HEAD_DIM_512_UPSTREAM_BUG.md` for upstream tracking (FlashInfer PR #2959, open).

### Test 2 — triton MoE + flashinfer attn, eager (no CUDA graphs)

- **bench_crash** — same FlashInfer `head_dim=512` dispatch bug as Test 1, but hits during the first benchmark request instead of during CUDA graph capture (eager mode skips CG capture but the first `forward_decode` still goes through FlashInfer attention).

### Test 4 — triton MoE + triton attn, CUDA graphs on

- **STABLE** — all three concurrencies passed (0 failed requests). **First successful Gemma-4 serving on the cluster.**
- Peak tok/s: **40.1 / 114.8 / 163.9** (n=1/n=4/n=8).
- TTFT: 2.05s (n=1), 0.76s (n=4 p50), 0.41s (n=8 p50).
- Weight load: 43s, 13.68 GB (TP0), FP8 KV cache, sliding window memory pool (2.5M SWA + 3.1M full tokens, 74.4 GB).
- **163.9 tok/s at n=8** — the highest throughput of any model on this cluster, driven by only ~3.8B active parameters per token. For comparison: Qwen3.5-397B with MTP reaches 110.9 tok/s at n=8 (17B active).
