# SGLang Test Log — Gemma-4 31B-it (BF16 dense), 4 Nodes, TP=4

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
| Image | `xomoxcc/dgx-spark-sglang:main-gemma4-sm121` |
| Model | `google/gemma-4-31B-it` |
| NCCL | 2.29.7+cuda13.2 (dgxspark-3node-ring) |
| Transport | **RoCE** via SR-IOV VF |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/gemma-4-31b-it/nv580.142_sglang-0.5.10_gemma-4-31b-it_n4_ep1.yaml`

---

## Model Notes

- 30.7B **dense** (NOT MoE), native BF16, multimodal-capable (text+image+video).
- Architecture: 60 layers, hybrid attention 5:1 sliding-to-full (`sliding_window=1024`).
- `num_attention_heads=32, num_key_value_heads=16` (2:1 GQA), `head_dim=256`, `global_head_dim=512`.
- Global layers: Unified K/V (`num_global_kv_heads=4`) + proportional RoPE (`partial_rotary_factor=0.25`).
- BF16 weights ~62 GB → TP=4 → ~15.5 GB/GPU. Comfortable fit on 4× 128 GB.
- No MoE → no `moe_runner_backend` sweep.
- No MTP head, no public EAGLE3 draft → speculative decoding not available.

## Image requirement

**Requires `xomoxcc/dgx-spark-sglang:main-gemma4-sm121`** (SGLang main branch,
pinned to PR #22079 merge commit). The upstream `scitrera/dgx-spark-sglang:0.5.10`
image does not include native Gemma-4 support (PR #21952, merged after v0.5.10
was cut). See `SGLANG_GEMMA4_UPSTREAM_BUG.md`.

## Expected issues

- **`attention_backend=flashinfer` will crash** — Gemma-4's `global_head_dim=512`
  is not in FlashInfer 0.6.7.post3's dispatch table. Tests 1–3 are expected to
  fail with `FlashInfer Internal Error: Invalid configuration` at `prefill.cuh:2615`.
  See `FLASHINFER_HEAD_DIM_512_UPSTREAM_BUG.md`. Confirmed on the MoE variant
  (`google/gemma-4-26B-A4B-it`) — same architecture, same crash.
- **`attention_backend=triton` should work** — PR #22079 added SM120/121-specific
  block sizes for Triton attention with `head_dim=512`.
- This is a **dense** model (60 layers, all compute active per token) — expect
  significantly lower tok/s than the 26B MoE variant (which has only ~3.8B
  active params). Rough estimate: ~30B active / 3.8B active ≈ ~8× more compute
  → expect ~20–25 tok/s at n=8 (vs 180 tok/s for MoE).

---

## Configuration Matrix

All tests use: `tp=4, pp=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.85, context_length=262144` unless noted. No MoE, no FP4.

| # | nccl | attention | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|---|------|-----------|----------------|---------------|--------|-----------|----------|----------|
| 1 | roce | fi | false | true | **startup_crash** | — | — | — |
| 2 | roce | fi | true | true | **bench_crash** | — | — | — |
| 3 | roce | fi | false | false | **startup_crash** | — | — | — |
| 4 | roce | triton | false | true | *running* | 10.8 | 40.8 | — |
| 5 | roce | triton | true | true | *pending* | — | — | — |
| 6 | roce | triton | false | false | *pending* | — | — | — |

Tests 1–3 use `attention_backend=flashinfer` — expected to crash (FlashInfer `head_dim=512` dispatch bug). Tests 4–6 use `attention_backend=triton` and should work.

### Column Legend

| Column | Description |
|--------|-------------|
| nccl | `nccl_transport` — NCCL inter-node transport (`roce` = RDMA/RoCE via SR-IOV VF) |
| attention | `attention_backend` — attention kernel (`fi` = FlashInfer, `triton` = Triton) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs |
| dis_piecewise | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |

---

## Results

### Tests 1–3 — flashinfer attn (all variants)

- Test 1 (CG on) and 3 (piecewise): **startup_crash** — FlashInfer `head_dim=512` dispatch bug during CUDA graph capture. `prefill.cuh:2615: Invalid configuration`.
- Test 2 (eager): **bench_crash** — same FlashInfer bug at first benchmark request.
- Identical to Gemma-4 26B MoE Tests 1–3. See `FLASHINFER_HEAD_DIM_512_UPSTREAM_BUG.md`.
