# SGLang Test Log — Gemma-4 26B-A4B-it NVFP4, 4 Nodes, TP=4 EP=1

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
| Model | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` |
| NCCL | 2.29.7+cuda13.2 (dgxspark-3node-ring) |
| Transport | **RoCE** via SR-IOV VF |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it-nvfp4/nv580.142_sglang-0.5.10_gemma-4-26b-a4b-it-nvfp4_n4_ep1.yaml`

---

## Model Notes

- 26B total / ~3.8B active MoE (128 experts, top_k=8, softmax routing), NVFP4 quantized.
- Architecture: Gemma-4 MoE — 30 layers, hybrid attention 5:1 sliding-to-full (`sliding_window=1024`).
- `num_attention_heads=16, num_key_value_heads=8` (2:1 GQA), `head_dim=256`, `global_head_dim=512`.
- MoE: `num_experts=128, top_k_experts=8, moe_intermediate_size=704`.
- NVFP4: MoE FFN weights are FP4; attention, lm_head remain BF16.
- ~13–15 GB total after quantization → ~3–4 GB/GPU at TP=4. Huge headroom.
- No MTP head, no public EAGLE3 draft → speculative decoding not available.

## Blocker: NVFP4 + TP=4 cannot work (2026-04-16)

**All 36 tests will fail at startup.** The NVFP4 MoE weight processing in
`modelopt_quant.py` crashes with:

```
NVFP4 w2_weight_scale K' not multiple of 4: shape=(128, 2816, 11), group_size=16
AssertionError: The intermediate size required padding, but padding is also
  implemented for gated activations
```

**Root cause:** Gemma-4 MoE has `moe_intermediate_size=704`. At TP=4 this becomes
704/4 = 176 per shard → 176/16 (group_size) = **11** → not a multiple of 4 →
requires padding. But the modelopt NVFP4 weight processor cannot handle padding
AND gated activations (GEGLU) simultaneously. This is an upstream limitation in
`modelopt_quant.py:1846`, not covered by any known PR.

**This is independent of all other Gemma-4 bugs** (dual head_dim, top_k_experts
naming, per-expert weight loading, GEGLU activation). Those were fixed or
patched; this one blocks before any of them matter.

**Possible workarounds (untested):**
- **TP=2** → 704/2 = 352 → 352/16 = 22 → still not multiple of 4 → still blocked.
- **TP=1** → 704/1 = 704 → 704/16 = 44 → **multiple of 4** → might work, but requires a single GPU with enough VRAM (~15 GB fits on one GB10).
- **Upstream fix** in `modelopt_quant.py` to support padding + gated activations simultaneously.

See `SGLANG_GEMMA4_UPSTREAM_BUG.md` for the full bug tracking document.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.90, disable_deep_gemm=true, context_length=262144` unless noted.

| #  | nccl | moe_runner | attention | fp4_gemm   | dis_cuda_graph | dis_piecewise | Status      | n=1 tok/s | n=4 peak | n=8 peak |
|----|------|------------|-----------|------------|----------------|---------------|-------------|-----------|----------|----------|
| 1  | roce | triton     | fi        | fi_cutlass | false          | true          | **blocked** | —         | —        | —        |
| 2  | roce | triton     | fi        | fi_cutlass | true           | true          | **blocked** | —         | —        | —        |
| 3  | roce | triton     | fi        | fi_cutlass | false          | false         | **blocked** | —         | —        | —        |
| 4  | roce | triton     | triton    | fi_cutlass | false          | true          | **blocked** | —         | —        | —        |
| 5  | roce | triton     | triton    | fi_cutlass | true           | true          | **blocked** | —         | —        | —        |
| 6  | roce | triton     | triton    | fi_cutlass | false          | false         | **blocked** | —         | —        | —        |
| 7  | roce | triton     | fi        | fi_cudnn   | false          | true          | **blocked** | —         | —        | —        |
| 8  | roce | triton     | fi        | fi_cudnn   | true           | true          | **blocked** | —         | —        | —        |
| 9  | roce | triton     | fi        | fi_cudnn   | false          | false         | **blocked** | —         | —        | —        |
| 10 | roce | triton     | triton    | fi_cudnn   | false          | true          | **blocked** | —         | —        | —        |
| 11 | roce | triton     | triton    | fi_cudnn   | true           | true          | **blocked** | —         | —        | —        |
| 12 | roce | triton     | triton    | fi_cudnn   | false          | false         | **blocked** | —         | —        | —        |
| 13 | roce | fi_cutlass | fi        | fi_cutlass | false          | true          | **blocked** | —         | —        | —        |
| 14 | roce | fi_cutlass | fi        | fi_cutlass | true           | true          | **blocked** | —         | —        | —        |
| 15 | roce | fi_cutlass | fi        | fi_cutlass | false          | false         | **blocked** | —         | —        | —        |
| 16 | roce | fi_cutlass | triton    | fi_cutlass | false          | true          | **blocked** | —         | —        | —        |
| 17 | roce | fi_cutlass | triton    | fi_cutlass | true           | true          | **blocked** | —         | —        | —        |
| 18 | roce | fi_cutlass | triton    | fi_cutlass | false          | false         | **blocked** | —         | —        | —        |
| 19 | roce | fi_cutlass | fi        | fi_cudnn   | false          | true          | **blocked** | —         | —        | —        |
| 20 | roce | fi_cutlass | fi        | fi_cudnn   | true           | true          | **blocked** | —         | —        | —        |
| 21 | roce | fi_cutlass | fi        | fi_cudnn   | false          | false         | **blocked** | —         | —        | —        |
| 22 | roce | fi_cutlass | triton    | fi_cudnn   | false          | true          | **blocked** | —         | —        | —        |
| 23 | roce | fi_cutlass | triton    | fi_cudnn   | true           | true          | **blocked** | —         | —        | —        |
| 24 | roce | fi_cutlass | triton    | fi_cudnn   | false          | false         | **blocked** | —         | —        | —        |
| 25 | roce | cutlass    | fi        | fi_cutlass | false          | true          | **blocked** | —         | —        | —        |
| 26 | roce | cutlass    | fi        | fi_cutlass | true           | true          | **blocked** | —         | —        | —        |
| 27 | roce | cutlass    | fi        | fi_cutlass | false          | false         | **blocked** | —         | —        | —        |
| 28 | roce | cutlass    | triton    | fi_cutlass | false          | true          | **blocked** | —         | —        | —        |
| 29 | roce | cutlass    | triton    | fi_cutlass | true           | true          | **blocked** | —         | —        | —        |
| 30 | roce | cutlass    | triton    | fi_cutlass | false          | false         | **blocked** | —         | —        | —        |
| 31 | roce | cutlass    | fi        | fi_cudnn   | false          | true          | **blocked** | —         | —        | —        |
| 32 | roce | cutlass    | fi        | fi_cudnn   | true           | true          | **blocked** | —         | —        | —        |
| 33 | roce | cutlass    | fi        | fi_cudnn   | false          | false         | **blocked** | —         | —        | —        |
| 34 | roce | cutlass    | triton    | fi_cudnn   | false          | true          | **blocked** | —         | —        | —        |
| 35 | roce | cutlass    | triton    | fi_cudnn   | true           | true          | **blocked** | —         | —        | —        |
| 36 | roce | cutlass    | triton    | fi_cudnn   | false          | false         | **blocked** | —         | —        | —        |

All 36 tests blocked by the `modelopt_quant.py` intermediate-size padding + gated activation assertion at TP=4. No test was run.

### Test 1 — triton MoE + flashinfer attn + fi_cutlass FP4, CUDA graphs on

- **startup_crash** — `AssertionError: The intermediate size required padding, but padding is also implemented for gated activations` in `modelopt_quant.py:1846` during `process_weights_after_loading`.
- Image: `xomoxcc/dgx-spark-sglang:main-gemma4-sm121` (SGLang main @ PR #22079 + Gemma4 NVFP4 patches #22929/#22928).
- The native Gemma-4 model implementation loaded successfully (no Transformers fallback). Weight loading progressed through all 3 shards. Crash occurred during post-load weight quantization processing.
- Warning before crash: `w1_weight_scale_2 must match w3_weight_scale_2. Accuracy may be affected.` — indicates the fused gate_up_proj has mismatched per-channel scales, a secondary issue.
