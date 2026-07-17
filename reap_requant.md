# GLM-5.2-REAP-504B-v2 → NVFP4 W4A4 attention requant — technical report

Companion document to the model card of
[`vroomfondel/glm-5.2-reap-504B-v2-W4A4`](https://huggingface.co/vroomfondel/glm-5.2-reap-504B-v2-W4A4).

A downstream serving-optimization experiment on
[`0xSero/glm-5.2-reap-504B-v2`](https://huggingface.co/0xSero/glm-5.2-reap-504B-v2),
produced on and for the **dgxarley** DGX Spark cluster: 4× NVIDIA GB10
(**SM121**, consumer Blackwell, 128 GB unified memory, ARM64), tensor-parallel 4,
SGLang with DSA sparse attention + MTP speculative decoding.

**Status (2026-07-17): deployed and empirically validated** — GSM8K 93.0% (0
errors), single-stream decode measurably faster; the loop/attractor sweep is in
progress. This is an **experimental research artifact**, not a production release.

## Summary

The published REAP checkpoint is NVFP4 on the routed experts but **BF16 on the
attention**. On consumer Blackwell at batch size 1 those unquantized BF16 MLA
projections dominate the decode step (pure weight bandwidth). This variant
selectively requantizes the two heaviest of them — `o_proj` and `q_b_proj`
(≈80% of the attention weight bytes) — from BF16 to **NVFP4 W4A4**, data-free.
Everything else is byte-identical to the base: routed experts (already NVFP4),
the low-rank `q_a_proj` / `kv_a_proj_with_mqa` / `kv_b_proj` projections, the DSA
indexer, router gates, shared expert, `lm_head`, and the MTP layer all stay BF16.

## 1. Motivation (profiled, not estimated)

Live SGLang profiling of the decode step (head, 22 steps, ~131 ms kernel
time/token) with the cluster's DSA runtime patches (native SM120 sparse attention
+ Triton indexer) already applied:

| ms/token | share | what |
|---|---|---|
| ~59 | ~45% | cuBLAS **BF16 GEMV** (166 µs × ~3.25/layer) + fused-a-gemm + lm_head |
| ~12 | ~10% | small BF16 wmma GEMMs (kv_b / indexer projections) |
| ~27 | ~21% | NVFP4 MoE (grouped cutlass) |
| ~10 | ~8%  | NCCL AllReduce (TP4) |
| ~3  | ~2.5% | sparse attention |

The **unquantized BF16 MLA projections are the decode floor** (~71 ms/token, pure
weight bandwidth at bs=1). Of those, `o_proj` and `q_b_proj` carry the bulk of the
bytes; `q_a` / `kv_a` / `kv_b` are small **and** quantization-sensitive (low-rank
compressions of the MLA), so they are deliberately left BF16.

**Goal:** quantize `o_proj` + `q_b_proj` to NVFP4 → cut the decode floor, a gain
that is multiplicative with MTP speculative decoding.

## 2. Approach

- **W4A4 (cutlass), not W4A16 (Marlin).** A GB10 batch sweep on the real
  per-rank shapes showed the Marlin W4A16 (weight-only) path collapses above
  bs≈64 (≈3× slower than cutlass at bs=512) and is unstable at bs=1, whereas the
  cutlass NVFP4 **W4A4** path (both weights AND activations FP4) is stable across
  batch sizes and clearly faster at high concurrency. W4A4 was chosen so the win
  is not limited to bs=1.
- **Heuristic `input_scale`, no calibration.** The W4A4 dense-linear kernel needs
  a *static* per-tensor `input_scale` (it does not support dynamic activation
  quant). The 504B model does not fit a single-node modelopt PTQ pass, so
  `input_scale` is a **generous heuristic** amax overestimate (overestimating =
  no clipping, modest precision loss; the per-block e4m3 activation scales adapt).
  This is the one accuracy-relevant approximation — validated post-hoc by the
  GSM8K gate (§4), not proven a priori.
- **Data-free weights.** `weight` / `weight_scale` / `weight_scale_2` are computed
  directly from the BF16 weights with the standard modelopt NVFP4 packing.
- **Directly on the published v2.** In this REAP model the BF16 attention *is* the
  knowledge-distillation recovery — a logit-KD LoRA (targeting exactly
  `q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj`) is merged into it.
  Rebuilding from an unquantized base would discard that recovery, so the requant
  operates on the published checkpoint and touches only two tensors per layer.

## 3. Format and config surgery

NVFP4, per quantized matrix: `weight` (e2m1 FP4, 2 values/byte in a uint8
container), `weight_scale` (float8_e4m3fn block scales, block size 16, **linear**
layout, `is_sf_swizzled_layout=False`), `weight_scale_2` (fp32 scalar =
`amax/(6·448)`), plus the static `input_scale` (fp32 scalar) required by W4A4.
The exact packing/scale layout was verified against a real NVFP4 expert tensor in
the base checkpoint (dequant round-trip ≈0.09 rel-err = ordinary NVFP4 fidelity).

Config: `o_proj` and `q_b_proj` are removed from `quantization_config.ignore` so
they fall into the checkpoint's existing W4A4 group (same algorithm as the
experts) — no `MIXED_PRECISION` needed. The requant streams shards tensor by
tensor (never the whole model in memory), quantizes only those two projections
for the requantized layer range, and passes everything else through
byte-identical.

### Two SGLang-loader gotchas (the reusable findings)

Getting the surgical `ignore` list right on an MLA (DeepSeek/GLM) model has two
non-obvious traps. Both surface as a load-time shape/dtype `AssertionError`, which
crashes the head and then cascades into a distributed-rendezvous failure that
*looks* like a networking/timeout problem but is not.

1. **`fused_qkv_a_proj_with_mqa` — the trailing-`*` trap.** SGLang fuses
   `q_a_proj` + `kv_a_proj_with_mqa` into one runtime parameter
   `fused_qkv_a_proj_with_mqa`; the separate names never exist as model params.
   Its modelopt loader (`is_layer_excluded`) only recognises that fused param as
   excluded when an `ignore` entry's **last dot-segment exactly equals** one of
   `{q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj}` (set membership, then a
   substring match into the fused name). A trailing `*` (e.g.
   `kv_a_proj_with_mqa*`) makes the tail `kv_a_proj_with_mqa*`, which is **not** in
   the set → the fused layer is treated as W4A4 while the checkpoint is BF16 →
   assertion `[2624,3072] uint8` vs `[2624,6144] bf16`. Fix: emit the
   fused-component `ignore` entries **without** a trailing `*` (non-fused entries
   keep it). The stock base model dodges this by using a broad
   `model.layers.N.self_attn*` glob that direct-matches the fused name.
2. **MTP layer over-reach.** The model is 78 layers (0–77) + 1 MTP/NEXTN layer
   (`model.layers.78`). The requant covers layers 0–77 only, so the MTP layer's
   `o_proj`/`q_b_proj` stay BF16 in the checkpoint. The ignore-list rewrite must
   be **gated on the requant layer range** — if it also un-ignores the MTP layer,
   SGLang expects W4A4 for a BF16 MTP `q_b_proj` → assertion `[4096,1024] uint8`
   vs `[4096,2048] bf16`. Fix: out-of-range layers keep their broad `self_attn*`
   glob (attention stays BF16).

Both fixes were validated **before deploying** by replicating SGLang's
`is_layer_excluded` in a few lines of Python against the patched `config.json`'s
`ignore` list: `fused_qkv_a_proj_with_mqa` excluded (BF16) for every layer,
`o_proj`/`q_b_proj` W4A4 for the requant layers and BF16 for the MTP layer — 79
layers, 0 mismatches.

## 4. Validation results

### Phase 0 — synthetic GEMV benchmark (decision gate: GO)

Per-rank TP4 shapes, BF16-cuBLAS vs. the served NVFP4 path (offline FP4 weight +
dynamic activation FP4 + `cutlass_scaled_fp4_mm`, timing including activation
quant):

| shape | bs=1 | bs=8 | bs=32 |
|---|---|---|---|
| `o_proj` (out 6144, in 4096) | **5.58×** | 3.15× | 3.86× |
| `q_b_proj` (out 4096, in 2048) | **2.89×** | 0.91× | 1.22× |

Both bs=1 (the memory-bound decode floor) clear the ≥2.5× gate → **GO**. `q_b_proj`
drops below 1× from bs=8 (compute-bound + quant overhead); `o_proj` stays faster
even at batch. The synthetic rel-err here is not a quality proof — accuracy is
settled by the GSM8K gate below.

### GSM8K accuracy — PASS

5-shot, greedy (temp 0), n=200: **93.0%** flexible and strict (186/200), **0
errors / 0 empty responses**. A broken W4A4 (bad `input_scale`) would score <70%
or emit garbage — 93% with zero failures means the heuristic `input_scale` holds
and accuracy is **not** degraded. (The base's own GSM8K figure was measured with a
different harness/shot count, so this is a standalone go/no-go, not a paired A/B on
identical items.)

### Decode throughput — faster, as intended

Single-stream (batch 1) with MTP: measured **~18–29 tok/s** (accept length ~3–4
depending on prompt predictability) vs. the un-requantized base's **~11.7–12.4
tok/s** reference (accept ~2.1). Normalizing out MTP acceptance (tok/s ÷
accept-length = raw target forward-pass rate), the **requant-attributable speedup
is ~15–18% per forward pass** — the `o_proj`/`q_b_proj` NVFP4 GEMV win. The larger
end-to-end delta partly reflects higher MTP acceptance on the test prompts (the
base was not re-measured on identical prompts), so the clean requant-own number is
the ~15–18%.

### Loop/attractor gate — in progress

The base model documents an elevated loop/attractor (non-termination) rate that
its `recommended_sampling` (`min_p=0.05`, `repetition_penalty` 1.05→1.10)
recovers; the concern is whether NVFP4-rounding the KD-LoRA-bearing `o_proj`/`q_b`
worsens termination. Measured with the cluster's operational streaming
repetition-detector across three sampling conditions (raw / rec-1.05 / rec-1.10)
on open-ended prompts. **Result pending** — this section will be filled once the
sweep completes.

## 5. Reproduce

- **Requant:** a data-free streaming script quantizes only `o_proj`/`q_b_proj`
  over the requant layer range, writes the NVFP4 tensors + the range-gated
  ignore-list rewrite, and passes all other tensors through byte-identical.
- **Serving profile:**
  [`roles/k8s_dgx/model_profiles/vroomfondel-glm-5.2-reap-504b-v2-w4a4.yml`](roles/k8s_dgx/model_profiles/vroomfondel-glm-5.2-reap-504b-v2-w4a4.yml)
  — the exact SGLang launch config (DSA + MTP + `modelopt_fp4` + the runtime DSA
  patches this consumer-Blackwell stack needs).
- **Config-fix validation:** before deploying, replicate SGLang's
  `is_layer_excluded` against the patched `config.json`'s `ignore` list (§3) to
  confirm the fused-QKV-A and MTP layers land on the intended precision.

## 6. Provenance and credit

The base model `0xSero/glm-5.2-reap-504B-v2` (REAP expert pruning + Router-KD gate
recovery + logit-KD LoRA + NVFP4, from `zai-org/GLM-5.2`) is entirely 0xSero's
work; its base compute was sponsored by Lambda. Only the `o_proj`/`q_b_proj`
NVFP4 W4A4 requant, the config-surgery findings, and this report are ours. Not
affiliated with 0xSero. See the base model's `REPORT.md` for the full pruning +
recovery methodology and its honest loop-rate accounting.
