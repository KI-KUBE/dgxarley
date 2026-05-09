# SGLang Test Log — Gemma-4 26B-A4B-it (MoE, BF16), 4 Nodes, TP=4 EP=1, v0.5.11

## Environment

| Component | Value                                              |
|-----------|----------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node     |
| Driver    | 580.142                                            |
| CUDA      | 13.2 host / 13.0 image (PR #21498)                 |
| Kernel    | 6.19.13-custom                                     |
| OS        | Ubuntu 24.04 LTS (aarch64)                         |
| K3s       | v1.35.3+k3s1                                       |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)        |
| Image     | `scitrera/dgx-spark-sglang:0.5.11`                 |
| Model     | `google/gemma-4-26B-A4B-it`                        |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
| Transport | **RoCE** via SR-IOV VF                             |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it/nv580.142_sglang-0.5.11_gemma-4-26b-a4b-it_n4_ep1.yaml`

Toolchain delta vs `_sglang-0.5.10_*` testlog: PyTorch 2.9 → 2.11, CUDA 13 default,
sgl-kernel 0.4.1.post1 → 0.4.2, FlashInfer 0.6.7.post2 → 0.6.8.post1.
**Gemma 4 is now native in SGLang 0.5.11** (PR #21952 + follow-ups #22079, #24048,
#22842 — see [cookbook](https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4)).
The `0.5.10-20260429-gemma4-sm121-dev1`-Patch image becomes redundant once we move
to a 0.5.11-based custom image. New `flashinfer_cutedsl` MoE backend (PR #21339)
added in Tests 13–18. See `SGLANG_v0.5.11_VERSION_CHANGES.md`.

---

## Model Notes

- 26B / 3.8B-active **MoE**, multimodal (vision + text), native 256K context.
  We run text-only.
- Native Gemma 4 path in 0.5.11 — the `transformers`-fallback monkey-patch from
  `sglang_launch.sh` is redundant for BF16 on this image.

## What changes vs the 0.5.10 sweep

1. **Native Gemma 4 path** (no transformers fallback) — codepath is materially different.
2. **`flashinfer_cutedsl` MoE runner** — 4th MoE backend option.
3. **Spec V2 + Overlap-Scheduling default** (PR #21062) — Hybrid-mamba models (this
   one is dense-attn but VLM, not hybrid-mamba) should be unaffected by the
   Word-Salad concurrency-race observed on Qwen3.6-35B-A3B (`Qwen3_5MoeForConditionalGeneration`
   arch). Verify output quality at n=4 / n=8 anyway.
4. **PCG + fused RMSNorm + Residual-Add + Scalar** for Gemma-4 VLM (PR #24048).
5. **`gemma_weight` precomputed** to skip redundant per-forward add (PR #22673).

## Configuration Matrix (18 cases)

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, mem_fraction_static=0.50, context_length=262144`. BF16 → no FP4/FP8 GEMM sweep. `cutlass` MoE skipped (FP4-only).

| #  | moe_runner   | attention | dis_cuda_graph | dis_piecewise | Status | n=1 | n=4 | n=8 |
|----|--------------|-----------|----------------|---------------|--------|-----|-----|-----|
| 1  | triton       | fi        | false          | true          | tbd    | —   | —   | —   |
| 2  | triton       | fi        | true           | true          | tbd    | —   | —   | —   |
| 3  | triton       | fi        | false          | false         | tbd    | —   | —   | —   |
| 4  | triton       | triton    | false          | true          | tbd    | —   | —   | —   |
| 5  | triton       | triton    | true           | true          | tbd    | —   | —   | —   |
| 6  | triton       | triton    | false          | false         | tbd    | —   | —   | —   |
| 7  | fi_cutlass   | fi        | false          | true          | tbd    | —   | —   | —   |
| 8  | fi_cutlass   | fi        | true           | true          | tbd    | —   | —   | —   |
| 9  | fi_cutlass   | fi        | false          | false         | tbd    | —   | —   | —   |
| 10 | fi_cutlass   | triton    | false          | true          | tbd    | —   | —   | —   |
| 11 | fi_cutlass   | triton    | true           | true          | tbd    | —   | —   | —   |
| 12 | fi_cutlass   | triton    | false          | false         | tbd    | —   | —   | —   |
| 13 | fi_cutedsl   | fi        | false          | true          | tbd    | —   | —   | —   |
| 14 | fi_cutedsl   | fi        | true           | true          | tbd    | —   | —   | —   |
| 15 | fi_cutedsl   | fi        | false          | false         | tbd    | —   | —   | —   |
| 16 | fi_cutedsl   | triton    | false          | true          | tbd    | —   | —   | —   |
| 17 | fi_cutedsl   | triton    | true           | true          | tbd    | —   | —   | —   |
| 18 | fi_cutedsl   | triton    | false          | false         | tbd    | —   | —   | —   |

---

## Results

**Run pending.**

Result dir: `kikube/matrixtest/<DATE>/results/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it/0.5.11/`.

### Comparison to 0.5.10 baseline

Reference winners from `TESTLOG_nv580.142_sglang-0.5.10_gemma-4-26b-a4b-it_4n.md` —
populate after the 0.5.11 run. Key questions:
- Native Gemma 4 path vs the 0.5.10 transformers-fallback — throughput delta?
- `fi_cutedsl` viability on this BF16 MoE model.
- Output quality at n=4 / n=8 (Word-Salad regression check; expected unaffected
  on this arch but verify).
