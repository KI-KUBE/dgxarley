# SGLang Test Log — Gemma-4 26B-A4B-it NVFP4 (MoE), 4 Nodes, TP=4 EP=1, v0.5.11

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
| Image     | `xomoxcc/dgx-spark-sglang:0.5.11-gemma4-sm121`     |
| Model     | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`      |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
| Transport | **RoCE** via SR-IOV VF                             |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it-nvfp4/nv580.142_sglang-0.5.11_gemma-4-26b-a4b-it-nvfp4_n4_ep1.yaml`

Toolchain delta vs `_sglang-0.5.10_*` testlog: PyTorch 2.9 → 2.11, CUDA 13 default,
sgl-kernel 0.4.1.post1 → 0.4.2, FlashInfer 0.6.7.post2 → 0.6.8.post1.

**Image is the gemma4-patched SM121 build** because Gemma-4 NVFP4 still requires the
two unmerged source patches (PRs #22929 / #22928) on top of v0.5.11 — see
`SGLANG_GEMMA4_UPSTREAM_BUG.md` and `scripts/patches/sglang-0.5.11-gemma4-sm121.recipe`.

New `flashinfer_cutedsl` MoE backend (PR #21339) added in Tests 13–18 / 25–27.
See `SGLANG_v0.5.11_VERSION_CHANGES.md`.

---

## Model Notes

- 26B / 3.8B-active **MoE**, multimodal (vision + text), native 256K context.
  We run text-only. NVFP4 weight quantization (modelopt).
- Two stacked source patches required (PRs #22929 / #22928) for per-expert NVFP4
  weight loading and GEGLU + FP4 block scale NaN clamp on SM120/121.
- `cutlass_moe_fp4` requires the SM121 sgl-kernel patch (101 KB shared-mem fix).
- **Gemma-4 NVFP4 is exploratory on SM121** — neither upstream PRs are merged
  as of 2026-05-09; expect bench_crash / startup_crash on edge configs.

## Configuration Matrix (27 cases)

`triton` and `fi_cutlass` MoE × `fi`/`triton` attention × `fi_cutlass`/`fi_cudnn`
fp4_gemm × cuda_graph variants. Plus 3 `cutlass`-direct MoE cases (FP4 weights).

| #     | moe_runner   | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status | n=1 | n=4 | n=8 |
|-------|--------------|-----------|--------------|----------------|---------------|--------|-----|-----|-----|
| 1–6   | triton       | fi/triton | fi_cutlass   | …              | …             | tbd    | —   | —   | —   |
| 7–12  | triton       | fi/triton | fi_cudnn     | …              | …             | tbd    | —   | —   | —   |
| 13–18 | fi_cutlass   | fi/triton | fi_cutlass   | …              | …             | tbd    | —   | —   | —   |
| 19–24 | fi_cutlass   | fi/triton | fi_cudnn     | …              | …             | tbd    | —   | —   | —   |
| 25–27 | cutlass      | fi        | fi_cutlass   | mixed          | mixed         | tbd    | —   | —   | —   |

(Full matrix in YAML — 48 case entries; populate detailed table after run.)

---

## Results

**Run pending.**

Result dir: `kikube/matrixtest/<DATE>/results/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it-nvfp4/0.5.11/`.

### Comparison to 0.5.10 baseline

Reference: `TESTLOG_nv580.142_sglang-0.5.10_gemma-4-26b-a4b-it-nvfp4_4n.md` — populate after run.

Key questions:
- Do PRs #22929 / #22928 still apply cleanly on the v0.5.11 source tree?
- Does `cutlass_moe_fp4` on SM121 still need our shared-mem patch?
- Does `fi_cutedsl` MoE work on Gemma-4 NVFP4?
- Output-quality check (Word-Salad regression on hybrid-mamba is Qwen3.6-specific;
  Gemma-4 should not be affected).
