# SGLang Test Log — Gemma-4 31B-it NVFP4 (dense), 4 Nodes, TP=4 EP=1, v0.5.11

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
| Model     | `nvidia/Gemma-4-31B-IT-NVFP4`                      |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
| Transport | **RoCE** via SR-IOV VF                             |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/gemma-4-31b-it-nvfp4/nv580.142_sglang-0.5.11_gemma-4-31b-it-nvfp4_n4_ep1.yaml`

Toolchain delta vs any prior 0.5.10 testlog: PyTorch 2.9 → 2.11, CUDA 13 default,
sgl-kernel 0.4.1.post1 → 0.4.2, FlashInfer 0.6.7.post2 → 0.6.8.post1.

**Image is the gemma4-patched SM121 build** because Gemma-4 NVFP4 still requires
the two unmerged source patches (PRs #22929 / #22928) on top of v0.5.11 — see
`SGLANG_GEMMA4_UPSTREAM_BUG.md`.

This is a **first-time test** for this model on this cluster (no 0.5.10 sibling
testlog in this directory).

See `SGLANG_v0.5.11_VERSION_CHANGES.md`.

---

## Model Notes

- 30.7B **dense** (NOT MoE), multimodal (vision + text), native 256K context.
  We run text-only. NVFP4 weight quantization (modelopt).
- Dense → no MoE-runner sweep; only attention × fp4_gemm × cuda_graph variants.
- Same NVFP4 / SM121 caveats as the 26B MoE NVFP4 sibling.

## Configuration Matrix (12 cases)

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, mem_fraction_static=0.50, context_length=262144, quantization=modelopt`. Dense → no MoE sweep.

| #  | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status | n=1 | n=4 | n=8 |
|----|-----------|--------------|----------------|---------------|--------|-----|-----|-----|
| 1  | fi        | fi_cutlass   | false          | true          | tbd    | —   | —   | —   |
| 2  | fi        | fi_cutlass   | true           | true          | tbd    | —   | —   | —   |
| 3  | fi        | fi_cutlass   | false          | false         | tbd    | —   | —   | —   |
| 4  | triton    | fi_cutlass   | false          | true          | tbd    | —   | —   | —   |
| 5  | triton    | fi_cutlass   | true           | true          | tbd    | —   | —   | —   |
| 6  | triton    | fi_cutlass   | false          | false         | tbd    | —   | —   | —   |
| 7  | fi        | fi_cudnn     | false          | true          | tbd    | —   | —   | —   |
| 8  | fi        | fi_cudnn     | true           | true          | tbd    | —   | —   | —   |
| 9  | fi        | fi_cudnn     | false          | false         | tbd    | —   | —   | —   |
| 10 | triton    | fi_cudnn     | false          | true          | tbd    | —   | —   | —   |
| 11 | triton    | fi_cudnn     | true           | true          | tbd    | —   | —   | —   |
| 12 | triton    | fi_cudnn     | false          | false         | tbd    | —   | —   | —   |

---

## Results

**Run pending — first-time test for this model.**

Result dir: `kikube/matrixtest/<DATE>/results/sglang_nn4_tp4_ep1/gemma-4-31b-it-nvfp4/0.5.11/`.

Key questions:
- Does the dense Gemma-4 NVFP4 path work on SM121 at all?
- `fi_cutlass-fp4` vs `fi_cudnn-fp4` GEMM backend — which wins on SM121?
- Output-quality check (Word-Salad regression is Qwen3.6-specific; Gemma-4
  should not be affected, verify anyway).
