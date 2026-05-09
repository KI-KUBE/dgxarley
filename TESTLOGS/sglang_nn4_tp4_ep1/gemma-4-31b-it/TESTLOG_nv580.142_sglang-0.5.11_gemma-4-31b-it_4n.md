# SGLang Test Log — Gemma-4 31B-it (dense, BF16), 4 Nodes, TP=4 EP=1, v0.5.11

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
| Model     | `google/gemma-4-31B-it`                            |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
| Transport | **RoCE** via SR-IOV VF                             |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/gemma-4-31b-it/nv580.142_sglang-0.5.11_gemma-4-31b-it_n4_ep1.yaml`

Toolchain delta vs `_sglang-0.5.10_*` testlog: PyTorch 2.9 → 2.11, CUDA 13 default,
sgl-kernel 0.4.1.post1 → 0.4.2, FlashInfer 0.6.7.post2 → 0.6.8.post1. Native Gemma 4
support in 0.5.11. See `SGLANG_v0.5.11_VERSION_CHANGES.md`.

---

## Model Notes

- 30.7B **dense** (NOT MoE), multimodal (vision + text), native 256K context.
  We run text-only.
- Native Gemma 4 BF16 path in 0.5.11.
- Dense → no MoE-runner sweep; only attention × cuda_graph variants.

## Configuration Matrix (6 cases)

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, mem_fraction_static=0.50, context_length=262144`. BF16 dense → no MoE/FP4/FP8 sweep.

| #  | attention | dis_cuda_graph | dis_piecewise | Status | n=1 | n=4 | n=8 |
|----|-----------|----------------|---------------|--------|-----|-----|-----|
| 1  | fi        | false          | true          | tbd    | —   | —   | —   |
| 2  | fi        | true           | true          | tbd    | —   | —   | —   |
| 3  | fi        | false          | false         | tbd    | —   | —   | —   |
| 4  | triton    | false          | true          | tbd    | —   | —   | —   |
| 5  | triton    | true           | true          | tbd    | —   | —   | —   |
| 6  | triton    | false          | false         | tbd    | —   | —   | —   |

---

## Results

**Run pending.**

Result dir: `kikube/matrixtest/<DATE>/results/sglang_nn4_tp4_ep1/gemma-4-31b-it/0.5.11/`.

### Comparison to 0.5.10 baseline

Reference winners from `TESTLOG_nv580.142_sglang-0.5.10_gemma-4-31b-it_4n.md` —
populate after the 0.5.11 run. Key question: native Gemma 4 path vs the 0.5.10
transformers-fallback — throughput delta?

Note: Gemma-4 had a known head_dim=512 dispatch issue with FlashInfer-attn on
0.5.10. Re-verify whether fi-attn now works on 0.5.11 native path.
