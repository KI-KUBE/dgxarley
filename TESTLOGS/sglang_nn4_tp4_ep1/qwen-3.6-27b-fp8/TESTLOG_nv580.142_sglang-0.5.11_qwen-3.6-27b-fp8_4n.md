# SGLang Test Log — Qwen3.6 27B-FP8 (dense), 4 Nodes, TP=4 EP=1, v0.5.11

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
| Model     | `Qwen/Qwen3.6-27B-FP8`                             |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
| Transport | **RoCE** via SR-IOV VF                             |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-27b-fp8/nv580.142_sglang-0.5.11_qwen-3.6-27b-fp8_n4_ep1.yaml`

Toolchain delta vs `_sglang-0.5.10_*` testlog: PyTorch 2.9 → 2.11, CUDA 13 default,
sgl-kernel 0.4.1.post1 → 0.4.2, FlashInfer 0.6.7.post2 → 0.6.8.post1. Spec V2 with
Overlap-Scheduling is now baseline (PR #21062). See `SGLANG_v0.5.11_VERSION_CHANGES.md`.

---

## Model Notes

- 27B **dense** (NOT MoE), hybrid Gated DeltaNet + Gated Attention. Fine-grained FP8 (block 128).
- Architecture: 16 layers of (3× Gated DeltaNet → FFN) + (1× Gated Attention → FFN).
  - Gated DeltaNet: 48 linear-attn V-heads, 16 QK-heads, head_dim=128.
  - Gated Attention: 24 Q-heads, 4 KV-heads, head_dim=256, RoPE dim=64.
  - FFN intermediate: 17 408.
- Native context 262 144 (extensible to ~1 010 000 via YaRN).
- HF arch class: `Qwen3_5MoeForConditionalGeneration`-style hybrid.
- Same hybrid-mamba arch family as Qwen3.6-35B-A3B-FP8 — **inherits the same
  word-salad concurrency-race observed there in v0.5.11** (see
  `qwen-3.6-35b-a3b-fp8/TESTLOG_..._sglang-0.5.11_*` Correctness Debug Sweep).
  Verify output quality manually for n=4 and n=8.

## Configuration Matrix (8 cases)

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.50, disable_deep_gemm=true, fp8_gemm_runner_backend=cutlass, context_length=262144`. Dense → no MoE-runner sweep. FP8 → no FP4 sweep. MTP cases (7–8) need `mamba_scheduler_strategy=extra_buffer + enable_spec_v2=true`.

| #  | attention | dis_cuda_graph | dis_piecewise | spec  | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|----------------|---------------|-------|--------|-----------|----------|----------|
| 1  | fi        | false          | true          | —     | tbd    | —         | —        | —        |
| 2  | fi        | true           | true          | —     | tbd    | —         | —        | —        |
| 3  | fi        | false          | false         | —     | tbd    | —         | —        | —        |
| 4  | triton    | false          | true          | —     | tbd    | —         | —        | —        |
| 5  | triton    | true           | true          | —     | tbd    | —         | —        | —        |
| 6  | triton    | false          | false         | —     | tbd    | —         | —        | —        |
| 7  | fi        | false          | false         | NEXTN | tbd    | —         | —        | —        |
| 8  | triton    | false          | false         | NEXTN | tbd    | —         | —        | —        |

---

## Results

**Run pending.**

Result dir: `kikube/matrixtest/<DATE>/results/sglang_nn4_tp4_ep1/qwen-3.6-27b-fp8/0.5.11/`.

### Comparison to 0.5.10 baseline

Reference winners from `TESTLOG_nv580.142_sglang-0.5.10_qwen-3.6-27b-fp8_4n.md`
(populate after 0.5.10 sweep results are recorded).

After the 0.5.11 run, populate the table above and add a delta section here.
Pay particular attention to **output quality at n=4 and n=8** — see
qwen-3.6-35b-a3b-fp8 testlog Correctness Debug Sweep for the Word-Salad pattern.
