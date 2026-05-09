# SGLang Test Log — Qwen3.6 35B-A3B-FP8 (MoE), 4 Nodes, TP=4 EP=1, v0.5.11

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
| Model     | `Qwen/Qwen3.6-35B-A3B-FP8`                         |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
| Transport | **RoCE** via SR-IOV VF                             |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/nv580.142_sglang-0.5.11_qwen-3.6-35b-a3b-fp8_n4_ep1.yaml`

Toolchain delta vs `_sglang-0.5.10_*` testlog: PyTorch 2.9 → 2.11, CUDA 13 default,
sgl-kernel 0.4.1.post1 → 0.4.2, FlashInfer 0.6.7.post2 → 0.6.8.post1. Spec V2 with
Overlap-Scheduling is now baseline (PR #21062). New `flashinfer_cutedsl` MoE
backend (PR #21339) added in Tests 15–20. See `SGLANG_v0.5.11_VERSION_CHANGES.md`.

---

## Model Notes

- 35B total / 3B active **MoE** (Gated DeltaNet hybrid). Fine-grained FP8 (block 128).
- Architecture: 10 × (3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE)) = 40 layers.
  - Gated DeltaNet: 32 V-heads, 16 QK-heads, head_dim=128.
  - Gated Attention: 16 Q-heads, 2 KV-heads, head_dim=256, RoPE dim=64.
  - 256 routed experts (top-8) + 1 shared = 9 active per token, expert intermediate=512.
- Native context 262 144 (extensible to ~1 010 000 via YaRN).
- HF arch class: `Qwen3_5MoeForConditionalGeneration` (inherits `Qwen3VLForConditionalGeneration`).
- VL-fähig (Vision-Encoder), wir fahren rein Text — keine speziellen Flags.

## What changes vs the 0.5.10 sweep

1. **`flashinfer_cutedsl` MoE runner is new** (Tests 15–20). On 0.5.10 only
   `triton`, `cutlass`, `flashinfer_cutlass` existed; `cutlass_moe_fp4` is
   FP4-only and `flashinfer_cutlass` was 6/6 startup_crash on FP8 due to
   `Fp8MoEMethod.runner` missing. Open question: does the new cutedsl backend
   work on FP8 weights? If yes, it's a fourth option besides `triton`.
2. **Spec V2 + Overlap-Scheduling is default** (PR #21062). MTP cases (13–14)
   should benefit from the lower per-step CPU overhead. The
   `mamba_scheduler_strategy=extra_buffer` + `enable_spec_v2=true` knobs are
   still required for hybrid-mamba radix-cache compat.
3. **FlashInfer 0.6.8.post1 + sgl-kernel 0.4.2** under the hood. Tests 7–12
   (fi_cutlass MoE) are re-runnable to see if the FP8 incompatibility
   (`Fp8MoEMethod has no attribute 'runner'`) was fixed by the lib bumps.

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.50, disable_deep_gemm=true, fp8_gemm_runner_backend=cutlass, context_length=262144, num_experts=256, enable_eplb=false` unless noted. FP8 → no FP4 sweep. `cutlass` MoE skipped (FP4-only).

| #  | moe_runner       | attention | dis_cuda_graph | dis_piecewise | spec  | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|------------------|-----------|----------------|---------------|-------|--------|-----------|----------|----------|
| 1  | triton           | fi        | false          | true          | —     | tbd    | —         | —        | —        |
| 2  | triton           | fi        | true           | true          | —     | tbd    | —         | —        | —        |
| 3  | triton           | fi        | false          | false         | —     | tbd    | —         | —        | —        |
| 4  | triton           | triton    | false          | true          | —     | tbd    | —         | —        | —        |
| 5  | triton           | triton    | true           | true          | —     | tbd    | —         | —        | —        |
| 6  | triton           | triton    | false          | false         | —     | tbd    | —         | —        | —        |
| 7  | fi_cutlass       | fi        | false          | true          | —     | tbd    | —         | —        | —        |
| 8  | fi_cutlass       | fi        | true           | true          | —     | tbd    | —         | —        | —        |
| 9  | fi_cutlass       | fi        | false          | false         | —     | tbd    | —         | —        | —        |
| 10 | fi_cutlass       | triton    | false          | true          | —     | tbd    | —         | —        | —        |
| 11 | fi_cutlass       | triton    | true           | true          | —     | tbd    | —         | —        | —        |
| 12 | fi_cutlass       | triton    | false          | false         | —     | tbd    | —         | —        | —        |
| 13 | triton           | triton    | false          | false         | NEXTN | tbd    | —         | —        | —        |
| 14 | triton           | fi        | false          | false         | NEXTN | tbd    | —         | —        | —        |
| 15 | fi_cutedsl       | fi        | false          | true          | —     | tbd    | —         | —        | —        |
| 16 | fi_cutedsl       | fi        | true           | true          | —     | tbd    | —         | —        | —        |
| 17 | fi_cutedsl       | fi        | false          | false         | —     | tbd    | —         | —        | —        |
| 18 | fi_cutedsl       | triton    | false          | true          | —     | tbd    | —         | —        | —        |
| 19 | fi_cutedsl       | triton    | true           | true          | —     | tbd    | —         | —        | —        |
| 20 | fi_cutedsl       | triton    | false          | false         | —     | tbd    | —         | —        | —        |

### Column Legend

| Column         | Description |
|----------------|-------------|
| moe_runner     | `moe_runner_backend` — `triton`, `flashinfer_cutlass` (`fi_cutlass`), or **new** `flashinfer_cutedsl` (`fi_cutedsl`, PR #21339) |
| attention      | `attention_backend` — `fi` = FlashInfer, `triton` = Triton |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |
| spec           | speculative decoding (`NEXTN` = MTP, num_steps=3, eagle_topk=1, num_draft_tokens=4 + extra_buffer + spec_v2) |

---

## Results

**Run pending** — fill in after matrix execution.

Result dir: `kikube/matrixtest/<DATE>/results/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/0.5.11/`.

### Comparison to 0.5.10 baseline

Reference winners from `TESTLOG_nv580.142_sglang-0.5.10_qwen-3.6-35b-a3b-fp8_4n.md`:

| Config | n=1 | n=4 | n=8 |
|--------|----:|----:|----:|
| Test 6 (triton MoE + triton attn + piecewise on, no MTP) | 69.0 | 212.0 | 345.8 |
| Test 13 (triton MoE + triton attn + piecewise on + MTP) — winner | **104.2** | **277.8** | **410.7** |

After the 0.5.11 run, populate the table above and write a short delta vs 0.5.10
section here (toolchain bump impact + cutedsl viability + MTP under default
Spec V2 + Overlap-Scheduling).
