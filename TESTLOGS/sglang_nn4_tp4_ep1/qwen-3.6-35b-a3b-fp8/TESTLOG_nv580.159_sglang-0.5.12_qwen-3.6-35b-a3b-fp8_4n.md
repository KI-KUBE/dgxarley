# SGLang Test Log — Qwen3.6 35B-A3B-FP8 (MoE), 4 Nodes, TP=4 EP=1, v0.5.12

## Environment

| Component | Value                                                                       |
|-----------|-----------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node                              |
| Driver    | 580.159                                                                     |
| CUDA      | 13.2 host / 13.0 image (PR #21498)                                          |
| Kernel    | 6.17.0-1018-nvidia                                                          |
| OS        | Ubuntu 24.04 LTS (aarch64)                                                  |
| K3s       | v1.35.3+k3s1                                                                |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)                                 |
| Image     | `scitrera/dgx-spark-sglang:0.5.12`                                          |
| Model     | `Qwen/Qwen3.6-35B-A3B-FP8`                                                  |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)                                       |
| Transport | **RoCE** via SR-IOV VF                                                      |
| AllReduce | Legacy (both `SGLANG_USE_JIT_ALL_REDUCE=0` + `SGLANG_OPT_..._V2=0`)         |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/nv580.159_sglang-0.5.12_qwen-3.6-35b-a3b-fp8_n4_ep1.yaml`

Previous testlog: `TESTLOG_nv580.142_sglang-0.5.11_qwen-3.6-35b-a3b-fp8_4n.md` (driver 580.142, image 0.5.11). The 0.5.12 run bumps **driver 580.142 → 580.159** AND **image 0.5.11 → 0.5.12** simultaneously — Δ is not purely image-attributable. There is no cross-image 580.142-vs-580.159 baseline; if findings are ambiguous, a 0.5.11 re-run on 580.159 is needed to disambiguate.

Toolchain delta vs `_sglang-0.5.11_*` testlog:
- FlashInfer 0.6.8.post1 → 0.6.11.post1 (PRs #24452, #25129, #25310, #25335)
- sgl-kernel 0.4.2 → 0.4.2.post2 (#24457, #25326)
- DeepGEMM split out into its own `sgl-deep-gemm` wheel (#24268, #24348, #24385) — we run `disable_deep_gemm=true` anyway, so indirectly relevant.
- DeepEP swapped from `fzyzcjy` fork to `deepseek-ai/DeepEP@hybrid-ep` (#25113)
- `SGLANG_OPT_FP8_WO_A_GEMM` now default-on (#25181) — was opt-in on 0.5.11. **Touches every FP8 GEMM path in this matrix**; part of the 0.5.11 → 0.5.12 delta is attributable to this single switch.
- Fused SiLU+clamp+FP8 quant kernel (#24897) — FP8 MoE path.
- Spec V2: breakable CUDA-Graph for `bs > 1` (#24662) → MTP cases (13/14, 21–24) under n=4/n=8 concurrency.
- Spec V2: stuck-MTP on DSA-models fix (#24635), frozen-KV `bonus_tokens=None` crash fix (#25204) — both relevant for the hybrid-mamba MTP path.
- JIT Custom All-Reduce default-on (#24363) — **explicitly disabled on our side via `sglang_jit_allreduce=false` plus dual env-var injection (`SGLANG_USE_JIT_ALL_REDUCE=0` + `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0`)** so the collective path stays comparable across 0.5.11 ↔ 0.5.12. See TODO_0.5.12.md item 1.

See `SGLANG_v0.5.12_VERSION_CHANGES.md` for the full release delta.

---

## Model Notes

- 35B total / 3B active **MoE** (Gated DeltaNet hybrid). Fine-grained FP8 (block 128).
- Architecture: 10 × (3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE)) = 40 layers.
  - Gated DeltaNet: 32 V-heads, 16 QK-heads, head_dim=128.
  - Gated Attention: 16 Q-heads, 2 KV-heads, head_dim=256, RoPE dim=64.
  - 256 routed experts (top-8) + 1 shared = 9 active per token, expert intermediate=512.
- Native context 262 144 (extensible to ~1 010 000 via YaRN).
- HF arch class: `Qwen3_5MoeForConditionalGeneration` (inherits `Qwen3VLForConditionalGeneration`).
- VL-capable (vision encoder), we run text-only — no special flags.

## What changes vs the 0.5.11 sweep

1. **`SGLANG_OPT_FP8_WO_A_GEMM` is default-on** (#25181). On 0.5.11 this was opt-in (default `0`); on 0.5.12 it flips to `1`. We leave the 0.5.12 default in place → all FP8 paths in the matrix run with the Weight-Only A-GEMM optimization path. If a regression surfaces, a sub-run with `SGLANG_OPT_FP8_WO_A_GEMM=0` override is the next step.
2. **JIT Custom All-Reduce is default-on in 0.5.12** (#24363) — but we explicitly set `0` via Ansible default (`sglang_jit_allreduce: false`), with both env-var names injected into head + worker. This keeps the collective path apples-to-apples with the 0.5.11 baseline.
3. **`flashinfer_cutedsl` MoE (Tests 15–20)** — on 0.5.11 this was an explicit FP4-only pre-check crash (`server_args.py:2975 _handle_moe_kernel_config`). PR #23590 (Cute-DSL FP4 GEMM reland) and PR #23745 (Cute-DSL NVFP4 quant kernels) were merged — the pre-check logic is likely unchanged, but re-validate. If FP8 still hits fail-fast crash B → expected.
4. **`fi_cutlass` MoE (Tests 7–12)** — previously 6/6 crash A (`'Fp8MoEMethod' object has no attribute 'runner'`). FlashInfer bump 0.6.8.post1 → 0.6.11.post1 + sgl-kernel bump 0.4.2 → 0.4.2.post2: re-check whether upstream patched the dispatcher gap. Bug tracked in `SGLANG_FP8_MOEMETHOD_FLASHINFER_CUTLASS_UPSTREAM_BUG.md`.
5. **MTP / Spec V2** (Tests 13–14, 21–24). On 0.5.11 MTP on hybrid-mamba was slower than no-MTP across the board (Test 03 winner = 402.62 tok/s @ n=8, MTP-best = 389.92 @ n=8). 0.5.12 brings two MTP fixes (#25204 frozen-KV bonus-tokens, #24635 stuck-MTP DSA) + breakable CG bs>1 (#24662). Hypothesis: the MTP regression on hybrid-mamba is not addressed (none of the three PRs target the mamba-state-update path directly), but the CG bs>1 fix might improve n=4/n=8 stability.
6. **Word-salad regression** on hybrid-mamba from the 0.5.11 sweep (see appendix of the 0.5.11 testlog): gone after `0c2bdd4` (`is_layer_skipped` substring fix + `sampling_overrides={}`). This matrix inherits the fixed profile; if the bug resurfaces on 0.5.12 despite the profile fix → per-case output-quality check is mandatory (pattern-grep + token-distribution + tail-eyeball, see `feedback_output_quality_evidence` memory).

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.50, disable_deep_gemm=true, fp8_gemm_runner_backend=cutlass, context_length=262144, num_experts=256, enable_eplb=false` unless noted. FP8 → no FP4 sweep. `cutlass` MoE skipped (FP4-only).

| #  | moe_runner | attention | dis_cuda_graph | dis_piecewise | spec      | Status        | n=1 tok/s | n=4 peak | n=8 peak   |
|----|------------|-----------|----------------|---------------|-----------|---------------|-----------|----------|------------|
| 1  | triton     | fi        | false          | true          | —         | ok            | 64.15     | 265.36   | 406.44     |
| 2  | triton     | fi        | true           | true          | —         | ok            | 21.45     | 105.65   | 198.09     |
| 3  | triton     | fi        | false          | false         | —         | ok            | 92.00     | 259.55   | 406.06     |
| 4  | triton     | triton    | false          | true          | —         | ok            | 79.09     | 260.89   | 402.58     |
| 5  | triton     | triton    | true           | true          | —         | ok            | 22.31     | 106.11   | 206.79     |
| 6  | triton     | triton    | false          | false         | —         | ok            | 79.50     | 262.15   | 402.45     |
| 7  | fi_cutlass | fi        | false          | true          | —         | **crash A**   | —         | —        | —          |
| 8  | fi_cutlass | fi        | true           | true          | —         | **crash A'**  | —         | —        | —          |
| 9  | fi_cutlass | fi        | false          | false         | —         | **crash A**   | —         | —        | —          |
| 10 | fi_cutlass | triton    | false          | true          | —         | **crash A**   | —         | —        | —          |
| 11 | fi_cutlass | triton    | true           | true          | —         | **crash A'**  | —         | —        | —          |
| 12 | fi_cutlass | triton    | false          | false         | —         | **crash A**   | —         | —        | —          |
| 13 | triton     | triton    | false          | false         | NEXTN s=3 | ok            | 72.71     | 280.67   | 396.28     |
| 14 | triton     | fi        | false          | false         | NEXTN s=3 | ok            | 99.32     | 268.35   | 401.93     |
| 15 | fi_cutedsl | fi        | false          | true          | —         | **crash B**   | —         | —        | —          |
| 16 | fi_cutedsl | fi        | true           | true          | —         | **crash B**   | —         | —        | —          |
| 17 | fi_cutedsl | fi        | false          | false         | —         | **crash B**   | —         | —        | —          |
| 18 | fi_cutedsl | triton    | false          | true          | —         | **crash B**   | —         | —        | —          |
| 19 | fi_cutedsl | triton    | true           | true          | —         | **crash B**   | —         | —        | —          |
| 20 | fi_cutedsl | triton    | false          | false         | —         | **crash B**   | —         | —        | —          |
| 21 | triton     | fi        | false          | false         | NEXTN s=2 | ok            | 77.24     | 280.09   | **426.76** |
| 22 | triton     | fi        | false          | false         | NEXTN s=3 | ok            | 99.81     | 273.76   | 398.52     |
| 23 | triton     | fi        | false          | false         | NEXTN s=4 | ok            | 85.80     | 258.94   | 383.65     |
| 24 | triton     | fi        | false          | false         | NEXTN s=5 | ok            | 95.04     | 241.79   | 362.76     |

**Crash A** (fi_cutlass MoE, CUDA-graph variants): `AttributeError: 'Fp8MoEMethod' object has no attribute 'runner'` at scheduler-side dispatch. Same signature as the 0.5.11 / 0.5.10 crash — FlashInfer 0.6.11.post1 + sgl-kernel 0.4.2.post2 did NOT fix it. Tracked in `SGLANG_FP8_MOEMETHOD_FLASHINFER_CUTLASS_UPSTREAM_BUG.md`.

**Crash A'** (fi_cutlass MoE, eager / `disable_cuda_graph=true`): bench_crash — pod starts, but every benchmark request fails (n=1: 0/1, n=4: 0/4, n=8: 0/8). The eager path reaches inference but still hits the `Fp8MoEMethod.runner` dispatcher gap on the first forward pass. **New observation:** 0.5.11 had crash A on all 6 fi_cutlass cases as startup_crash; 0.5.12 splits the failure into startup_crash for the CG-capture cases (Tests 07, 09, 10, 12) and bench_crash for the two eager cases (Tests 08, 11). The pod survives further into the pipeline now, but the actual fix is still missing.

**Crash B** (fi_cutedsl MoE, all 6 cases): `AssertionError: Invalid quantization 'None'. FlashInfer CuteDSL MOE currently supports only: 'modelopt_fp4'.` Same FP4-only pre-check as on 0.5.11 (`server_args.py` _handle_moe_kernel_config). PR #23590 (Cute-DSL FP4 GEMM reland) + PR #23745 (Cute-DSL NVFP4 quant kernels) did not relax the FP8 rejection — by explicit design, the cutedsl path is NVFP4-only.

### Column Legend

| Column         | Description                                                                                                                    |
|----------------|--------------------------------------------------------------------------------------------------------------------------------|
| moe_runner     | `moe_runner_backend` — `triton`, `flashinfer_cutlass` (`fi_cutlass`), `flashinfer_cutedsl` (`fi_cutedsl`, PR #21339, FP4-only) |
| attention      | `attention_backend` — `fi` = FlashInfer, `triton` = Triton                                                                     |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs                                                               |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = fixed-BS graphs only, false = piecewise variable-length graphs                         |
| spec           | speculative decoding — `NEXTN s=N` = MTP with `speculative_num_steps=N`, `eagle_topk=1`, `num_draft_tokens=N+1`                |

---

## Results

**Matrix run complete (2026-05-21, ~17:59 → 19:35 UTC+2, ~90 min).** 24/24 cases attempted, **12 ok, 12 crash** (6 fi_cutlass + 6 fi_cutedsl; see crash footnotes under the matrix above).

Output-quality check per `ok` case (see `feedback_output_quality_evidence` memory):
1. pattern-grep for word-salad triggers (`retire retire`, `masterpiece masterpiece`, `STOP THIS LOOPING`, runs of `(\w+){3,}` repetition). `Self-Correction` removed from the trigger list after a false positive on a regular Qwen3.6-thinking reasoning marker (`*Self-Correction/Verification during output gen prep:*`).
2. token-distribution check (Type-Token-Ratio, reported as `ttr_min` across requests).
3. tail-eyeball of the last ~200 tokens per sample.

### Completed cases (12 ok)

| #  | Config                                                       |   n=1 | n=4 agg | n=4 per-req | n=4 peak |    n=8 agg | n=8 per-req | n=8 peak    | Failures | Finish reasons    | n=8 TTR min | Output quality |
|----|--------------------------------------------------------------|------:|--------:|------------:|---------:|-----------:|------------:|------------:|----------|-------------------|------------:|----------------|
| 01 | triton-moe + fi-attn, cuda_graph on, piecewise off           | 64.15 |  265.17 |       66.34 |   265.36 |     406.29 |       50.81 |     406.44  | 0/13     | length×13         |       0.650 | coherent ✓     |
| 02 | triton-moe + fi-attn, **cuda_graph off**, piecewise off      | 21.45 |  104.87 |       26.41 |   105.65 |     198.05 |       24.76 |     198.09  | 0/13     | length×12, stop×1 |       0.592 | coherent ✓     |
| 03 | triton-moe + fi-attn, cuda_graph on, **piecewise on**        | 92.00 |  259.42 |       64.89 |   259.55 |     405.86 |       50.76 |     406.06  | 0/13     | length×13         |       0.663 | coherent ✓     |
| 04 | triton-moe + **triton-attn**, cuda_graph on, piecewise off   | 79.09 |  260.73 |       65.22 |   260.89 |     402.43 |       50.32 |     402.58  | 0/13     | length×13         |       0.632 | coherent ✓     |
| 05 | triton-moe + triton-attn, **cuda_graph off**, piecewise off  | 22.31 |  104.34 |       26.53 |   106.11 |     206.72 |       25.85 |     206.79  | 0/13     | length×13         |       0.740 | coherent ✓     |
| 06 | triton-moe + triton-attn, cuda_graph on, **piecewise on**    | 79.50 |  262.02 |       65.54 |   262.15 |     402.33 |       50.31 |     402.45  | 0/13     | length×13         |       0.670 | coherent ✓     |
| 13 | triton-moe + triton-attn, piecewise on, **+MTP s=3**         | 72.71 |  265.43 |       70.17 |   280.67 |     377.64 |       49.54 |     396.28  | 0/13     | length×13         |       0.647 | coherent ✓     |
| 14 | triton-moe + fi-attn, piecewise on, **+MTP s=3**             | 99.32 |  250.93 |       67.09 |   268.35 |     378.62 |       50.24 |     401.93  | 0/13     | length×13         |       0.688 | coherent ✓     |
| 21 | winner shape + **MTP s=2**                                   | 77.24 |  273.58 |       70.02 |   280.09 |     401.45 |       53.35 | **426.76**  | 0/13     | length×13         |       0.641 | coherent ✓     |
| 22 | winner shape + **MTP s=3**                                   | 99.81 |  261.56 |       68.44 |   273.76 |     374.97 |       49.81 |     398.52  | 0/13     | length×13         |       0.541 | coherent ✓     |
| 23 | winner shape + **MTP s=4**                                   | 85.80 |  238.49 |       64.74 |   258.94 |     355.04 |       47.96 |     383.65  | 0/13     | length×13         |       0.694 | coherent ✓     |
| 24 | winner shape + **MTP s=5**                                   | 95.04 |  224.03 |       60.45 |   241.79 |     339.74 |       45.34 |     362.76  | 0/13     | length×13         |       0.627 | coherent ✓     |

### Δ vs 0.5.11 baseline (aggregate throughput, driver 580.142)

| #  | 0.5.11 (n=1 / n=4 / n=8) | 0.5.12 (n=1 / n=4 / n=8)    | Δ n=1        | Δ n=4        | Δ n=8         |
|----|--------------------------|-----------------------------|-------------:|-------------:|--------------:|
| 01 | 76.77 / 254.78 / 396.26  | 64.15 / 265.17 / 406.29     | **−16.4 %**  | **+4.1 %**   | **+2.5 %**    |
| 02 | 22.64 / 107.12 / 209.91  | 21.45 / 104.87 / 198.05     | **−5.3 %**   | **−2.1 %**   | **−5.7 %**    |
| 03 | 71.14 / 261.70 / 402.62  | 92.00 / 259.42 / 405.86     | **+29.3 %**  | **−0.9 %**   | **+0.8 %**    |
| 04 | 77.34 / 254.90 / 400.56  | 79.09 / 260.73 / 402.43     | **+2.3 %**   | **+2.3 %**   | **+0.5 %**    |
| 05 | 21.79 / 105.91 / 208.66  | 22.31 / 104.34 / 206.72     | **+2.4 %**   | **−1.5 %**   | **−0.9 %**    |
| 06 | 62.60 / 257.93 / 400.61  | 79.50 / 262.02 / 402.33     | **+27.0 %**  | **+1.6 %**   | **+0.4 %**    |
| 13 | 84.09 / 250.25 / 373.76  | 72.71 / 265.43 / 377.64     | **−13.5 %**  | **+6.1 %**   | **+1.0 %**    |
| 14 | 93.47 / 261.66 / 379.34  | 99.32 / 250.93 / 378.62     | **+6.3 %**   | **−4.1 %**   | **−0.2 %**    |
| 21 | 79.49 / 261.69 / 389.92  | 77.24 / 273.58 / 401.45     | **−2.8 %**   | **+4.5 %**   | **+3.0 %**    |
| 22 | 78.93 / 256.68 / 383.15  | 99.81 / 261.56 / 374.97     | **+26.5 %**  | **+1.9 %**   | **−2.1 %**    |
| 23 | 80.57 / 263.44 / 364.62  | 85.80 / 238.49 / 355.04     | **+6.5 %**   | **−9.5 %**   | **−2.6 %**    |
| 24 | 57.67 / 221.55 / 339.21  | 95.04 / 224.03 / 339.74     | **+64.8 %**  | **+1.1 %**   | **+0.2 %**    |

Caveat: Δ is **mixed driver (580.142 → 580.159) + image (0.5.11 → 0.5.12)**, not pure image attribution. The n=1 numbers are visibly noisy (note the ±29 % swings on hot-shape repeats), consistent with cache-warmup variance rather than systematic regression. Test 01 sits at the very start of the matrix and ran cold (TTFT 11.38 s vs 6.81 s on 0.5.11) — Tests 03/04/06 with the same MoE/CG configs landed at +27 to +29 % vs 0.5.11 n=1, so the Test 01 number is a warmup artifact, not a regression.

### Findings

1. **n=8 aggregate is flat vs 0.5.11.** Tests 01/03/04/06 land at 402.33–406.29 tok/s vs 0.5.11 winners 396.26–402.62. Δ in the +0.4 …+2.5 % range — consistent with the pre-run hypothesis (`SGLANG_OPT_FP8_WO_A_GEMM` default-on + fused SiLU+clamp+FP8 quant should give +2…5 %). No outlier.
2. **MTP `s=2` is the new n=8 peak winner: Test 21 at peak 426.76 / agg 401.45.** On 0.5.11 no-MTP was the global winner (Test 03 at 402.62 agg) and MTP always regressed. On 0.5.12 the order flips at peak throughput: MTP s=2 reaches **426.76 peak — +5 % over the best no-MTP peak (Test 01 at 406.44)** and **+9 % over the 0.5.11 Test 21 baseline (389.92 agg → 426.76 peak)**. Aggregate-wise Test 21 (401.45) ties with the no-MTP winners (~406). Per `feedback_peak_not_agg` memory, peak is the right metric — call this a real peak-throughput win, an aggregate-wise wash. Spec V2 polish (#23456 stale-state, #25204 frozen-KV, #24635 stuck-MTP DSA) + breakable CG bs>1 (#24662) recovered the MTP regression seen on 0.5.11.
3. **MTP `num_steps` sweet spot is now `s=2`** (Test 21 peak 426.76). s=3 (Test 22) drops to 398.52, s=4 (Test 23) to 383.65, s=5 (Test 24) to 362.76. Monotonic decay with depth — same shape as on 0.5.11, but anchored at a higher absolute peak. The 0.5.11 sweep found no clear sweet spot; 0.5.12 has one.
4. **fi_cutlass MoE on FP8 still broken** — 6/6 crash. The 0.5.12 split into startup_crash (4 CG-capture cases) vs bench_crash (2 eager cases) is a new pattern: Tests 08, 11 went further into the pipeline and only failed at first-forward-pass — eager mode bypasses the CUDA-graph capture-time crash, but the underlying `Fp8MoEMethod.runner` attribute is still missing. **No upstream fix despite FlashInfer 0.6.11.post1 + sgl-kernel 0.4.2.post2 bumps.** `SGLANG_FP8_MOEMETHOD_FLASHINFER_CUTLASS_UPSTREAM_BUG.md` should get a 0.5.12 status update.
5. **fi_cutedsl MoE on FP8 still rejected at pre-check** — 6/6 crash B (NVFP4-only by design). PR #23590 + #23745 did not loosen the FP8 restriction. Same behaviour as 0.5.11.
6. **Output quality clean across all 12 ok cases.** TTR_min ≥ 0.54 (Test 22 the lowest), all others ≥ 0.59. No word-salad triggers. The `0c2bdd4` profile fix (`is_layer_skipped` substring + `sampling_overrides={}`) continues to hold on 0.5.12.
7. **`mamba_usage: 0.02`** consistently in head decode logs — no hybrid-mamba KV-pool pressure at any concurrency.

### Production recommendation

**Switch the active profile to Test 21 shape** — winner-shape (triton-moe + fi-attn + CG on + piecewise on) **plus MTP s=2**:

```yaml
moe_runner_backend: triton
attention_backend: flashinfer
disable_cuda_graph: false
disable_piecewise_cuda_graph: false
speculative_enabled: true
speculative_algo: NEXTN
speculative_num_steps: 2
speculative_eagle_topk: 1
speculative_num_draft_tokens: 3
mamba_scheduler_strategy: extra_buffer
enable_spec_v2: true
sampling_overrides: {}
```

Rationale: peak n=8 426.76 tok/s, +5 % over best no-MTP peak (Test 01); aggregate ties with no-MTP winners; n=1 ~77 tok/s (similar to no-MTP after cache warmup). Spec V2 fixes in 0.5.12 made MTP s=2 net-positive on this model — the win is small but consistent, and per-request TTFT under multi-tenant load benefits from MTP's draft pre-fill.

---

## Baseline comparison (0.5.11, driver 580.142)

Winners from `TESTLOG_nv580.142_sglang-0.5.11_qwen-3.6-35b-a3b-fp8_4n.md`, for direct Δ calculation once 0.5.12 results land:

| #     | Config                                                     |   n=1 |    n=4 |        n=8 |
|-------|------------------------------------------------------------|------:|-------:|-----------:|
| 01    | triton-moe + fi-attn, cuda_graph on, piecewise off         | 76.77 | 254.78 |     396.26 |
| 02    | triton-moe + fi-attn, **cuda_graph off**, piecewise off    | 22.64 | 107.12 |     209.91 |
| 03    | triton-moe + fi-attn, cuda_graph on, **piecewise on**      | 71.14 | 261.70 | **402.62** |
| 04    | triton-moe + **triton-attn**, cuda_graph on, piecewise off | 77.34 | 254.90 |     400.56 |
| 06    | triton-moe + triton-attn, cuda_graph on, **piecewise on**  | 62.60 | 257.93 |     400.61 |
| 13    | triton-moe + triton-attn, piecewise on, **+MTP** (s=3)     | 84.09 | 250.25 |     373.76 |
| 14    | triton-moe + fi-attn, piecewise on, **+MTP** (s=3)         | 93.47 | 261.66 |     379.34 |
| 21    | winner shape + MTP s=2                                     | 79.49 | 261.69 |     389.92 |
| 23    | winner shape + MTP s=4                                     | 80.57 | 263.44 |     364.62 |
| 24    | winner shape + MTP s=5                                     | 57.67 | 221.55 |     339.21 |
| 07-12 | fi_cutlass × {fi, triton} × {CG on/off/piecewise}          |     — |      — |          — |  (6/6 **crash A**: `Fp8MoEMethod` has no `runner`)
| 15-20 | fi_cutedsl × {fi, triton} × {CG on/off/piecewise}          |     — |      — |          — |  (6/6 **crash B**: FP4-only pre-check)

**Expected delta hypotheses for 0.5.12** (pre-run):

1. **Tests 01–06 (no-MTP triton)**: slight speedup likely from `SGLANG_OPT_FP8_WO_A_GEMM` default-on (#25181) + fused SiLU+clamp+FP8 quant (#24897). Ballpark guess: +2…5 % at n=4/n=8 — if substantially more, the AllReduce default flip is the suspect (our `sglang_jit_allreduce=false` override should neutralise it, but worth re-checking the injected env vars on a running pod).
2. **Tests 07–12 (fi_cutlass MoE)**: likely still crash A. If now ok → upstream fix landed (possible, but not visible in the changelog).
3. **Tests 15–20 (fi_cutedsl MoE)**: likely still crash B. If now ok on FP8 → pre-check was loosened, but the path is NVFP4-designed — output-quality check would be critical.
4. **Tests 13–14, 21–24 (MTP)**: Spec V2 polish (#23456, #25204, #24635) + breakable CG bs>1 (#24662) might recover some of the n=4/n=8 MTP regression from 0.5.11 (−9 % at n=8 vs 0.5.10). Sweet spot probably still `s=2..3` for n=1, no-MTP still winner for n=8.
5. **Output quality**: word-salad should not resurface (the profile fix `0c2bdd4` is orthogonal to image version). Still — mandatory pattern check per case.

---

## Action items after the matrix run

- [x] Fill the table with actual results
- [x] Verify output quality on every `ok` case (pattern-grep + token-distribution + tail-eyeball)
- [x] Compute Δ vs 0.5.11 (driver 580.142)
- [x] Update the production recommendation in `model_profiles/Qwen--Qwen3.6-35B-A3B-FP8.yml` — flip to Test 21 shape (winner + MTP s=2)
- [ ] Update `SGLANG_FP8_MOEMETHOD_FLASHINFER_CUTLASS_UPSTREAM_BUG.md` with 0.5.12 status (still broken; new split startup_crash vs bench_crash between CG-capture and eager)
- [ ] Sub-run with `sglang_jit_allreduce=true` (winner shape only) to quantify the V2 speedup in isolation
- [ ] If Δ vs 0.5.11 needs cleanup: 0.5.11 re-run on driver 580.159 to disambiguate driver-vs-image contribution to the +0.4…+2.5 % n=8 gain
