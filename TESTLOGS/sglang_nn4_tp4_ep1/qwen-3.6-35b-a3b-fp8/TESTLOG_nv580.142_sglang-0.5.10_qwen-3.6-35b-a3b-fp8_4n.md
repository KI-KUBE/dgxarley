# SGLang Test Log — Qwen3.6 35B-A3B-FP8 (MoE), 4 Nodes, TP=4 EP=1, v0.5.10

## Environment

| Component | Value                                              |
|-----------|----------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node     |
| Driver    | 580.142                                            |
| CUDA      | 13.2                                               |
| Kernel    | 6.19.13-custom                                     |
| OS        | Ubuntu 24.04 LTS (aarch64)                         |
| K3s       | v1.35.3+k3s1                                       |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)        |
| Image     | `scitrera/dgx-spark-sglang:0.5.10`                 |
| Model     | `Qwen/Qwen3.6-35B-A3B-FP8`                         |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
| Transport | **RoCE** via SR-IOV VF                             |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/nv580.142_sglang-0.5.10_qwen-3.6-35b-a3b-fp8_n4_ep1.yaml`

---

## Model Notes

- 35B total / 3B active **MoE** (Gated DeltaNet hybrid). Fine-grained FP8 (block 128).
- Architecture: 10 × (3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE)) = 40 layers.
  - Gated DeltaNet: 32 V-heads, 16 QK-heads, head_dim=128.
  - Gated Attention: 16 Q-heads, 2 KV-heads, head_dim=256, RoPE dim=64.
  - 256 routed experts (top-8) + 1 shared = 9 active per token, expert intermediate=512.
  - Hidden=2048, embedding/lm_head=248 320 (padded).
- Native context 262 144 (extensible to ~1 010 000 via YaRN).
- HF arch class: `Qwen3_5MoeForConditionalGeneration` (inherits `Qwen3VLForConditionalGeneration`).
  Same arch class SGLang 0.5.10 already supports for Qwen3.5 MoE → no new image needed.
- VL-fähig (Vision-Encoder), wir fahren rein Text — keine speziellen Flags.

## Known caveats inherited from the Qwen3.5 MoE codepath

- **EPLB stays off** (`enable_eplb: false`): `Qwen3_5MoeForConditionalGeneration`
  lacks `routed_experts_weights_of_layer` — same crash-after-~1000-passes bug
  documented for Qwen3.5-122B-A10B. Confirmed broken on 0.5.9-dev2; PR #19767's
  fix is not effective in our 0.5.10 build.
- **`moe_runner_backend: cutlass` skipped** — `cutlass_moe_fp4` requires FP4
  tensors. FP8 weights → only `triton` and `flashinfer_cutlass` are valid here.

## MTP / speculative decoding

Model card recommends NEXTN with:
```
--speculative-algo NEXTN --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```
SGLang 0.5.10 ships `qwen3_5_mtp.py` so this should be supported out of the box.
Tests 13–14 cover MTP under both MoE runners with `attention=flashinfer` (the
shape that won on Qwen3.5-397B and GLM-4.7 at EP=1). Profile keeps it disabled
(commented-out block) until validated here.

## Expected behaviour

- `moe_runner_backend=triton`: the safe default on SM121; matches the Qwen3.5
  MoE family's stable shape. Expected to be the workhorse across attn variants.
- `moe_runner_backend=flashinfer_cutlass`: at EP=1 on Qwen3.5-397B-NVFP4 the
  fi_cutlass MoE was stable; on FP8 weights the codepath is exploratory.
  Crashes here would mirror the FP4 EP>1 instability and would not be a
  showstopper — `triton` MoE remains the fallback.
- Eager (`disable_cuda_graph: true`) is expected to be slower but stable; the
  `cutlass_moe_fp4` eager-mode `!`-token-collapse only applies to FP4.
- Piecewise CUDA graphs: previously winning on the dense Gemma-4 path; on FP8
  MoE the picture is mixed in the Qwen3.5 family — wait for results.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.50, disable_deep_gemm=true, fp8_gemm_runner_backend=cutlass, context_length=262144, num_experts=256, enable_eplb=false` unless noted. FP8 → no FP4 sweep. `cutlass` MoE skipped (FP4-only).

| #  | nccl | moe_runner   | attention | dis_cuda_graph | dis_piecewise | spec | Status            | n=1 tok/s | n=4 peak | n=8 peak  |
|----|------|--------------|-----------|----------------|---------------|------|-------------------|-----------|----------|-----------|
| 1  | roce | triton       | fi        | false          | true          | —    | **STABLE**        | 68.6      | **214.7**| 344.1     |
| 2  | roce | triton       | fi        | true           | true          | —    | **STABLE**        | 21.0      | 102.7    | 207.0     |
| 3  | roce | triton       | fi        | false          | false         | —    | **STABLE**        | 62.4      | 210.4    | 336.3     |
| 4  | roce | triton       | triton    | false          | true          | —    | **STABLE**        | 69.0      | 213.3    | 343.2     |
| 5  | roce | triton       | triton    | true           | true          | —    | **STABLE**        | 20.5      | 105.1    | 205.5     |
| 6  | roce | triton       | triton    | false          | false         | —    | **STABLE ★**      | **69.0**  | 212.0    | **345.8** |
| 7  | roce | fi_cutlass   | fi        | false          | true          | —    | **startup_crash** | —         | —        | —         |
| 8  | roce | fi_cutlass   | fi        | true           | true          | —    | **bench_crash**   | —         | —        | —         |
| 9  | roce | fi_cutlass   | fi        | false          | false         | —    | **startup_crash** | —         | —        | —         |
| 10 | roce | fi_cutlass   | triton    | false          | true          | —    | **timeout**       | —         | —        | —         |
| 11 | roce | fi_cutlass   | triton    | true           | true          | —    | **bench_crash**   | —         | —        | —         |
| 12 | roce | fi_cutlass   | triton    | false          | false         | —    | **startup_crash** | —         | —        | —         |
| 13 | roce | triton       | fi        | false          | true          | NEXTN| **startup_crash** | —         | —        | —         |
| 14 | roce | fi_cutlass   | fi        | false          | true          | NEXTN| **startup_crash** | —         | —        | —         |

### Column Legend

| Column         | Description |
|----------------|-------------|
| nccl           | `nccl_transport` — NCCL inter-node transport (`roce` = RDMA via SR-IOV VF) |
| moe_runner     | `moe_runner_backend` — `triton` or `flashinfer_cutlass` (`fi_cutlass`) |
| attention      | `attention_backend` — attention kernel (`fi` = FlashInfer, `triton` = Triton) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |
| spec           | speculative decoding (`NEXTN` = MTP, num_steps=3, eagle_topk=1, num_draft_tokens=4) |

---

## Results

Run completed 2026-04-29, 14/14 cases finished (`kikube/matrixtest/2026-04-28/results/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/0.5.10/`).

**6/14 STABLE, 8/14 failed.** Clean split by MoE runner: every `triton`
MoE config works; every `flashinfer_cutlass` MoE config fails. Both MTP
variants fail at startup with the same root cause regardless of MoE runner.

### Tests 1–6 — triton MoE (all STABLE)

| Config | n=1 | n=4 | n=8 | n=1 TTFT |
|--------|----:|----:|----:|---------:|
| **Test 6** (triton attn, piecewise) | **69.0** | 212.0 | **345.8** | 0.48s |
| Test 1 (fi attn, CG on)             | 68.6     | **214.7** | 344.1 | **1.16s¹** |
| Test 4 (triton attn, CG on)         | 69.0     | 213.3 | 343.2 | 0.56s |
| Test 3 (fi attn, piecewise)         | 62.4     | 210.4 | 336.3 | 1.79s |
| Test 2 (fi attn, eager)             | 21.0     | 102.7 | 207.0 | 11.86s |
| Test 5 (triton attn, eager)         | 20.5     | 105.1 | 205.5 | 14.55s |

¹ first request of the day; warmup-cost noise.

- **CG-on configs (1, 3, 4, 6) all cluster within ~3% at n=8 (336–346 tok/s).**
  Test 6 wins by a hair (345.8) over Test 1 (344.1) — within run-to-run
  noise, no clear architectural advantage of triton-attn over fi-attn.
- **`attention_backend: flashinfer` works** — head_dim=256 on the gated-attn
  layer is in FlashInfer's dispatch table, no `head_dim=512` problem like Gemma-4.
- **Eager mode (2, 5) is 1.7×–3.3× slower** than CG-on at every batch size.
  CUDA graphs are mandatory for production throughput on this codepath.
- **Piecewise vs fixed-BS at n=4 split:** CG-on wins (214.7) over piecewise
  (210.4) when fi-attn; piecewise wins at n=8 by 1.7 tok/s. Pick CG-on for
  interactive (best n=4), piecewise for max-batch.
- n=8 peak (~346 tok/s) is **~3.4× the Qwen3.5-397B-NVFP4 winner** (102 tok/s)
  — small-active-MoE pays off (3B vs 17B active).

### Tests 7–12 — flashinfer_cutlass MoE (all FAIL)

All 6 fi_cutlass configs failed regardless of attention backend or CUDA-graph
setting. Distribution: 3× startup_crash, 2× bench_crash, 1× timeout. Common
root cause from the head log:

```
[TP0] Scheduler hit an exception:
AttributeError: 'Fp8MoEMethod' object has no attribute 'runner'
[TP0] Received sigquit from a child process. It usually means the child failed.
```

**`flashinfer_cutlass` MoE runner is incompatible with FP8 weights** — the
`Fp8MoEMethod` doesn't define a `runner` attribute that the fi_cutlass
codepath expects. This is consistent with the existing CLAUDE.md guidance
that `cutlass_moe_fp4` requires FP4 tensors. Profile correctly defaults to
`moe_runner_backend: triton`; fi_cutlass should not be used on FP8.

Test 8 (eager) and Test 11 (eager + triton attn) hit the same kernel error
at first benchmark request rather than during graph capture, hence
`bench_crash`. Test 10 timed out at startup (900s) — a slower variant of
the same crash. Test 12 was the lone server-side `sglang stuck` / pod restart.

### Tests 13–14 — MTP / NEXTN (both FAIL)

Both MTP variants startup-crashed with an explicit, actionable error from SGLang:

```
ValueError: Speculative decoding for Qwen3_5MoeForConditionalGeneration is
not compatible with radix cache when using --mamba-scheduler-strategy
no_buffer. To use radix cache with speculative decoding, please use
--mamba-scheduler-strategy extra_buffer and set SGLANG_ENABLE_SPEC_V2=1.
```

The model card's `--speculative-algo NEXTN ...` recipe is incomplete on
SGLang 0.5.10 for this hybrid-mamba arch. Two extra knobs are required:

- `mamba_scheduler_strategy: extra_buffer`
- `enable_spec_v2: true` (sets `SGLANG_ENABLE_SPEC_V2=1`)

Both keys already exist in `defaults/main.yml`. A follow-up matrix should
add these to the MTP cases (or a 4-case MTP-only sweep) and re-run; the
failure is configuration, not a code bug.

### Production profile recommendation

```yaml
moe_runner_backend: triton              # mandatory — fi_cutlass crashes on FP8
attention_backend: triton               # tied with flashinfer; triton is conservative
disable_cuda_graph: false               # mandatory — eager is 1.7×–3.3× slower
disable_piecewise_cuda_graph: false     # piecewise edges out CG-on at n=8 (Test 6)
cuda_graph_max_bs: 8
nccl_transport: roce
```

This is **Test 6**'s shape; it sits within noise of Tests 1/4 at every batch
size and wins n=8 outright.

### MTP re-run (2026-04-29) — both STABLE, big throughput win

Tests 13/14 re-run with the missing MTP knobs added (`mamba_scheduler_strategy:
extra_buffer` + `enable_spec_v2: true`). Test shape switched to the Test-6
winner layout (piecewise CG on, two attn variants). Result dir:
`kikube/matrixtest/2026-04-29/results/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/0.5.10/`.

Server args confirm the fix is wired: `mamba_scheduler_strategy='extra_buffer'`,
`speculative_algorithm='EAGLE'` (SGLang's internal name for NEXTN under
spec_v2). No startup crash.

| Test | moe_runner | attn   | spec  | n=1   | n=4   | n=8       | n=8 TTFT |
|------|------------|--------|-------|------:|------:|----------:|---------:|
| 13   | triton     | triton | NEXTN | **104.2** | 277.8 | **410.7** | **0.74s** |
| 14   | triton     | fi     | NEXTN | 85.8  | **284.3** | 409.9 | 0.97s    |
| 6 (no MTP, prior run) | triton | triton | —     | 69.0  | 212.0 | 345.8 | —        |

- **MTP delivers a massive boost over the no-MTP winner (Test 6):**
  **+51% at n=1** (104.2 vs 69.0), **+31% at n=4** (277.8 vs 212.0),
  **+19% at n=8** (410.7 vs 345.8). Higher gain at low concurrency is expected —
  speculative decoding hides latency, and at high batch the scheduler is
  already saturated.
- **Tests 13 and 14 tie at n=8** (410.7 vs 409.9, within noise). Test 13's
  triton-attn has the better n=1 (+22%) and lower n=8 TTFT (0.74 vs 0.97s);
  fi-attn edges ahead at n=4 (+2%). Triton-attn is the better default for
  interactive workloads.
- The card's claim of ~2× speedup is not reproduced here at n=8 (1.19×) but
  is closer at n=1 (1.51×). Plausible that benchmark-prompt mix matters.

### Recommended MTP profile (when enabled)

```yaml
moe_runner_backend: triton
attention_backend: triton
disable_cuda_graph: false
disable_piecewise_cuda_graph: false
nccl_transport: roce
cuda_graph_max_bs: 8
speculative_enabled: true
speculative_algo: NEXTN
speculative_num_steps: 3
speculative_eagle_topk: 1
speculative_num_draft_tokens: 4
mamba_scheduler_strategy: extra_buffer
enable_spec_v2: true
```

### Open follow-ups

- **27B-FP8 matrix** — kicked off on 2026-04-29 but no cases completed yet
  (`MATRIX_SUMMARY` file exists, empty results array).


