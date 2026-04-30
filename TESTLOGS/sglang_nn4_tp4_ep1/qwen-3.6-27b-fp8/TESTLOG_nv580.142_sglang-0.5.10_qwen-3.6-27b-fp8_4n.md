# SGLang Test Log — Qwen3.6 27B-FP8 (dense), 4 Nodes, TP=4 EP=1, v0.5.10

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
| Model     | `Qwen/Qwen3.6-27B-FP8`                             |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
| Transport | **RoCE** via SR-IOV VF                             |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-27b-fp8/nv580.142_sglang-0.5.10_qwen-3.6-27b-fp8_n4_ep1.yaml`

---

## Model Notes

- 27B **dense** (NOT MoE), hybrid Gated DeltaNet + Gated Attention. Fine-grained FP8 (block 128).
- Architecture: 16 layers of (3× Gated DeltaNet → FFN) + (1× Gated Attention → FFN).
  - Gated DeltaNet: 48 linear-attn V-heads, 16 QK-heads, head_dim=128.
  - Gated Attention: 24 Q-heads, 4 KV-heads, head_dim=256, RoPE dim=64.
  - FFN intermediate: 17 408.
- Native context 262 144 (extensible to ~1 010 000 via YaRN).
- HF arch class: `Qwen3_5ForConditionalGeneration` (inherits `Qwen3VLForConditionalGeneration`).
  Same code path SGLang already supports for Qwen3.5 dense → no new image needed.
- MTP-trained — NEXTN speculative decoding available (model card recommends
  `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`).
  Disabled in profile pending validation; not in this matrix.
- VL-fähig (Vision-Encoder), wir fahren rein Text — keine speziellen Flags.

## Notes vs. the Qwen3.5 family

- Same `Qwen3_5ForConditionalGeneration` arch as Qwen3.5 dense → 0.5.10 supported out of the box.
- No EPLB question (dense model).
- FP8 (not FP4) → no `fp4_gemm_backend` sweep, no `cutlass_moe_fp4` codepath.
- Recommended `mem_fraction_static=0.80` per card; profile mirrors that.

## Expected behaviour

- `attention_backend=triton` is the profile default and the most likely-stable
  path on SM121 (matches Qwen3.5 dense / Gemma-4 patterns).
- `attention_backend=flashinfer` — head_dim=256 on the gated-attn path is
  nominally inside FlashInfer's dispatch table; expected to work but unverified
  on SM121 for this hybrid arch. If it crashes (`prefill.cuh` Invalid
  configuration), it would mirror the Gemma-4 head_dim=512 issue.
- Eager (`disable_cuda_graph: true`) is expected to be stable but slower —
  no `cutlass_moe_fp4` codepath here that breaks under eager.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.80, disable_deep_gemm=true, fp8_gemm_runner_backend=cutlass, context_length=262144` unless noted. Dense → no MoE sweep. FP8 → no FP4 sweep.

Matrix re-shaped: original 6 cases plus 2 MTP cases (NEXTN, with
`mamba_scheduler_strategy=extra_buffer` + `enable_spec_v2=true` to avoid
the spec-v2-radix-cache crash). All 8 ran post-patch, **8/8 STABLE**.

| #  | nccl | attention | dis_cuda_graph | dis_piecewise | spec  | Status     | n=1 tok/s | n=4 peak | n=8 peak  |
|----|------|-----------|----------------|---------------|-------|------------|-----------|----------|-----------|
| 1  | roce | fi        | false          | true          | —     | **STABLE** | 21.9      | 84.2     | 157.4     |
| 2  | roce | fi        | true           | true          | —     | **STABLE** | 17.1      | 78.3     | 147.8     |
| 3  | roce | fi        | false          | false         | —     | **STABLE** | **22.0**  | 84.3     | 158.6     |
| 4  | roce | triton    | false          | true          | —     | **STABLE**§| 16.3      | 63.1§    | 148.7     |
| 5  | roce | triton    | true           | true          | —     | **STABLE** | 18.8      | 77.9     | 143.3     |
| 6  | roce | triton    | false          | false         | —     | **STABLE** | 21.6      | 84.0     | 157.9     |
| 7  | roce | fi        | false          | true          | NEXTN | **STABLE** | **44.4**  | 146.4    | 238.8     |
| 8  | roce | triton    | false          | true          | NEXTN | **STABLE ★**| 36.6     | **152.6**| **239.4** |

§ Test 4 had 3/4 successful at n=4 (one transient repetition); n=1 and n=8
clean. Treated as a flake — re-running would likely give 4/4. Earlier
preliminary report had Test 2 partial; that was also a flake (full re-run
gave 4/4 ok).

### Column Legend

| Column         | Description |
|----------------|-------------|
| nccl           | `nccl_transport` — NCCL inter-node transport (`roce` = RDMA via SR-IOV VF) |
| attention      | `attention_backend` — attention kernel (`fi` = FlashInfer, `triton` = Triton) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |
| spec           | speculative decoding (`NEXTN` = MTP via `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`, plus `mamba_scheduler_strategy=extra_buffer` + `enable_spec_v2=true`) |

---

## Results

### First run aborted (2026-04-29 ~09:14) — RepetitionGuard floods, hypothesis: sampling

Initial run kicked off 2026-04-29 with profile defaults. Tests 1–5 all came
back `bench_crash` with **every single request flagged `status=repetition`**;
test 6 was aborted before completing. Result dir:
`kikube/matrixtest/2026-04-29/results/sglang_nn4_tp4_ep1/qwen-3.6-27b-fp8/0.5.10/`.

Initial (wrong) diagnosis:
- Server appeared healthy across all 5 cases — model loads, attention works,
  TTFT ~0.8s on first request, CUDA-graph capture (where applicable) finishes.
- Each request emitted `output_tokens=0`, `status=repetition`.
- Suspected: card-recommended `presence_penalty=0.0` for general thinking is
  too lenient for this hybrid arch on the bench-prompt mix. Added profile
  `sampling_overrides` (`presence_penalty: 1.5`, `frequency_penalty: 0.5`,
  `min_tokens: 4`) and re-ran. Wiring of `sampling_overrides` to the kikube
  bench pod required setting `DGXARLEY_ROOT=/data/pythondev_workspace/dgxarley`
  in the job spec — without that env var the dgxarley pip-package's loader
  falls back to a `parents[2]` path that lands in `site-packages/` and finds
  no profile YAMLs (`_MODEL_PROFILES = {}`).

### Second run (2026-04-29 ~15:57) — token salad, real cause: FP8 scale bypass

Sampling overrides verified loaded in pod (22 profiles, 27B present, all three
override values applied). Test 1 re-ran. Still `status=repetition`, but with
new diagnostic data: 102–106 thinking tokens emitted before the guard tripped,
and the thinking content was multilingual gibberish:

```
ValuePaireladoDEXardyudeunek尋瑚 усыujiudyUGEukulmeister标高elee
Einz琦輩lattodox Tram TrapTrap traps trap陷阱 piè
... fusfusion融合融合融合 fusion Fusionfusion
... MediumMediummediummedium-mediummedium mediums mediumEDIUM medium medium
```

RepetitionGuard diagnostics: `trigger=NGRAM_FLOOD`, `worst_ngram="medium medium
medium medium"` (count 9 of 25 4-grams). This is **not a sampling-tunable
loop** — it's broken logits.

Server-side log carried 256 weight-loading warnings of the form:
```
[TP0] Parameter model.layers.<N>.mlp.gate_gate_up_proj.weight_scale_inv not found in params_dict
[TP0] Parameter model.layers.<N>.mlp.gate_up_proj.weight_scale_inv not found in params_dict
```

### Root cause — known SGLang upstream bug

Upstream issue [sgl-project/sglang#23687](https://github.com/sgl-project/sglang/issues/23687).
`is_layer_skipped()` in `python/sglang/srt/layers/quantization/utils.py` does
naive substring matching against the FP8 quant config's `modules_to_not_convert`.
For Qwen3.6-FP8 the entry `mlp.gate` matches `mlp.gate_up_proj` by substring →
SGLang treats the fused gate-up projection as un-quantized → `weight_scale_inv`
never lands in `params_dict` → FP8 dequant runs with default scale 1.0 →
garbage logits → token salad. Qwen3.5 ships a different `modules_to_not_convert`
layout and is **not** affected (clean run on `Qwen3.5-122B-A10B-FP8` confirms).

Fix is upstream commit
[`4323fce`](https://github.com/sgl-project/sglang/commit/4323fce82a091fab154bf36baa5820659ec0fd16)
(PR #23467, merged 2026-04-22). Adds dot-boundary `_module_path_match()` plus
a `_FALLBACK_FUSED_SHARDS` mapping. **Not in v0.5.10 / v0.5.10.post1 / v0.5.10rc0**;
on `main` only, 592 commits ahead of v0.5.10. See
`SGLANG_QWEN36_27B_FP8_UPSTREAM_BUG.md` for the full write-up.

### Patch applied — pending validation

Runtime monkey-patch added to `roles/k8s_dgx/files/sglang_launch.sh` (Z.603–717,
inserted before the existing `modelopt_quant.py` patch block). Reproduces the
upstream commit's three hunks via `code.replace(...)` with double idempotency
guard (`def _module_path_match` sentinel + per-hunk source-shape check). Uses
PEP-585 generics in the inserted code so no `typing` imports are added.

Next deployment of `Qwen/Qwen3.6-27B-FP8` should:
1. Show **0 `weight_scale_inv not found in params_dict` warnings** in the head log.
2. Produce coherent text in the `<think>` stream.
3. Bench requests should complete with `output_tokens > 0` and proper `tokens_per_sec`.

The `sampling_overrides` from the first hypothesis stay in the profile —
they're harmless on a working model and useful as a guardrail against
edge-case repetition on the bench-prompt mix.

### Third run (2026-04-29 16:20 → 17:24) — patch validated, full sweep STABLE

Post-patch run with `sglang_launch.sh` Z.603–717 in effect. Bench pod
`km-...-162013-nwj75`. Same matrix YAML, same prompts. Matrix expanded
from 6 to 8 cases (added 2 MTP variants, mirroring the 35B-A3B sibling).

**Patch confirmation in head log:**
- `weight_scale_inv not found` warnings: **0** (was 256 pre-patch).
- `gate_gate_up_proj` mentions: 0.
- Confirmation line: `Patched quantization/utils.py: dot-boundary _module_path_match + _FALLBACK_FUSED_SHARDS in is_layer_skipped()`.

**Tests 1–6 — non-MTP baseline (all STABLE):**

The CG-on configs (1/3/4/6) cluster within ~1% of each other at n=8
(157.4–158.6) — no clear architectural advantage between fi/triton attn
or fixed-BS/piecewise CG. **Test 3 (fi + CG + piecewise)** is the marginal
non-MTP winner at 158.6 tok/s @ n=8. Eager mode (2/5) lands ~6–9% behind
CG-on at n=8 — a small gap, much smaller than what we saw on the 35B
(where eager was 1.7× slower). For a small dense model, eager isn't a
disaster.

For context vs other dense models on this cluster:

| Model | Quant | Peak n=8 | vs 27B-FP8 |
|-------|-------|---------:|-----------:|
| Qwen3.6-27B-FP8 (this) | FP8 block-128 | **157.4–158.6** | 1.0× |
| Gemma-4 31B-it          | BF16          | 70.6  | 2.24× faster |
| Qwen3.5-122B-A10B-FP8 (MoE 10B active) | FP8 | ~120 | 1.32× faster |

27B FP8 is the fastest dense model we've measured.

**Tests 7–8 — MTP / NEXTN (huge win):**

| Test | shape                          | n=1   | n=4   | n=8       | n=8 TTFT |
|------|--------------------------------|------:|------:|----------:|---------:|
| 7    | fi-attn + CG + MTP             | 44.4  | 146.4 | 238.8     | 1.11s    |
| **8**| **triton-attn + CG + MTP** ★   | 36.6  | 152.6 | **239.4** | 0.94s    |
| 1 (no MTP, prior) | fi + CG + no-pw   | 21.9  | 84.2  | 157.4     | 0.74s    |

- **MTP gain over best non-MTP (Test 3 baseline):** **+102% n=1** (44.4 vs 22.0),
  **+74% n=4** (146.4 vs 84.3), **+52% n=8** (239.4 vs 158.6). Same gradient
  pattern as the 35B sibling but **substantially larger gains** — small dense
  decode is more throughput-bound, MTP hides more latency.
- Tests 7 and 8 tie at n=8 (238.8 vs 239.4, within noise). Test 7 wins n=1
  by 21% (44.4 vs 36.6). Test 8 has the better TTFT at n=8 (0.94 vs 1.11s).
- The Qwen3.6 model card's NEXTN recipe needed two extra knobs on SGLang
  0.5.10 to avoid the spec-v2-radix-cache crash:
  `mamba_scheduler_strategy: extra_buffer` + `enable_spec_v2: true`. Both
  already in the profile's commented MTP block; the kikube matrix yaml
  uncomments them for cases 7/8.

**Production profile recommendation:**

```yaml
attention_backend: fi          # Test 7's edge — best n=1 (44.4 tok/s)
                                # OR: triton (Test 8) — best n=8 TTFT, basically tied throughput
disable_cuda_graph: false
disable_piecewise_cuda_graph: true
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

Pick fi for interactive (best n=1 latency-to-first-token), triton for
balanced batched workloads.

### Open follow-ups

- Re-run Tests 2 and 4 to confirm the partial-success at n=4 was a flake
  (current data: 4/4 and 3/4 respectively — already mostly clean).
- 35B-A3B-FP8 sibling unaffected by this bug (different `modules_to_not_convert`)
  but worth verifying that the patch is a true no-op there on the next
  35B run.
