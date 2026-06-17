# SGLang Test Log — NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 (NemotronH hybrid), 4 Nodes, TP=4 EP=1, v0.5.13-dev MTP

> **STATUS: COMPLETE — executed 2026-06-16 (17:17–21:33 UTC).** All 14 cases ran;
> 12 booted, 2 PROBE cases startup-crashed as predicted (Cases 05 fi_cudnn, 06 triton).
> **Headline: MTP finally pays off on NemotronH-NVFP4 on this image — EAGLE steps=3/draft=4
> (Case 09) wins at 1.70× n=1 / 1.37× n=8 over the no-spec baseline, accept_len ≈ 2.7,
> zero NaN. It BEATS the NVIDIA cookbook 5/5 recipe (Case 10), which both ran slower AND
> lost a request to the repetition detector.** The 0.5.12 first-contact TESTLOG
> (`TESTLOG_nv580.159_sglang-0.5.12_..._4n.md`) is the no-MTP reference.

## Environment

| Component | Value                                                                                   |
|-----------|-----------------------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node                                          |
| Driver    | 580.159.03  *(verified on spark1–4, 2026-06-16 — matches the `nv580.159` filename label)* |
| CUDA      | 13.0.3 host toolkit (`/usr/local/cuda-13.0`); nvidia-smi reports max CUDA 13.0          |
| Kernel    | 6.17.0-1021-nvidia (aarch64)                                                            |
| OS        | Ubuntu 24.04.4 LTS (aarch64)                                                            |
| K3s       | v1.36.1+k3s1                                                                            |
| Nodes     | spark1 (head), spark2, spark3, spark4 (workers) — GB10, 1 GPU each; control-plane = elite800 (amd64, no GPU) |
| Image     | `xomoxcc/dgx-spark-sglang:0.5.13-dev-nemotronh-mtp-sm121` — **dedicated NemotronH-MTP** |
| Model     | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`                                        |
| Arch      | `NemotronHForCausalLM` (`model_type=nemotron_h`) — Mamba2 + MoE + attn hybrid           |
| Quant     | `modelopt_mixed` / `MIXED_PRECISION` (expert FFN FP4 g16, attn/latent/MTP/emb FP8/BF16) |
| NCCL      | **2.30.4** *(from the boot log of the actual run — note: the image on-node previously carried 2.30.7)* |
| Transport | **RoCE** via SR-IOV VF                                                                  |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/nemotron-3-super-120b-a12b-nvfp4/nv580.159_sglang-0.5.13-mtp_nemotron-3-super-120b-a12b-nvfp4_n4_ep1.yaml`
Results: `kikube/matrixtest/2026-06-16/results/sglang_nn4_tp4_ep1/nemotron-3-super-120b-a12b-nvfp4/0.5.13-mtp/`
Summary: `…/MATRIX_SUMMARY_nv580.159_sglang-0.5.13-mtp_nemotron-3-super-120b-a12b-nvfp4_4n_1pp_4tp_ep1.json`
EP=4 sibling: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep4/nemotron-3-super-120b-a12b-nvfp4/nv580.159_sglang-0.5.13-mtp_nemotron-3-super-120b-a12b-nvfp4_n4_ep4.yaml`

### Why a new image (the "extra image")

MTP on NemotronH-NVFP4 needs a build carrying the June-2026 MTP fixes; the upstream
`scitrera:0.5.12` base used for the first-contact run does NOT. This run pins the dedicated
`xomoxcc/dgx-spark-sglang:0.5.13-dev-nemotronh-mtp-sm121` (now also the profile's
`sglang_image:`). Relevant upstream (researched 2026-06-16):

- **#24955** "Support Nemotron DP attention and MTP" — MERGED 2026-06-12
- **#28102** "Fix DP attention + EP mode of Nemotron" — MERGED 2026-06-13
- **#27184** "fix Nemotron Super MTP deploy (spec-v2+B200)" — MERGED 2026-06-03
- **#27998** "[NemotronH] MTP with radix cache" — GB10-VALIDATED on THIS model (98% prefix
  reuse, MTP active, no regression); removes the `--disable-radix-cache` requirement and
  auto-selects mamba `extra_buffer` for spec+radix.

**These fixes are confirmed present in this image:** MTP boots, `accept_len > 1` on every
MTP case (loader fix #21138/#27998 landed — see R3), and no `--disable-radix-cache` was
needed.

---

## Why this matrix exists — two questions, one run

The 0.5.12 first-contact run left MTP out of scope (two blockers) and ran on a different
image. This run answers:

1. **Q1 — Non-MTP image delta (Blocks A–E, `speculative_enabled=false`):** does
   0.5.13-dev-nemotronh-mtp run the Super better/differently than 0.5.12 **even without
   MTP**? → **YES, ~8% faster no-spec** (Case 03 ctx524k: 145.8 vs 0.5.12's 135.1 n=8 peak).
2. **Q2 — MTP payoff (Blocks F–G):** does EAGLE MTP finally pay off on NemotronH-NVFP4, and
   at which draft depth? → **YES. Best = steps=3/draft=4 (Case 09): 1.70× n=1, 1.37× n=8.**

## Dominant risks / success criteria — VERDICTS

- **R1 — arch boot on the new dev image (Case 01).** → **PASS.** `NemotronHForCausalLM`
  NVFP4 + Mamba2 SM121 loads and emits coherent tokens on 0.5.13-dev. Case 01 booted clean
  (8/8, eager). Since 01 booted, every dependent case booted too.
- **R2 — MTP NaN logits (#27828).** → **PASS — no NaN anywhere.** Grepped every MTP head
  log (Cases 08–14): zero NaN/Inf-logits events. The NVFP4 MTP target-logits path is clean
  on this image.
- **R3 — accept rate (#21138 / #27998).** → **PASS — loader fix landed.** Every MTP case
  has `accept_len > 1` (mean 1.81 → 3.07 across cases), NOT the old ≈0.33-bug `accept≈1`.
  Higher draft depth lifts accept_len (5/5 mean 2.97, 5/7 mean 2.93) but NOT net throughput.

## Closed axes (re-probed on the new image, confirmed)

- **attention = flashinfer ONLY** — triton attn hard-asserted off on NemotronH. Confirmed.
- **MoE runner = flashinfer_cutlass** — triton STILL startup-crashes (Case 06: same
  `cutlass_moe_fp4` shape assert as 0.5.12). Not optional.
- **fp4_gemm = flashinfer_cutlass** — fi_cudnn STILL absent (Case 05: `RuntimeError: cuDNN
  is not available`). The dev image does NOT ship the cuDNN-FP4 wheel either.
- **piecewise CUDA graph = off** (card flag; Case 07 boots and is marginally fastest no-spec
  at n=8 147.8, but the card sets `--disable-piecewise-cuda-graph` and full-CG is within
  noise — kept off).
- **context** 262k/524k/1M all boot at ~equal throughput; 524288 serving default.
  `mem_fraction_static=0.80` comfortable; MTP draft buffers + 512K KV co-fit (Case 12).

---

## Configuration Matrix

All cases: `tp=4, pp=1, ep=1, nccl_transport=roce, attention_backend=flashinfer,
kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.80, disable_deep_gemm=true,
quantization=modelopt_fp4, cuda_graph_max_bs=8` unless noted. MTP cases: `speculative_algo=EAGLE,
speculative_eagle_topk=1, speculative_draft_model_path=""` (built-in MTP layer).

CG variant encoding:
- **no-CG** : `disable_cuda_graph=true,  disable_piecewise_cuda_graph=true` (eager)
- **full-CG** : `disable_cuda_graph=false, disable_piecewise_cuda_graph=true` (serving)
- **piecewise** : `disable_cuda_graph=false, disable_piecewise_cuda_graph=false` (PROBE only)

> **`n=N peak` = Σ per-request tok/s over the *successful* requests** (NOT aggregate
> total_tokens/wall_time). `n=8 ok` = successful/failed.

### Block A — NO-SPEC boot litmus + CG (fi_cutlass MoE, fi_cutlass-fp4, ctx262k) — Cases 01–02

| #  | spec | CG variant     | ctx  | Status | n=1 tok/s | n=4 peak | n=8 peak | n=8 ok | Output |
|----|------|----------------|------|--------|----------:|---------:|---------:|--------|--------|
| 01 | off  | no-CG (eager)  | 262k | ✅ OK  | 23.15     | 85.6     | 140.4    | 8/0    | coherent |
| 02 | off  | full-CG        | 262k | ✅ OK  | 31.67     | 95.0     | 146.3    | 8/0    | coherent |

- Eager (01) is ~27% slower at n=1 than full-CG (02). Full-CG is the no-spec serving baseline
  and the MTP denominator.

### Block B — NO-SPEC context scaling (fi_cutlass MoE, full-CG) — Cases 03–04

| #  | spec | ctx     | json_override                         | Status | n=1 tok/s | n=8 peak | n=8 ok | Output |
|----|------|---------|---------------------------------------|--------|----------:|---------:|--------|--------|
| 03 | off  | 524288  | `{"max_position_embeddings":524288}`  | ✅ OK  | 30.22     | 145.8    | 8/0    | coherent |
| 04 | off  | 1048576 | `{"max_position_embeddings":1048576}` | ✅ OK  | 31.58     | 146.1    | 8/0    | coherent |

- Context scaling is ~free (NoPE + 80/88 Mamba layers → KV barely grows): 262k/524k/1M all
  land at ~146 n=8 peak. Confirms the 0.5.12 finding on the new image.

### Block C — NO-SPEC fp4_gemm delta (fi_cutlass MoE, full-CG, ctx262k) — Case 05

| #  | fp4_gemm | Status            | Note |
|----|----------|-------------------|------|
| 05 | fi_cudnn | ❌ startup_crash | `RuntimeError: cuDNN is not available` (`flashinfer/gemm/gemm_base.py` `_check_cudnn_availability`). The dev image does NOT ship `nvidia-cudnn-cu12` either — same gap as 0.5.12. |

### Block D — NO-SPEC MoE runner PROBE (full-CG, ctx262k) — Case 06

| #  | moe_runner | Status            | Note |
|----|------------|-------------------|------|
| 06 | triton     | ❌ startup_crash | `AssertionError: mismatch in expected n` (`cutlass_moe.py:428 cutlass_moe_fp4`). The triton flag is ignored on the NVFP4 modelopt path → always dispatches `cutlass_moe_fp4` → LatentMoE/512-expert shape trips the assert. Same crash as 0.5.12. |

### Block E — NO-SPEC piecewise-CG PROBE (fi_cutlass MoE, ctx262k) — Case 07

| #  | CG variant | Status | n=1 tok/s | n=8 peak | n=8 ok | Output |
|----|------------|--------|----------:|---------:|--------|--------|
| 07 | piecewise  | ✅ OK  | 30.69     | 147.8    | 8/0    | coherent |

- Piecewise boots and is marginally the fastest no-spec at n=8 (147.8 vs 146.3 full-CG) — but
  within run-to-run noise, and the card sets `--disable-piecewise-cuda-graph`. Kept off.

### Block F — MTP / EAGLE draft-depth sweep (fi_cutlass, full-CG, ctx262k) — Cases 08–11

> Speedup read against Block-A Case 02 (no-spec, same image, n=1 31.67 / n=8 peak 146.3).
> **accept_len > 1 met on all (R3 PASS).** accept_len = mean over the decode log.

| #  | steps | draft | Status | accept_len | n=1 tok/s | n=1 vs 02 | n=8 peak | n=8 vs 02 | n=8 ok | NaN? | Output |
|----|------:|------:|--------|-----------:|----------:|----------:|---------:|----------:|--------|------|--------|
| 08 | 1     | 2     | ✅ OK  | 1.81       | 49.15     | 1.55×     | 183.0    | 1.25×     | 8/0    | no   | coherent |
| 09 | 3     | 4     | 🏆 **WIN** | 2.67   | **53.99** | **1.70×** | **199.7** | **1.37×** | 8/0    | no   | coherent |
| 10 | 5     | 5     | ⚠️ OK  | 2.97       | 51.50     | 1.63×     | 152.7    | 1.04×     | 7/1    | no   | 1 repetition (req 2) |
| 11 | 5     | 7     | ✅ OK  | 2.93       | 53.21     | 1.68×     | 175.1    | 1.20×     | 8/0    | no   | coherent |

- **09 (3/4) is the overall winner** — fastest at BOTH n=1 (53.99) and n=8 (199.7), 8/8 clean.
- **10 = the NVIDIA cookbook 5/5 recipe LOSES here:** higher accept_len (2.97) does NOT
  translate to throughput — the extra draft-compute per step eats the win, AND req 2 was
  flagged `repetition` (7/8), collapsing n=8 peak to 152.7 (barely above no-spec).
- **11 (5/7, the TRT accept-3.45 point)** boots clean but the deeper draft costs net
  throughput (175.1 < 199.7). Bigger draft → more accept but more wasted draft compute.

### Block G — MTP serving-context + robustness PROBES (cookbook 5/5) — Cases 12–14

| #  | variant                          | ctx    | Status | accept_len | n=1 tok/s | n=8 peak | n=8 ok | Output |
|----|----------------------------------|--------|--------|-----------:|----------:|---------:|--------|--------|
| 12 | serving memory                   | 524288 | ✅ OK  | 2.83       | 45.50     | 180.7    | 8/0    | coherent |
| 13 | `mamba_scheduler_strategy=extra_buffer` | 262144 | ⚠️ OK | 3.07 | (n1 fail) | 176.6    | 8/0    | n=1 repetition |
| 14 | `enable_spec_v2=true`            | 262144 | ✅ OK  | 2.95       | 52.84     | 177.3    | 8/0    | coherent |

- **12 — MTP + 512K KV co-fit: PASS.** 5/5 draft buffers + 524k KV boot and serve (180.7 n=8,
  8/8). n=1 is lower (45.5) at long context, but the bigger serving context is viable with MTP.
  → A 3/4 build (lighter draft buffers than this 5/5 probe) fits 524k with more headroom.
- **13 — manual `extra_buffer`: no help, mild harm.** accept_len highest (3.07) but n=1 hit the
  repetition detector (0/1) and n=8 (176.6) is below the 3/4 winner. #27998 auto-selects
  extra_buffer for spec+radix; setting it manually is unnecessary and showed a degenerate n=1.
  Leave it OFF.
- **14 — spec-v2: works, no win.** Boots clean (52.84 n=1, 177.3 n=8) but does not beat the
  default spec path at 3/4. Not in the base recipe; leave OFF.

### Column legend

| Column     | Description                                                                                                     |
|------------|-----------------------------------------------------------------------------------------------------------------|
| spec       | `speculative_enabled` — off (no-spec baseline) / EAGLE MTP                                                       |
| accept_len | mean accepted draft tokens per step (decode log) — **> 1 = MTP paying off; ≈ 1 = loader bug (R3)**               |
| n=1 vs 02  | single-stream speedup vs the no-spec Case 02 on this image (the MTP denominator)                                 |
| n=N peak   | **peak** throughput = Σ per-request tok/s over the *successful* requests (NOT aggregate total_tokens/wall_time) |
| NaN? (R2)  | did the NVFP4 MTP target-logits path emit NaN at boot/decode? (`yes` = the #27828 gap)                          |

---

## Detailed results (n=8)

All booting cases ran 8 concurrent requests, 3072 max output tokens, default sampling preset.
Finish reasons are `length` (hit cap) / `stop` (natural EOS); `repetition` = harness
repetition-detector trip (counted as failed).

| #  | n=8 peak | n=8 agg | avg_ttft | ok/fail | finish reasons | notes |
|----|---------:|--------:|---------:|---------|----------------|-------|
| 01 | 140.4 | 139.3 | 0.67 | 8/0 | length/stop | eager baseline |
| 02 | 146.3 | 130.0 | 0.77 | 8/0 | length/stop | no-spec serving baseline |
| 03 | 145.8 | 137.2 | 1.48 | 8/0 | length/stop | ctx524k |
| 04 | 146.1 | 146.1 | 0.70 | 8/0 | length/stop | ctx1M |
| 07 | 147.8 | 133.0 | 0.76 | 8/0 | length/stop | piecewise (fastest no-spec, within noise) |
| 08 | 183.0 | 173.3 | 1.48 | 8/0 | length/stop | MTP 1/2 |
| 09 | **199.7** | 182.4 | 0.93 | 8/0 | length/stop | **WINNER** MTP 3/4 |
| 10 | 152.7 | 143.2 | 1.12 | 7/1 | length/stop + 1 repetition | cookbook 5/5 — degraded |
| 11 | 175.1 | 159.3 | 1.37 | 8/0 | length/stop | MTP 5/7 |
| 12 | 180.7 | 162.1 | 1.55 | 8/0 | length/stop | MTP 5/5 @ ctx524k |
| 13 | 176.6 | 159.6 | 0.93 | 8/0 | length/stop | extra_buffer (n=1 separately failed) |
| 14 | 177.3 | 156.7 | 1.46 | 8/0 | length/stop | spec-v2 |

Output spot-check (winner Case 09, req 1): coherent long-form technical prose, natural `stop`
at 3008 tokens, no looping/degeneration.

## Crash details

- **Case 05 (fi_cudnn fp4_gemm)** — `[TP0] Scheduler hit an exception` →
  `RuntimeError: cuDNN is not available. Please install cuDNN to use FP8 GEMM functions`
  (`flashinfer/gemm/gemm_base.py:2011 _check_cudnn_availability`). All 4 pods restarted ×1.
  Identical to the 0.5.12 Case 05 — the dev image still lacks the `nvidia-cudnn-cu12` wheel.
- **Case 06 (triton MoE)** — `[TP0] Scheduler hit an exception` →
  `AssertionError: mismatch in expected n` (`sglang/srt/layers/moe/cutlass_moe.py:428
  cutlass_moe_fp4`). The triton runner flag is ignored on the NVFP4 modelopt path, which
  always routes the FFN through `cutlass_moe_fp4`; the LatentMoE/512-expert shape trips the
  hard assert. Identical to the 0.5.12 Case 06.

No MTP case crashed; no NaN-logit events in any MTP head log.

---

## Findings

1. **R1 — arch boots on 0.5.13-dev-nemotronh-mtp: YES.** Case 01 loaded NVFP4 + Mamba2 SM121
   and emitted coherent tokens.
2. **Q1 — non-MTP image delta: ~8% faster.** Best no-spec on the new image is ~146–148 n=8
   peak (Cases 02/04/07); the 0.5.12 winner (Case 03 ctx524k) was 135.1. The Case 03 ctx524k
   apples-to-apples comparison: 145.8 vs 135.1 = +8%. Cases 05 (fi_cudnn) and 06 (triton)
   did NOT change verdict — both still startup-crash with the same signatures.
3. **R3 — MTP pays off: YES.** `accept_len > 1` on every MTP case (the #21138/#27998 loader
   fix is in this image). Best draft depth = **steps=3/draft=4** (Case 09): accept_len ≈ 2.7,
   1.70× n=1 (53.99 vs 31.67) and 1.37× n=8 (199.7 vs 146.3) over no-spec Case 02. Deeper
   drafts (5/5, 5/7) raise accept_len but LOSE net throughput and risk repetition.
4. **R2 — NVFP4 MTP NaN logits: none.** #27828's gap is not present on this image.
5. **Memory — MTP + 512K KV co-fit: YES** (Case 12, 5/5 @ 524k, 8/8). The lighter 3/4 winner
   fits 524k with more headroom → serving at ctx524288 with MTP 3/4 is safe.
6. **extra_buffer / spec_v2 probes:** neither helps. `extra_buffer` (13) gave a degenerate
   n=1 (repetition); `spec_v2` (14) boots but doesn't beat the default path. Both stay OFF.
   #27998 auto-selects extra_buffer for spec+radix, so it never needs manual setting.
7. **Best overall serving shape:** **MTP EAGLE steps=3 / draft=4 / topk=1, full-CG,
   fi_cutlass MoE + fi_cutlass fp4_gemm, flashinfer attn, kv fp8_e4m3, mem 0.80, ctx524288.**
   The cookbook 5/5 is explicitly NOT recommended here — it is slower and lost a request to
   repetition on this model/image.

## Production recommendation

Serve with **EAGLE MTP, `speculative_num_steps=3`, `speculative_num_draft_tokens=4`,
`speculative_eagle_topk=1`** on `xomoxcc/dgx-spark-sglang:0.5.13-dev-nemotronh-mtp-sm121`,
full-CG, fi_cutlass MoE + fp4_gemm, flashinfer attn, kv fp8_e4m3, `mem_fraction_static=0.80`,
`context_length=524288` (`max_position_embeddings` override). Expected ≈1.70× single-stream
and ≈1.37× at 8-way concurrency over the no-spec baseline, accept_len ≈ 2.7, no NaN.

**Do NOT use the NVIDIA cookbook 5/5 recipe on this model/image** — it is slower than 3/4
and tripped the repetition detector (Case 10). Keep `enable_spec_v2` and a manual
`mamba_scheduler_strategy=extra_buffer` OFF (Cases 13/14 showed no benefit).

> The model profile `roles/k8s_dgx/model_profiles/nvidia-nvidia-nemotron-3-super-120b-a12b-nvfp4.yml`
> has been set to this winner config (2026-06-16).

## Action items / follow-ups

- [x] Run the matrix (14 cases; Cases 01 boot litmus + 08 MTP litmus passed first).
- [x] Fill every result cell + Findings / Detailed-results / Crash sections.
- [x] Compare against the 0.5.12 TESTLOG (image delta: +8% no-spec).
- [x] MTP pays off → profile MTP block set to the validated 3/4 winner with accept_len ≈ 2.7.
- [x] Driver/kernel/OS/K3s verified on spark1–4 (2026-06-16); NCCL = 2.30.4 (from the run log).
- [ ] Compare against the EP=4 sibling run (`sglang_nn4_tp4_ep4/.../0.5.13-mtp/`) — separate analysis.
- [ ] Optional: re-probe `cuda_graph_max_bs`/`max_mamba_cache_size` for >8-way concurrency
      (this matrix only exercised n≤8 at `cuda_graph_max_bs=8`; the serving profile carries
      96/32 for headroom, still unvalidated above 8 parallel).
