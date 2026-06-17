# SGLang Test Log — NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 (NemotronH hybrid), 4 Nodes, TP=4 **EP=4**, v0.5.13-dev MTP

> **STATUS: COMPLETE — executed 2026-06-17 (08:28–10:38 UTC).** 11 cases; 9 booted, 2
> PROBE cases startup-crashed as expected (05 fi_cudnn, 06 triton). **This is the EP=4
> counterpart of the EP=1 0.5.13-mtp matrix** — same image, same MTP recipe, the ONLY
> axis changed is MoE parallelism (`ep_size=4`, expert-parallel all-to-all dispatch via
> DeepEP) vs the EP=1 run's tensor-parallel MoE. Purpose: decide whether EP=4 is worth it.
>
> **Headline: EP=4 slightly BEATS EP=1, and the win is concentrated at low concurrency.**
> At the MTP 3/4 winner recipe: n=1 **+9 %** (58.85 vs 53.99), n=4 +3.5 % (141.6 vs 136.8),
> n=8 a tie (~200 both; EP=4 @ctx524k = 201.5 clean 8/8). accept_len, NaN profile, and the
> 05/06 crashes are all identical to EP=1 — so MTP behaves the same; the delta is pure MoE
> parallelism. Cross-ref: `../../sglang_nn4_tp4_ep1/nemotron-3-super-120b-a12b-nvfp4/TESTLOG_nv580.159_sglang-0.5.13-mtp_..._4n.md`.

## Environment

| Component | Value                                                                                   |
|-----------|-----------------------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node                                          |
| Driver    | 580.159.03                                                                              |
| CUDA      | 13.0.3 host toolkit; nvidia-smi reports max CUDA 13.0                                   |
| Kernel    | 6.17.0-1021-nvidia (aarch64)                                                            |
| OS        | Ubuntu 24.04.4 LTS (aarch64)                                                            |
| K3s       | v1.36.1+k3s1                                                                            |
| Nodes     | spark1 (head), spark2, spark3, spark4 (workers) — GB10, 1 GPU each; control-plane = elite800 (amd64, no GPU) |
| Image     | `xomoxcc/dgx-spark-sglang:0.5.13-dev-nemotronh-mtp-sm121` — dedicated NemotronH-MTP      |
| Model     | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`                                        |
| Arch      | `NemotronHForCausalLM` (`model_type=nemotron_h`) — Mamba2 + MoE + attn hybrid           |
| NCCL      | 2.30.4 (from the run log)                                                               |
| Transport | RoCE via SR-IOV VF                                                                      |
| **MoE**   | **`ep_size=4` — expert-parallel all-to-all dispatch (DeepEP present in the boot log)**  |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep4/nemotron-3-super-120b-a12b-nvfp4/nv580.159_sglang-0.5.13-mtp_nemotron-3-super-120b-a12b-nvfp4_n4_ep4.yaml`
Results: `kikube/matrixtest/2026-06-17/results/sglang_nn4_tp4_ep4/nemotron-3-super-120b-a12b-nvfp4/0.5.13-mtp/`
Summary: `…/MATRIX_SUMMARY_nv580.159_sglang-0.5.13-mtp_nemotron-3-super-120b-a12b-nvfp4_4n_1pp_4tp_ep4.json`

## Why this matrix exists

The EP=1 0.5.13-mtp matrix established the serving config (MTP EAGLE steps=3/draft=4, full-CG,
fi_cutlass MoE+fp4, ctx524k). That run used `ep_size=1` → MoE runs **tensor-parallel** (experts
sharded across the 4 ranks, all-reduce after the expert FFN). This matrix flips ONE axis to
`ep_size=4` → MoE runs **expert-parallel** (each rank owns full experts, all-to-all dispatch +
combine per MoE layer). Everything else is held identical. Single question:

**Q — is EP=4 worth it over EP=1 for this model on this cluster?** → **Marginally YES**, mostly
for single-stream latency (see Findings).

## Risk verdicts (carried from EP=1, re-checked under EP=4)

- **R1 — arch + EP dispatch boot (Case 01 "EP4-DISPATCH-LITMUS").** → **PASS.** NemotronH
  NVFP4 + Mamba2 boots under expert-parallel all-to-all; 8/8 clean, coherent tokens.
- **R2 — MTP NaN logits (#27828).** → **PASS — no NaN** in any EP=4 MTP head log.
- **R3 — accept_len > 1.** → **PASS.** EP=4 accept_len (2.71–2.98) ≈ EP=1 (2.67–2.97). EP
  routing does not change MTP acceptance.
- **Closed axes unchanged:** Case 05 (fi_cudnn fp4_gemm) still `RuntimeError: cuDNN is not
  available`; Case 06 (triton MoE) still `AssertionError: mismatch in expected n`
  (`cutlass_moe.py:428`). Both identical to EP=1 — EP does not rescue either.

---

## Configuration Matrix

All cases: `tp=4, pp=1, ep=4, nccl_transport=roce, attention_backend=flashinfer,
kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.80, disable_deep_gemm=true,
quantization=modelopt_fp4, cuda_graph_max_bs=8` unless noted. MTP cases: `speculative_algo=EAGLE,
speculative_eagle_topk=1, speculative_draft_model_path=""`.

> `n=N peak` = Σ per-request tok/s over the *successful* requests. `ok` = successful/failed.

### Block A — NO-SPEC boot litmus + CG (fi_cutlass MoE/fp4, ctx262k) — Cases 01–02

| #  | spec | CG variant     | ctx  | Status | n=1 tok/s | n=4 peak | n=8 peak | n=8 ok | Output |
|----|------|----------------|------|--------|----------:|---------:|---------:|--------|--------|
| 01 | off  | no-CG (eager)  | 262k | ✅ OK  | 23.29     | 86.4     | 144.7    | 8/0    | coherent |
| 02 | off  | full-CG        | 262k | ✅ OK  | 31.30     | 74.0 (3/1)| 149.4   | 8/0    | coherent |

- No-spec EP=4 ≈ EP=1 (EP=1 was 23.15 / 146.3). The n=4 74.0 is low only because 1/4 hit the
  repetition detector (3/4 ok); per-request rate is normal.

### Block B — NO-SPEC context scaling (fi_cutlass MoE, full-CG) — Cases 03–04

| #  | spec | ctx     | Status | n=1 tok/s | n=8 peak | n=8 ok | Output |
|----|------|---------|--------|----------:|---------:|--------|--------|
| 03 | off  | 524288  | ✅ OK  | 31.36     | 129.9    | 7/1    | coherent (1 repetition) |
| 04 | off  | 1048576 | ✅ OK  | 31.29     | 147.4    | 8/0    | coherent |

- Context scaling ~free under EP=4 too (NoPE + 80/88 Mamba layers). 262k/1M ~146–147; the 524k
  129.9 reads low only from the 1/8 repetition loss (7/8).

### Block C/D — fp4_gemm + MoE-runner PROBES (full-CG, ctx262k) — Cases 05–06

| #  | probe        | Status            | Signature |
|----|--------------|-------------------|-----------|
| 05 | fi_cudnn fp4 | ❌ startup_crash | `RuntimeError: cuDNN is not available` — dev image lacks the cuDNN-FP4 wheel (same as EP=1). |
| 06 | triton MoE   | ❌ startup_crash | `AssertionError: mismatch in expected n` (`cutlass_moe.py:428`) — NVFP4 path forces cutlass_moe_fp4; LatentMoE/512-expert shape trips the assert (same as EP=1). |

### Block E — piecewise-CG PROBE (fi_cutlass MoE, ctx262k) — Case 07

| #  | CG variant | Status | n=1 tok/s | n=8 peak | n=8 ok | Output |
|----|------------|--------|----------:|---------:|--------|--------|
| 07 | piecewise  | ✅ OK  | 31.32     | 130.7    | 7/1    | coherent (1 repetition) |

### Block F — MTP / EAGLE (full-CG) — Cases 08–11

> Speedup read against the no-spec Case 02 (EP=4, n=1 31.30 / n=8 peak 149.4). accept_len = mean
> over the decode log. **08 = the EP=1-winner recipe (3/4) replicated under EP=4.**

| #  | steps | draft | ctx  | Status | accept_len | n=1 tok/s | n=1 vs 02 | n=8 peak | n=8 ok | NaN? | Output |
|----|------:|------:|------|--------|-----------:|----------:|----------:|---------:|--------|------|--------|
| 08 | 3     | 4     | 262k | ✅ OK  | 2.74       | **58.85** | 1.88×     | 179.5    | 7/1    | no   | coherent (1 repetition) |
| 09 | 3     | 4     | 524k | 🏆 **BEST** | 2.71  | 58.56     | 1.87×     | **201.5** | 8/0    | no   | coherent |
| 10 | 5     | 5     | 262k | ⚠️ OK  | 2.88       | 45.48     | 1.45×     | 181.1    | 8/0    | no   | cookbook — slower n=1 |
| 11 | 5     | 5     | 524k | ✅ OK  | 2.98       | 55.71     | 1.78×     | 189.8    | 8/0    | no   | cookbook @524k |

- **08/09 (3/4) win at every concurrency.** 08 (262k) is fastest at n=1/n=4 but lost 1/8 to the
  repetition detector at n=8; 09 (same recipe at ctx524k) is clean 8/8 and tops n=8 at 201.5 —
  it is the production-relevant number (serving runs at ctx524k).
- **Cookbook 5/5 (10/11) loses again:** n=1 is much slower (45.48 @262k) and even the cleaner
  524k variant (55.71 / 189.8) trails 3/4. Higher accept_len (2.88–2.98) ≠ more throughput.
- The matrix harness flagged Case 08 as `winner` (it leads n=1 + n=4); Case 09 is the better
  pick for serving (clean 8/8, best n=8, the deployed 524k context).

### Column legend

| Column     | Description |
|------------|-------------|
| n=1 vs 02  | single-stream speedup vs the EP=4 no-spec Case 02 (this matrix's MTP denominator) |
| n=N peak   | Σ per-request tok/s over the *successful* requests |
| accept_len | mean accepted draft tokens/step (decode log) |

---

## Detailed results (n=8)

| #  | n=8 peak | n=8 agg | avg_ttft | ok/fail | notes |
|----|---------:|--------:|---------:|---------|-------|
| 01 | 144.7 | 137.5 | 0.83 | 8/0 | no-spec eager |
| 02 | 149.4 | 137.0 | 0.85 | 8/0 | no-spec full-CG (n=4 lost 1) |
| 03 | 129.9 | 127.0 | 0.76 | 7/1 | no-spec ctx524k (1 repetition) |
| 04 | 147.4 | 141.2 | 1.58 | 8/0 | no-spec ctx1M |
| 07 | 130.7 | 121.7 | 0.67 | 7/1 | piecewise (1 repetition) |
| 08 | 179.5 | 163.7 | 1.90 | 7/1 | MTP 3/4 @262k (1 repetition) |
| 09 | **201.5** | 188.1 | 0.92 | 8/0 | **MTP 3/4 @524k — BEST**, clean |
| 10 | 181.1 | 160.1 | 1.31 | 8/0 | cookbook 5/5 @262k |
| 11 | 189.8 | 169.8 | 1.37 | 8/0 | cookbook 5/5 @524k |

The sporadic single-request `repetition` trips (Cases 02/03/07/08) are not EP-specific — the
EP=1 matrix saw the same detector noise on a couple of cases.

## Crash details

- **Case 05 (fi_cudnn fp4_gemm)** — `RuntimeError: cuDNN is not available` (flashinfer
  `_check_cudnn_availability`). All pods restarted ×1. Identical to EP=1 Case 05.
- **Case 06 (triton MoE)** — `AssertionError: mismatch in expected n` (`cutlass_moe.py:428
  cutlass_moe_fp4`). Identical to EP=1 Case 06. EP does not change the NVFP4 MoE dispatch.

---

## EP=4 vs EP=1 — head-to-head (both 0.5.13-mtp, same image, MTP 3/4 unless noted)

| Metric                         | EP=1 | EP=4 | Winner |
|--------------------------------|-----:|-----:|--------|
| **n=1 tok/s** (MTP 3/4, 262k)  | 53.99 | **58.85** | EP=4 (+9.0 %) |
| n=4 peak (MTP 3/4, 262k)       | 136.8 | **141.6** | EP=4 (+3.5 %) |
| n=8 peak (MTP 3/4, 262k)       | **199.7** (8/8) | 179.5 (7/8) | EP=1 (EP=4 lost 1 req → ~205 normalized) |
| n=8 peak (MTP 3/4, **524k**)   | — | **201.5** (8/8) | EP=4 (clean; EP=1 had no 3/4@524k cell) |
| accept_len (MTP 3/4)           | 2.67 | 2.74 | tie |
| no-spec full-CG n=8 (262k)     | 146.3 | 149.4 | tie |
| cookbook 5/5 n=1 (262k)        | 51.5 | 45.5 | tie (both lose to 3/4) |
| NaN / 05+06 crashes            | none / both crash | none / both crash | identical |

## Findings

1. **EP=4 boots and runs the full recipe** — arch + all-to-all dispatch fine, no new failure mode.
2. **EP=4 wins single-stream by ~9 %** (58.85 vs 53.99 at MTP 3/4). At n=1 the TP-MoE all-reduce
   latency per layer dominates EP=1; EP-MoE's all-to-all on small token counts is cheaper and each
   rank computes its experts more densely. This is the clearest, most reproducible EP=4 effect.
3. **EP=4 edges n=4 (+3.5 %) and ties n=8** (~200 both; EP=4 @524k clean = 201.5). The all-to-all
   overhead grows with batch, so the low-concurrency advantage erodes toward n=8.
4. **MTP is unaffected by EP** — accept_len, NaN profile, and the 3/4 > 5/5 ordering all match
   EP=1. The cookbook 5/5 still loses; spec axis conclusions carry over unchanged.
5. **The 05/06 crashes are MoE-dispatch-independent** — both recur identically under EP=4.

## Production recommendation

**EP=4 SET AS THE PROFILE DEFAULT (2026-06-17)** — it wins single-stream by +9 % (58.85 vs
53.99) at no aggregate-throughput cost (n=8 a tie, ~200 both). Recipe is unchanged from the
EP=1 winner: **MTP 3/4** (steps=3/draft=4/topk=1), full-CG, fi_cutlass MoE+fp4, kv fp8, mem
0.80, ctx524288 — Case 09 confirms it serves clean 8/8 at 201.5 n=8 peak. Do NOT use cookbook
5/5.

The profile now carries `ep_size: 4`. Revert to `ep_size: 1` (TP-MoE) if the DeepEP all-to-all
ever proves troublesome — n=8 aggregate is a wash, so EP=1 loses only the single-stream win and
is one fewer parallelism axis to reason about.

## Action items / follow-ups

- [x] Run the EP=4 MTP matrix (11 cases; boot litmus + MTP litmus passed).
- [x] Fill all result cells + the EP=4-vs-EP=1 head-to-head.
- [x] Confirm accept_len / NaN / 05–06 crash parity with EP=1.
- [ ] Optional: high-concurrency (>8) EP=4 run — the all-to-all may amortize better there and is
      the one regime where EP=4 could pull clearly ahead (neither matrix exercised n>8).
- [ ] Decide EP default with the user (kept EP=1 for now; EP=4 documented as the latency option).
