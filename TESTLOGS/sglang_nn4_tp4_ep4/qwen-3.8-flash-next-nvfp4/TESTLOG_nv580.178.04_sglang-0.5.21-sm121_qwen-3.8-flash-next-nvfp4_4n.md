# SGLang Test Log: Qwen3.8-Flash-Next-NVFP4 (NVIDIA export), 4 Nodes, TP=4 EP=4, v0.5.21-sm121, MTP sweep

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (SM121), 128 GB unified per node, driver 580.178.04 (head node `nvidia-smi`), kernel 7.0.0-1019-nvidia |
| Nodes | 4x DGX Spark: node1 (head/rank0), node2, node3, node4 (1 GB10 each) |
| Image | `xomoxcc/dgx-spark-sglang:0.5.21-sm121` (pinned in profile) |
| Model | `nvidia/Qwen3.8-Flash-Next-NVFP4` (quant `modelopt_mixed`, MIXED_PRECISION; FP8 in-checkpoint MTP head) |
| Transport | RoCE via SR-IOV VF |
| Parallel | tp=4, pp=1, ep=4; `moe_runner_backend: flashinfer_cutlass`, MTP draft MoE `triton` |
| Date | 2026-10-10 |

Profile: `roles/k8s_dgx/model_profiles/nvidia-qwen3.8-flash-next-nvfp4.yml`. Predecessor measurement (0.5.20-sm121, 2026-09-23, MTP 3/1/4): see the table in the profile header.
Cluster context: the Nemotron-3-Super default and the Nano-Omni vision instance were replaced by this model (`enable_vision_model: false`, `_sglang_default_instance: {}`); LiteLLM redeployed with `litellm_vision_alias_fallback_to_base: true`. No other instance shared the GPUs during the runs.

## Method

One Ansible rollout per case (`ansible-playbook k8s_dgx.yml --tags sglang`; case 01 with `--tags sglang,litellm`), profile restored from a saved copy before each edit. Bench via `sglang-gsm8k run` (chat API, GSM8K, temperature 0, max_tokens 8192), levels n=1/4/8/16 with 4/8/16/32 questions, 20 s pause between levels. Peak = harness `concurrent_peak_tok_s` (sum of concurrent per-request tok/s); for n=1 it is single-stream tok/s. Runs are short (one to two waves per level), so differences between cases are within noise unless large (the same configuration, case 03 vs 03t, differs by 45 % at n=16 peak, see below).

"!"-collapse probe (cases with MTP on, plus case 02 as control): 16 requests, max_tokens 400, 16 short diverse prompts, enable_thinking alternating true/false, 4 requests with a `get_weather` tool definition, concurrency 4, temperature 1.0, top_p 0.95. A generation counts as collapsed if content+reasoning matches `!{4,}` or any char repeated 10+ times.

Head-log metrics (accept len mean, `cuda graph: False` count) are over all `Decode batch` lines from after the boot through the end of the bench (probe + 4 levels). Case 03t used no probe.

The deploy of case 01 first failed in `Tear down disabled gated SGLang companion instances` (`object of type 'dict' has no attribute 'nodes'`, sglang.yml:53) after head and workers were already applied. A rerun with the extra var `-e '{"_sglang_vision_instance":{"nodes":[1,2,3,4]}}'` passed and removed the vision instance; that extra var was used for every later rollout. Cause not investigated, no code changed.

## Matrix

| # | Delta | Status | n=1 tok/s | n=4 peak | n=8 peak | n=16 peak | GSM8K correct (n=1/4/8/16 levels) | mean accept len | probe collapses |
|---|-------|--------|----------:|---------:|---------:|----------:|-----------------------------------|----------------:|----------------:|
| 01 | profile as is, MTP EAGLE steps 3 / topk 1 / draft 4 | ok | 67.9 | 209.6 | 225.5 | 272.2 | 4/4, 8/8, 15/16, 31/32 | 3.01 | 0/16 |
| 02 | `speculative_enabled: false` (control) | ok | 34.6 | 116.0 | 187.9 | 284.5 | 4/4, 8/8, 15/16, 31/32 | n/a | 0/16 |
| 03 | MTP steps 2 / draft 3 | ok | 61.9 | 177.4 | 287.4 | 370.3 | 4/4, 8/8, 15/16, 31/32 | 2.53 | 0/16 |
| 04 | MTP steps 1 / draft 2 | ok | 57.4 | 163.4 | 268.7 | 364.1 | 4/4, 8/8, 15/16, 31/32 | 1.85 | not run |
| 05 | `speculative_adaptive: true` (3/1/4 start, existing candidate config) | ok | 71.9 | 162.0 | 265.0 | 327.5 | 4/4, 8/8, 15/16, 31/32 | 3.07 | 0/16 |
| 06 | tree drafting: steps 2 / topk 2 / draft 4 | BOOT FAIL | - | - | - | - | - | - | - |
| 03t | case 03 config (shipped), bench at `--temperature 1.0` | ok | 63.5 | 182.0 | 274.6 | 205.1 | 4/4, 8/8, 15/16, 31/32 | 2.34 | not run |

Quality in every ok case: one wrong answer at the n=8 level and one at the n=16 level (the same in all cases, including MTP off and temperature 1.0), `char_runs` empty and `odd_nonascii` empty in all analyze outputs, no failed requests other than those wrong answers. No `cuda graph: False` decode batches in any case. No WARNING/ERROR/Traceback lines in the head logs of cases 01 to 05 (grep over the post-boot log for `WARNING|ERROR|Traceback|illegal|no scaling`). 03t used temperature 1.0 without top_p (`sglang-gsm8k run` has no top_p flag).

Probe finish reasons: 01: 11 stop / 4 tool_calls / 1 length; 02: 9 stop / 4 tool_calls / 3 length; 03: 10 / 4 / 2; 05: 10 / 4 / 2. In every probed case all 4 tool-definition requests returned a parsed `tool_calls` entry.

## Boot findings

- Case 01 boot: weights loaded in 202.6 s (target) + 25.3 s (MTP head), head Ready about 6 min after rollout (`scheduler_e2e` 313.7 s), cuda graph capture target_verify 8.1 s + draft_decode 10.0 s. JuiceFS cache warm. Other cases: `scheduler_e2e` 274 to 332 s.
- p66 applied on every boot: `Patched __init__.py: widen the shipped SM121 QSA contract to TP4 (6 q, 1 kv)` and `Using the Codex/Kimi K3 KDA Qwen3.8 QSA kernel on SM121 (radixark/KDA-1.5#4, ...)`. Also `QSA MTP index sharing enabled` (disabled under adaptive: `index_share_for_mtp_iteration is disabled under adaptive speculative decoding`).
- `max_running_requests=16` in every case. `max_total_num_tokens`: 01 2,853,760; 02 (no MTP) 2,215,232; 03 2,555,968; 04 2,436,480; 05 2,638,976.
- One non-fatal line in case 01 capture: TileLang `Warning: Data race detected: Logits(bx, position) is written by multiple threads` (no impact observed).
- Case 05 (adaptive): boots with `speculative_moe_runner_backend: triton` and `speculative_adaptive: true`; the "Unsupported moe_runner_backend ... Use flashinfer_cutlass instead" assertion (upstream #38795) did NOT occur. #38932 (commit 42875bcd2a) is an ancestor of v0.5.21 (checked by the coordinator), so the profile comment "fix PR #38932 unmerged" is stale. Log: `AdaptiveSpeculativeParams initialized: steps=3, candidate_steps=[1, 2, 3]`, runtime states built for steps 0/1/2 (4 to 5 s each). During probe+bench it logged 14 switches, all between steps 3 and 2 (7x 3->2, 7x 2->3; ema_accept_len 0.74 to 1.70 at switch time). No switch to 0 or 1 appeared in the log.
- Case 06 failure (head `TP0 EP0` and worker-1 `TP1 EP1`, read from Loki): `Scheduler hit an exception: Traceback ... NotImplementedError: Qwen4-Exp QSA MTP currently supports speculative_eagle_topk=1`. Head and workers went to Error and restarted; nothing was served. Tree drafting (topk > 1) is not available for this model on 0.5.21; the case was abandoned and the profile restored from the saved copy.
- Case 02 (MTP off) uses a plain decode graph (`decode=5.07 s`) and no draft pools.

## LiteLLM check (case 01)

From inside the litellm pod via `http://localhost:4000/v1/chat/completions` with the master key: `hermes-default` answered `'\n\nHi'` (finish stop), `vision-default` answered `'\n\nHi'` (finish stop, falls back to the base model). Both OK.

## Observations

- Single stream: MTP on gives 57 to 72 tok/s vs 34.6 tok/s without MTP (about 1.7 to 2.1x), with accept len 3.01 (3/1/4), 2.53 (2/1/3), 1.85 (1/1/2). At n=4 MTP is also ahead (162 to 210 vs 116).
- At n=8 and n=16 the cases are close and the ordering is not stable: MTP off peaks 284.5 at n=16, above case 01 (272.2), while 03 and 04 read 364 to 370. The shipped config re-measured at temperature 1.0 (03t) reads 205.1 at n=16 against 370.3 at temperature 0; accept len fell from 2.53 to 2.34, which does not explain that gap, so run-to-run variance of these short runs is large. n=8/16 differences must not be read as a ranking.
- The 0.5.20 profile-header numbers (51.5 t/s single stream, aggregate 215.4 at 16 streams, 3/1/4) are not directly comparable (different sampling, wave count, aggregate definition); the 0.5.21 n=1 peak of 67.9 (case 01) and 71.9 (case 05) is above it.
- The "!" collapse did not reproduce with MTP on (0/16 in cases 01, 03, 05; control 02 also 0/16). 16 requests of up to 400 tokens per case is informative against the earlier 3/16 to 4/16 rate on the RadixArk checkpoint, not exhaustive.

## Results / decision

- MTP stays on: GSM8K pass counts equal the MTP-off control, no char runs, no collapses, tool calls parse.
- The profile now ships MTP EAGLE steps 2 / topk 1 / draft 3, `speculative_adaptive: false` (user decision, based on cases 01 to 06). Only `speculative_num_steps` (3 to 2) and `speculative_num_draft_tokens` (4 to 3) differ from HEAD.
- #38932 is in v0.5.21, so adaptive is no longer blocked (case 05 booted and ran clean); the "unmerged" comment block in the profile is stale and left for the maintainer to update.
- Topk 2 (tree drafting) is rejected by the model's QSA MTP path (case 06).
- Cluster final state: deployed with the 2/1/3 profile (case 03t), head 2/2 and 3 workers 1/1 Running, `/health` 200 and a chat completion answered through the port-forward (since closed).

## Definitive A/B: fixed 2/1/3 vs adaptive (corrected tiers)

### Method

- A = fixed MTP EAGLE steps 2 / topk 1 / draft 3, `speculative_adaptive: false` (already deployed; server_args verified). B = `speculative_adaptive: true`, start steps 2, draft 3, topk 1, `speculative_adaptive_config` = `"1": [1,2,3]`, `"8": [1,2,3]`, `"32": [1,2]` (only these profile keys changed; the "64" tier was dropped). Verified for B: server_args `speculative_adaptive: True`, config file `/tmp/speculative_adaptive_config.json` with exactly those tiers, `AdaptiveSpeculativeParams initialized: steps=2, candidate_steps=[1, 2, 3]`, runtime states built for steps 1 and 3, no disabled warning. Plain `--tags sglang` deployed without the `_sglang_vision_instance` workaround (teardown fix works).
- Source rule (../sglang adaptive_spec_params.py, working tree is not exactly the v0.5.21 tag): a start value not in a tier's candidate_steps does not raise, it falls back to the middle candidate; `_find_closest_bs` is a floor lookup (largest key <= padded batch size, else the smallest key). With max_running_requests 16: batch sizes 1 to 7 use tier "1", 8 to 16 use tier "8"; tier "32" is unreachable.
- Levels (concurrency, questions) 1/8, 4/32, 8/64, 16/128, the same first-N GSM8K questions for every run, temperature 0, max_tokens 8192, `--minutes 30`, 30 s pause between levels. Order: A rep 1, A rep 2, rollout B, B rep 1, B rep 2, rollout back to A (profile byte-identical to the saved copy), A rep 3 (n=8, n=16 only) as drift control. A first drift-control attempt failed on a stale port-forward (RemoteDisconnected on all requests) and was discarded and rerun; it is not counted.
- Peak = harness `concurrent_peak_tok_s`; aggregate = `aggregate_tok_s`; per-request median = median of completion_tokens/latency from the result JSONL. Accept len = mean over all `Decode batch` lines in the head log between bench start and end (probe excluded); those lines are logged periodically, so counts are small (127 to 287 per rep).

### Results (peak / aggregate tok/s)

| Level | A rep 1 | A rep 2 | A rep 3 (after redeploy) | B rep 1 | B rep 2 |
|-------|---------|---------|--------------------------|---------|---------|
| n=1  | 68.6 / 56.3 | 62.8 / 53.9 | not run | 73.3 / 69.5 | 74.6 / 70.3 |
| n=4  | 178.8 / 114.4 | 152.6 / 79.8 | not run | 205.6 / 181.2 | 212.9 / 187.1 |
| n=8  | 217.3 / 172.7 | 252.2 / 124.5 | 276.9 / 236.5 | 302.1 / 245.2 | 317.2 / 214.1 |
| n=16 | 343.4 / 165.3 | 263.2 / 160.1 | 419.5 / 219.5 | 435.0 / 235.3 | 469.7 / 238.3 |

Per-request median tok/s (A r1 / r2 / r3 ; B r1 / r2):

| Level | A | B |
|-------|---|---|
| n=1  | 61.5 / 49.3 / - | 70.0 / 71.1 |
| n=4  | 34.3 / 34.5 / - | 47.1 / 49.4 |
| n=8  | 21.6 / 20.8 / 30.9 | 33.2 / 33.7 |
| n=16 | 12.3 / 12.0 / 21.7 | 21.5 / 23.1 |

Level wall time (s), A r1 / r2 / r3 ; B r1 / r2: n=1 43.8 / 44.0 / - ; 37.9 / 36.1. n=4 138.7 / 211.8 / - ; 55.5 / 54.4. n=8 124.3 / 204.7 / 83.1 ; 85.7 / 104.1. n=16 423.8 / 471.9 / 272.4 ; 254.4 / 259.9.

Run-to-run spread (max/min of peak): A n=1 1.09x, n=4 1.17x (two reps), n=8 1.27x, n=16 1.59x (three reps); B between its two reps n=1 1.02x, n=4 1.04x, n=8 1.05x, n=16 1.08x.

Drift: A rep 3 (after the redeploy) is faster than A reps 1 and 2 at the same levels (n=8 peak 277 vs 217/252, n=16 peak 420 vs 343/263; n=16 per-request median 21.7 vs 12.3/12.0). Cause not investigated; no other SGLang instance shared the GPUs. There is no A rep 3 for n=1 and n=4, so the A-vs-B comparison at those levels rests on A reps 1/2 only and may be inflated by the same slow state.

### Accept length and adaptive behaviour

| Case | mean accept len | per `#running-req` bucket |
|------|----------------:|---------------------------|
| A rep 1 / 2 / 3 | 2.52 / 2.51 / 2.54 | 2.45 to 2.73 |
| B rep 1 / 2 | 3.10 / 3.08 | 2.85 to 3.28 (n=16 bucket 3.17 / 3.20) |

- `cuda graph: False` decode batches: 0 in every rep. WARNING/ERROR/Traceback/illegal lines in the bench windows: 0.
- B switches in the bench windows: rep 1 15x 3->2 and 16x 2->3; rep 2 17x 3->2 and 17x 2->3. No switch involving step 1 was logged. The log line (`Adaptive spec params updated: steps X -> Y (ema_accept_len=...)`) carries no tier or batch size, so per-tier counts and steps used per batch size are not available. Accept len 3.1 (A: 2.5) at every batch size indicates B ran mostly 3 steps, including at n=8/16 (tier "8"), which the old config could not do.

### Probe and correctness

- "!" probe (16 requests): A 0/16 collapsed (10 stop / 4 tool_calls / 2 length, tool calls parsed 4/4); B 0/16 collapsed (9 stop / 4 tool_calls / 3 length, tool calls parsed 4/4).
- GSM8K exact/n (A r1, r2, r3 ; B r1, r2): n=1 8/8 all; n=4 31/32, 30/32, - ; 31/32, 31/32; n=8 63/64, 62/64, 63/64 ; 63/64, 63/64; n=16 123/128, 123/128, 124/128 ; 125/128, 124/128. No char runs; all failures are wrong answers, no failed requests.

### Decision

- The profile is restored to A (byte-identical to the saved copy) and deployed.
- Against A reps 1/2, B is ahead at every level (peak n=1 +10 to +19 %, n=4 +15 to +40 %, n=8 +20 to +46 %, n=16 +27 to +78 %), but A rep 3 shows that A itself moved by up to 59 % at n=16 within the session. Against A rep 3, B is +9 % / +15 % peak at n=8 (302 / 317 vs 277) and +4 % / +12 % at n=16 (435 / 470 vs 420); aggregate 245 / 214 vs 237 (n=8) and 235 / 238 vs 220 (n=16). These differences are smaller than the measured run-to-run spread of A, so no throughput win of B at n=8 or n=16 is established. At n=1 and n=4 B exceeds the spread between A reps, but without a drift-matched A rep, so not conclusive.
- What is consistent: B accept len 3.1 vs 2.5, B rep-to-rep agreement within 1 to 8 %, no correctness or stability regression. Whether 3 steps beats 2 steps in throughput cannot be separated from drift with this data; an interleaved A/B (alternating reps) is needed.
