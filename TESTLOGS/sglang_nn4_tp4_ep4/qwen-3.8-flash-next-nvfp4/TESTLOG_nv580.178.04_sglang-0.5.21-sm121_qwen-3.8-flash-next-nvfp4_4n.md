# SGLang Test Log: Qwen3.8-Flash-Next-NVFP4 (NVIDIA export), 4 Nodes, TP=4 EP=4, v0.5.21-sm121, MTP sweep

## Environment

| Component | Value                                                                                                              |
|-----------|--------------------------------------------------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121), 128 GB unified per node, driver 580.178.04 (head node `nvidia-smi`), kernel 7.0.0-1019-nvidia |
| Nodes     | 4x DGX Spark: node1 (head/rank0), node2, node3, node4 (1 GB10 each)                                                |
| Image     | `xomoxcc/dgx-spark-sglang:0.5.21-sm121` (pinned in profile)                                                        |
| Model     | `nvidia/Qwen3.8-Flash-Next-NVFP4` (quant `modelopt_mixed`, MIXED_PRECISION; FP8 in-checkpoint MTP head)            |
| Transport | RoCE via SR-IOV VF                                                                                                 |
| Parallel  | tp=4, pp=1, ep=4; `moe_runner_backend: flashinfer_cutlass`, MTP draft MoE `triton`                                 |
| Date      | 2026-10-10                                                                                                         |

Profile: `roles/k8s_dgx/model_profiles/nvidia-qwen3.8-flash-next-nvfp4.yml`. Predecessor measurement (0.5.20-sm121, 2026-09-23, MTP 3/1/4): see the table in the profile header.
Cluster context: the Nemotron-3-Super default and the Nano-Omni vision instance were replaced by this model (`enable_vision_model: false`, `_sglang_default_instance: {}`); LiteLLM redeployed with `litellm_vision_alias_fallback_to_base: true`. No other instance shared the GPUs during the runs.

## Method

One Ansible rollout per case (`ansible-playbook k8s_dgx.yml --tags sglang`; case 01 with `--tags sglang,litellm`), profile restored from a saved copy before each edit. Bench via `sglang-gsm8k run` (chat API, GSM8K, temperature 0, max_tokens 8192), levels n=1/4/8/16 with 4/8/16/32 questions, 20 s pause between levels. n=1 tok/s = single-stream rate. n>1 throughput of this sweep is not listed (`concurrent_peak_tok_s` overstates full load); full-load numbers are in "Engine-log rerun" below.

"!"-collapse probe (cases with MTP on, plus case 02 as control): 16 requests, max_tokens 400, 16 short diverse prompts, enable_thinking alternating true/false, 4 requests with a `get_weather` tool definition, concurrency 4, temperature 1.0, top_p 0.95. A generation counts as collapsed if content+reasoning matches `!{4,}` or any char repeated 10+ times.

Head-log metrics (accept len mean, `cuda graph: False` count) are over all `Decode batch` lines from after the boot through the end of the bench (probe + 4 levels). Case 03t used no probe.

The deploy of case 01 first failed in `Tear down disabled gated SGLang companion instances` (`object of type 'dict' has no attribute 'nodes'`, sglang.yml:53) after head and workers were already applied. A rerun with the extra var `-e '{"_sglang_vision_instance":{"nodes":[1,2,3,4]}}'` passed and removed the vision instance; that extra var was used for every later rollout. Cause not investigated, no code changed.

## Matrix

| #   | Delta                                                                 | Status    | n=1 tok/s | GSM8K correct (n=1/4/8/16 levels) | mean accept len | probe collapses |
|-----|-----------------------------------------------------------------------|-----------|----------:|-----------------------------------|----------------:|----------------:|
| 01  | profile as is, MTP EAGLE steps 3 / topk 1 / draft 4                   | ok        |      67.9 | 4/4, 8/8, 15/16, 31/32            |            3.01 |            0/16 |
| 02  | `speculative_enabled: false` (control)                                | ok        |      34.6 | 4/4, 8/8, 15/16, 31/32            |             n/a |            0/16 |
| 03  | MTP steps 2 / draft 3                                                 | ok        |      61.9 | 4/4, 8/8, 15/16, 31/32            |            2.53 |            0/16 |
| 04  | MTP steps 1 / draft 2                                                 | ok        |      57.4 | 4/4, 8/8, 15/16, 31/32            |            1.85 |         not run |
| 05  | `speculative_adaptive: true` (3/1/4 start, existing candidate config) | ok        |      71.9 | 4/4, 8/8, 15/16, 31/32            |            3.07 |            0/16 |
| 06  | tree drafting: steps 2 / topk 2 / draft 4                             | BOOT FAIL |         - | -                                 |               - |               - |
| 03t | case 03 config (shipped), bench at `--temperature 1.0`                | ok        |      63.5 | 4/4, 8/8, 15/16, 31/32            |            2.34 |         not run |

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

- Single stream: MTP on gives 57 to 72 tok/s vs 34.6 tok/s without MTP (about 1.7 to 2.1x), with accept len 3.01 (3/1/4), 2.53 (2/1/3), 1.85 (1/1/2). In the engine-log rerun below (engine max, float32 SSM, 16 slots) MTP leads by x1.6-1.7 at n=4, x1.5 at n=8, x1.1-1.2 at n=16 and x1.05-1.1 at n=32.
- The temperature-1.0 rerun of the shipped config (03t) lowered accept len from 2.53 to 2.34.
- The 0.5.20 profile-header numbers (51.5 t/s single stream, aggregate 215.4 at 16 streams, 3/1/4) are not directly comparable (different sampling, wave count, aggregate definition); the 0.5.21 n=1 rate of 67.9 (case 01) and 71.9 (case 05) is above it.
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
- n=1 = single-stream tok/s; aggregate = `aggregate_tok_s` (total tokens / wall time, diluted by the longest request); per-request median = median of completion_tokens/latency from the result JSONL. Accept len = mean over all `Decode batch` lines in the head log between bench start and end (probe excluded); those lines are logged periodically, so counts are small (127 to 287 per rep).

### Results (n=1 tok/s; aggregate tok/s)

| Level             | A rep 1 | A rep 2 | A rep 3 (after redeploy) | B rep 1 | B rep 2 |
|-------------------|---------|---------|--------------------------|---------|---------|
| n=1 single stream | 68.6    | 62.8    | not run                  | 73.3    | 74.6    |
| n=1 aggregate     | 56.3    | 53.9    | not run                  | 69.5    | 70.3    |
| n=4 aggregate     | 114.4   | 79.8    | not run                  | 181.2   | 187.1   |
| n=8 aggregate     | 172.7   | 124.5   | 236.5                    | 245.2   | 214.1   |
| n=16 aggregate    | 165.3   | 160.1   | 219.5                    | 235.3   | 238.3   |

Per-request median tok/s (A r1 / r2 / r3 ; B r1 / r2):

| Level | A                  | B           |
|-------|--------------------|-------------|
| n=1   | 61.5 / 49.3 / -    | 70.0 / 71.1 |
| n=4   | 34.3 / 34.5 / -    | 47.1 / 49.4 |
| n=8   | 21.6 / 20.8 / 30.9 | 33.2 / 33.7 |
| n=16  | 12.3 / 12.0 / 21.7 | 21.5 / 23.1 |

Level wall time (s), A r1 / r2 / r3 ; B r1 / r2: n=1 43.8 / 44.0 / - ; 37.9 / 36.1. n=4 138.7 / 211.8 / - ; 55.5 / 54.4. n=8 124.3 / 204.7 / 83.1 ; 85.7 / 104.1. n=16 423.8 / 471.9 / 272.4 ; 254.4 / 259.9.

Run-to-run spread (max/min of per-request median): A n=1 1.25x, n=4 1.01x (two reps), n=8 1.49x, n=16 1.81x (three reps); B between its two reps n=1 1.02x, n=4 1.05x, n=8 1.02x, n=16 1.07x.

Drift: A rep 3 (after the redeploy) is faster than A reps 1 and 2 at the same levels (n=8 per-request median 30.9 vs 21.6/20.8, n=16 21.7 vs 12.3/12.0). Cause not investigated; no other SGLang instance shared the GPUs. There is no A rep 3 for n=1 and n=4, so the A-vs-B comparison at those levels rests on A reps 1/2 only and may be inflated by the same slow state.

### Accept length and adaptive behaviour

| Case            |    mean accept len | per `#running-req` bucket              |
|-----------------|-------------------:|----------------------------------------|
| A rep 1 / 2 / 3 | 2.52 / 2.51 / 2.54 | 2.45 to 2.73                           |
| B rep 1 / 2     |        3.10 / 3.08 | 2.85 to 3.28 (n=16 bucket 3.17 / 3.20) |

- `cuda graph: False` decode batches: 0 in every rep. WARNING/ERROR/Traceback/illegal lines in the bench windows: 0.
- B switches in the bench windows: rep 1 15x 3->2 and 16x 2->3; rep 2 17x 3->2 and 17x 2->3. No switch involving step 1 was logged. The log line (`Adaptive spec params updated: steps X -> Y (ema_accept_len=...)`) carries no tier or batch size, so per-tier counts and steps used per batch size are not available. Accept len 3.1 (A: 2.5) at every batch size indicates B ran mostly 3 steps, including at n=8/16 (tier "8"), which the old config could not do.

### Probe and correctness

- "!" probe (16 requests): A 0/16 collapsed (10 stop / 4 tool_calls / 2 length, tool calls parsed 4/4); B 0/16 collapsed (9 stop / 4 tool_calls / 3 length, tool calls parsed 4/4).
- GSM8K exact/n (A r1, r2, r3 ; B r1, r2): n=1 8/8 all; n=4 31/32, 30/32, - ; 31/32, 31/32; n=8 63/64, 62/64, 63/64 ; 63/64, 63/64; n=16 123/128, 123/128, 124/128 ; 125/128, 124/128. No char runs; all failures are wrong answers, no failed requests.

### Decision

- The profile is restored to A (byte-identical to the saved copy) and deployed.
- Against A reps 1/2, B is ahead at every level (per-request median n=1 +14 to +44 %, n=4 +37 to +44 %, n=8 +54 to +62 %, n=16 +75 to +93 %), but A rep 3 shows that A itself moved by up to 81 % at n=16 within the session. Against A rep 3, B is +7 % / +9 % per-request median at n=8 (33.2 / 33.7 vs 30.9) and -1 % / +6 % at n=16 (21.5 / 23.1 vs 21.7); aggregate 245 / 214 vs 237 (n=8) and 235 / 238 vs 220 (n=16). These differences are smaller than the measured run-to-run spread of A, so no throughput win of B at n=8 or n=16 is established. The engine-log rerun (below, one run each, back to back) has A and B within 8 % at every level (engine max n=4 202 vs 215, n=8 306 vs 302, n=16 400 vs 430, n=32 372 vs 396).
- What is consistent: B accept len 3.1 vs 2.5, B rep-to-rep agreement within 2 to 7 % (per-request median), no correctness or stability regression. Whether 3 steps beats 2 steps in throughput cannot be separated from drift with this data; an interleaved A/B (alternating reps) is needed.

## Engine-log rerun (2026-10-12)

All four configs re-run on 2026-10-12 after a power cycle of all Sparks, one rollout per config from a scratch copy of the repo with only the config's profile keys changed, cluster idle-checked (3 min without POSTs/running requests) before each deploy, default profile redeployed afterwards. Levels n4/n8/n16/n32 with 32/64/128/256 questions, 20 s pause, max-tokens 8192, temperature 0, harness `sglang-gsm8k` (token-exact streaming via `continuous_usage_stats`). Throughput comes from the head log (`Decode batch` lines, log interval 40 decode steps): each cell is the highest `gen throughput` (tok/s) in the level's time window. SGLang sums every generated token between two log lines and divides by the elapsed wall time, so each line is a measured rate over a few seconds. These numbers replace the client-side figures of 2026-10-11.

engine max tok/s:

| config                                             | slots | bench (UTC) | n4  | n8  | n16 | n32 | accept len |
|----------------------------------------------------|------:|-------------|-----|-----|-----|-----|-----------:|
| adaptive, bfloat16 SSM, max_running 32 (= profile) |    32 | 09:50-10:06 | 231 | 331 | 476 | 646 |       2.88 |
| adaptive, float32 SSM, max_running 16 (B)          |    16 | 10:53-11:12 | 215 | 302 | 430 | 396 |       2.97 |
| fixed 2/1/3, float32 SSM, max_running 16 (A)       |    16 | 11:24-11:44 | 202 | 306 | 400 | 372 |       2.52 |
| no MTP, float32 SSM, max_running 16                |    16 | 11:54-12:18 | 124 | 206 | 351 | 354 |          - |

- GSM8K (n4/n8/n16/n32): 31/32 everywhere; n8 62-63/64; n16 124-126/128; n32 250-253/256; at most 4 truncations at 8192 tokens per level.
- Head log POSTs = 480 of 480 expected, except the profile run (481, one foreign request); 0 decode batches with `cuda graph: False`.
- With 16 slots n32 is no faster than n16 (queue). The profile (bfloat16, 32 slots) leads from n16 on: n16 476 vs 351-430, n32 646 vs 354-396.
- Day-to-day: engine max at n32 vs 2026-10-11 (before a power cycle of all Sparks): profile 646 vs 435, adaptive float32 396 vs 416, fixed 2/1/3 372 vs 435, no MTP 354 vs 329. Cause not investigated.
- Comparison with the single node and the other models: `SINGLE_CLUSTER_COMPARE.md`.
