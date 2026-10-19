# SGLang Test Log: Qwen3.8-27B-NVFP4-ModelOpt, 4 Nodes (K3s cluster), TP=4, v0.5.21-sm121 (speculative sweep: DSpark g7 / DSpark g5 / MTP / none)

> **STATUS: COMPLETE, executed 2026-10-11 (08:01 to 09:47 UTC).** Five rollouts, GSM8K chat sweep n=1..32 each. No boot failures, no restarts, no `cuda graph: False` decode batches, no WARNING/ERROR/Traceback in any head log.

## Environment

| Component    | Value                                                                                                                                                                                                                                                                                                                       |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GPU          | NVIDIA GB10 (SM121), 4 nodes (spark1 head, spark2-4 workers), driver 580.178.04                                                                                                                                                                                                                                             |
| Image        | `xomoxcc/dgx-spark-sglang:0.5.21-sm121`                                                                                                                                                                                                                                                                                     |
| Model        | `vroomfondel/Qwen3.8-27B-NVFP4-ModelOpt` (uniform W4A4 NVFP4, FP8 KV)                                                                                                                                                                                                                                                       |
| Drafts       | `RadixArk/Qwen3.8-27B-DSpark` (block 7); built-in MTP head (driven as EAGLE)                                                                                                                                                                                                                                                |
| Parallelism  | TP=4, PP=1, EP=1, `nccl_transport: roce`                                                                                                                                                                                                                                                                                    |
| Profile args | `--mem-fraction-static 0.6 --context-length 262144 --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer --fp4-gemm-backend flashinfer_cutlass --max-running-requests 32 --cuda-graph-max-bs-decode 32 --mamba-radix-cache-strategy extra_buffer` (no `--mamba-ssm-dtype`, i.e. default; no `SGLANG_RAGGED_VERIFY_MODE`) |
| Deploy       | `ansible-playbook k8s_dgx.yml --tags sglang -e sglang_model=vroomfondel/Qwen3.8-27B-NVFP4-ModelOpt`, profile edited per case, restored byte-identical afterwards                                                                                                                                                            |
| Date         | 2026-10-11                                                                                                                                                                                                                                                                                                                  |

Single-node reference: `TESTLOGS/standalone_nn1_tp1/qwen-3.8-27b-nvfp4-modelopt/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-27b-nvfp4-modelopt_1n.md` (same image, TP=1).

## Method

One rollout per case. Bench via `sglang-gsm8k` through `kubectl port-forward svc/sglang` (chat API, GSM8K test split, temperature 0, max_tokens 8192), levels n=1/4/8/16/32 with the first 4/16/32/64/128 questions, 20 s pause between levels. n=1 = single-stream tok/s; n>1 throughput of this sweep is not listed (`concurrent_peak_tok_s` overstates full load), see "Engine-log rerun" below. Accept len = median of `accept len` over head `Decode batch` lines inside the bench window (the head logs only a sample of batches, 138 to 253 per case).

Differences to the single-node method (keep in mind when comparing): question counts at n=4/8/16 are 16/32/64 here vs 8/16/32 there; TP=4 vs TP=1; `mem_fraction_static` 0.6 vs 0.80; `max_running_requests` 32 vs 48; mamba SSM dtype default here vs float32 there; this role does not set `SGLANG_RAGGED_VERIFY_MODE=static` or `--speculative-draft-attention-backend`.

## Matrix

| #  | Spec config                         | n=1 tok/s | accept len median (ceiling) | max_running_requests | max_mamba_cache_size |
|----|-------------------------------------|----------:|----------------------------:|---------------------:|---------------------:|
| 1  | DSpark g7 (draft 8, = profile)      | 103.8     | 4.25 (8)                    | 32                   | 280                  |
| 2  | DSpark g5 (block 5, draft 6)        | 93.3      | 3.92 (6)                    | 32                   | 328                  |
| 3  | MTP (EAGLE) 3 / 1 / 4               | 69.7      | 3.02 (4)                    | 32                   | 405                  |
| 4  | none (control)                      | 36.0      | -                           | 32                   | 754                  |
| 5  | DSpark g7 repeat (drift control)    | 93.2      | 4.30 (8)                    | 32                   | 276                  |

GSM8K correct (n=1 / 4 / 8 / 16 / 32 questions 4/16/32/64/128):

| #   | n=1 |   n=4 |   n=8 |  n=16 |    n=32 | truncated at 8192 (all levels) |
|-----|----:|------:|------:|------:|--------:|-------------------------------:|
| 1   | 4/4 | 15/16 | 31/32 | 62/64 | 122/128 |                              3 |
| 2   | 4/4 | 15/16 | 31/32 | 62/64 | 122/128 |                              3 |
| 3   | 4/4 | 15/16 | 30/32 | 62/64 | 123/128 |                              5 |
| 4   | 4/4 | 15/16 | 30/32 | 62/64 | 123/128 |                              4 |
| 5   | 4/4 | 15/16 | 30/32 | 62/64 | 123/128 |                              1 |

Every run is "n=N ok" (no request failures, no empty responses, no char runs, no odd non-ASCII). Accuracy is the same within 1 question across cases, including the control; the few misses are truncations (runaway reasoning at the 8192 cap) plus a couple of wrong answers present without speculation as well.

Median accept len by `#running-req` (case 1): 1: 4.05, 4: 4.50, 8: 4.80, 16: 5.32, 29-30: 5.2 to 5.5. Case 2: 1: 3.95, 8: 4.35, 16: 4.51, 32: 4.43. Case 3: 1: 2.95, 8: 3.33, 16: 3.33, 32: 3.30. Case 5: 1: 4.22, 8: 4.34, 16: 5.20, 32: 5.21. Accept len rises with batch size and never sits at the ceiling.

## Boot findings

- All cases: head Ready 1 to 3 min after the playbook returned except case 1, which also paid the model download plus JuiceFS warmup (about 17 min cold: 08:01 start, jfs-warmup 14 of 19 GiB at 12 min). Server launch to `max_total_num_tokens` line: 1:30 to 2:34 (case 1 2:34 incl. first-time kernel/JIT work, 1:10 to 1:32 in cases 2 to 5). Head restart count 0 in every case.
- `max_total_num_tokens`: 3.62M (1), 3.63M (2), 4.47M (3), 3.95M (4), 3.56M (5).
- `max_running_requests` = 32 in every case (the configured cap). There was NO "capped ... by the mamba state cache" line in any case; the mamba pool (276 to 754 slots) is not the limit at TP=4 with 0.6 mem fraction, so the single-node cap of 15 at gamma 7 does not occur here and the n=16 and n=32 levels ran without queueing.
- Zero `cuda graph: False` decode batches in all benchmark windows. Zero WARNING/ERROR/Traceback lines.
- **FP8 KV scale warning present:** `Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0.` appears twice with a draft (target + draft: cases 1, 2, 3, 5) and once without (case 4). Same as single node, so SM121 review item 7 is still open on the cluster: the baked KV scales are not picked up on 0.5.21. GSM8K is unaffected.
- Speculative flags were verified in the rendered launch command per case (`--speculative-algo DSPARK ... --speculative-dspark-block-size 5 --speculative-num-draft-tokens 6` for case 2, `--speculative-algo EAGLE --speculative-draft-model-path vroomfondel/Qwen3.8-27B-NVFP4-ModelOpt --speculative-num-steps 3 --speculative-num-draft-tokens 4` for case 3, no `--speculative-*` for case 4).
- Open item C (role deploy hop at TP=4) is validated: the profile boots and serves as shipped.

## Comparison vs single node (TP=1, 1n log)

n=1 from the sweep above; n>1 = engine max tok/s (highest head-log `gen throughput` per level) from the engine-log reruns of 2026-10-12 (this log and the 1n log). Same-config factor (4 nodes / 1 node):

| Config                  | n=1                  | n=4               | n=8               | n=16              | n=32                |
|-------------------------|----------------------|-------------------|-------------------|-------------------|---------------------|
| none                    | x2.86 (36.0 / 12.6)  | x2.58 (129 / 50)  | x2.53 (235 / 93)  | x2.32 (381 / 164) | x2.18 (559 / 257)   |
| DSpark g7, float32 SSM  | x2.15 (103.8 / 48.3) | x1.72 (263 / 153) | x1.27 (309 / 244) | x1.05 (346 / 329) | x1.57 (504 / 322) ¹ |
| DSpark g7, bfloat16 SSM | x2.16 (96.0 / 44.5)  | x1.55 (292 / 188) | x1.00 (304 / 305) | x0.78 (330 / 424) | x0.93 (517 / 553)   |

¹ Single node float32 g7 runs 15 slots, so its n=32 is the rate at 15 running.

MTP 3/1/4 and DSpark g5 were not rerun on the single node. Four nodes give about x2.2 for single streams; with bfloat16 SSM on both sides one node matches four at n=8 and is faster from n=16 on.

Speculation gain on TP=4 (engine max, DSpark g7 over the control): float32 x2.04 n=4, x1.31 n=8, x0.91 n=16, x0.90 n=32; bfloat16 x2.26, x1.29, x0.87, x0.92; n=1 x2.7 (sweep, cases 1 and 5). Single node (bfloat16 preset over control): x3.8, x3.3, x2.6, x2.2. On TP=4 speculation pays only up to n=8; from n=16 on the control is faster: the target verify is cheap and per-step overhead (draft + inter-node sync) dominates, and the control scales almost linearly with n.

Ranking on TP=4 (engine max, 2026-10-12, one run each): n=4 g7 bf16 292, g5 270, g7 float32 263, MTP 217, control 129; n=8 MTP 344, g5 331, g7 304-309, control 235; n=16 control 381, MTP 368, g7 330-346, g5 323; n=32 control 559, g7 504-517, g5 437, MTP 399. At n=1 DSpark g7 (93-104) > g5 (93) > MTP (70) > control (36).

Drift control (case 5 vs 1): n=1 -10 %; the 4 questions include the first request's warm-up, read single-digit percent as noise. Day-to-day: engine max at n=32 vs 2026-10-11 (before a power cycle of all Sparks): g7 bf16 517 vs 542, g7 float32 504 vs 490, g5 437 vs 462, MTP 399 vs 474, control 559 vs 518. Cause of the MTP drop not investigated.

## Decision

- DSpark gamma 7 (RadixArk draft, bfloat16 SSM, profile) stays the pick for latency: x2.7 over the control at n=1 and x2.0-2.3 at n=4 (engine max). From n=16 on the control (no speculation) is faster on TP=4 (n=32 559 vs 504-517). No profile change made; healthy accept len (4.4 mean), no graph fallbacks, no warnings beyond the KV-scale line.
- MTP (EAGLE 3/1/4) is the slowest speculative config at n=32 (399); gamma 5 shows no consistent advantage over g7.
- A single TP=1 node with bfloat16 SSM reaches the cluster's g7 throughput at n=8 and exceeds it from n=16 on (n=16 424 vs 330, n=32 553 vs 517) while leaving three Sparks free. If the workload is mostly concurrent, a TP=1 instance (profile header suggests `sglang_instances`) is the better use of hardware; for single-stream latency TP=4 + DSpark g7 is x2.2 faster.
- FP8 KV scales not applied on 0.5.21 (cluster and single node): still open for the p51 / PR #31220 path; unaffected quality at this sample size.
- Header update proposal (not applied): item C resolved with the numbers above; item D re-measured on TP=4 (x2.7 at n=1; engine max x0.9 at n=16 and n=32).

## Engine-log rerun (2026-10-12)

All five configs re-run on 2026-10-12 after a power cycle of all Sparks, one rollout per config from a scratch copy of the repo with only the config's profile keys changed, cluster idle-checked (3 min without POSTs/running requests) before each deploy, default profile redeployed afterwards. Levels n4/n8/n16/n32 with 16/32/64/128 questions, 20 s pause, max-tokens 8192, temperature 0, harness `sglang-gsm8k` (token-exact streaming via `continuous_usage_stats`). Throughput comes from the head log (`Decode batch` lines, log interval 40 decode steps): each cell is the highest `gen throughput` (tok/s) in the level's time window. SGLang sums every generated token between two log lines and divides by the elapsed wall time, so each line is a measured rate over a few seconds. These numbers replace the client-side figures of 2026-10-11.

engine max tok/s:

| config                              | slots | bench (UTC) | n4  | n8  | n16 | n32 | accept len |
|-------------------------------------|------:|-------------|-----|-----|-----|-----|-----------:|
| DSpark g7, bfloat16 SSM (= profile) |    32 | 10:13-10:25 | 292 | 304 | 330 | 517 |       4.37 |
| DSpark g7, float32 SSM              |    32 | 12:25-12:36 | 263 | 309 | 346 | 504 |       4.33 |
| DSpark g5 (draft 6), float32 SSM    |    32 | 12:43-12:53 | 270 | 331 | 323 | 437 |       3.86 |
| MTP EAGLE 3/1/4, float32 SSM        |    32 | 13:00-13:12 | 217 | 344 | 368 | 399 |       3.05 |
| none, float32 SSM                   |    32 | 13:19-13:35 | 129 | 235 | 381 | 559 |          - |

- GSM8K (n4/n8/n16/n32): 15/16 everywhere; n8 31-32/32; n16 62/64; n32 122-125/128 with 2-5 truncations at 8192 tokens.
- Head log POSTs = 240 of 240 expected, except g7 bf16 (241, one foreign request); 0 decode batches with `cuda graph: False`. `max_running_requests` 32 in every case.
- Comparison with the single node: section above and `SINGLE_CLUSTER_COMPARE.md`.
