# SGLang Test Log: Qwen3.8-27B-NVFP4-ModelOpt, 4 Nodes (K3s cluster), TP=4, v0.5.21-sm121 (speculative sweep: DSpark g7 / DSpark g5 / MTP / none)

> **STATUS: COMPLETE, executed 2026-10-11 (08:01 to 09:47 UTC).** Five rollouts, GSM8K chat sweep n=1..32 each. No boot failures, no restarts, no `cuda graph: False` decode batches, no WARNING/ERROR/Traceback in any head log.

## Environment

| Component            | Value                                                                                          |
|----------------------|------------------------------------------------------------------------------------------------|
| GPU                  | NVIDIA GB10 (SM121), 4 nodes (spark1 head, spark2-4 workers), driver 580.178.04                |
| Image                | `xomoxcc/dgx-spark-sglang:0.5.21-sm121`                                                        |
| Model                | `vroomfondel/Qwen3.8-27B-NVFP4-ModelOpt` (uniform W4A4 NVFP4, FP8 KV)                          |
| Drafts               | `RadixArk/Qwen3.8-27B-DSpark` (block 7); built-in MTP head (driven as EAGLE)                   |
| Parallelism          | TP=4, PP=1, EP=1, `nccl_transport: roce`                                                       |
| Profile args         | `--mem-fraction-static 0.6 --context-length 262144 --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer --fp4-gemm-backend flashinfer_cutlass --max-running-requests 32 --cuda-graph-max-bs-decode 32 --mamba-radix-cache-strategy extra_buffer` (no `--mamba-ssm-dtype`, i.e. default; no `SGLANG_RAGGED_VERIFY_MODE`) |
| Deploy               | `ansible-playbook k8s_dgx.yml --tags sglang -e sglang_model=vroomfondel/Qwen3.8-27B-NVFP4-ModelOpt`, profile edited per case, restored byte-identical afterwards |
| Date                 | 2026-10-11                                                                                     |

Single-node reference: `TESTLOGS/standalone_nn1_tp1/qwen-3.8-27b-nvfp4-modelopt/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-27b-nvfp4-modelopt_1n.md` (same image, TP=1).

## Method

One rollout per case. Bench via `sglang-gsm8k` through `kubectl port-forward svc/sglang` (chat API, GSM8K test split, temperature 0, max_tokens 8192), levels n=1/4/8/16/32 with the first 4/16/32/64/128 questions, 20 s pause between levels. n=1 = single-stream tok/s; n>1 throughput of this sweep is not listed (`concurrent_peak_tok_s` overstates full load), see "Full-load rerun" below. Accept len = median of `accept len` over head `Decode batch` lines inside the bench window (the head logs only a sample of batches, 138 to 253 per case).

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

| #  | n=1 | n=4   | n=8   | n=16  | n=32    | truncated at 8192 (all levels) |
|----|----:|------:|------:|------:|--------:|-------------------------------:|
| 1  | 4/4 | 15/16 | 31/32 | 62/64 | 122/128 | 3 |
| 2  | 4/4 | 15/16 | 31/32 | 62/64 | 122/128 | 3 |
| 3  | 4/4 | 15/16 | 30/32 | 62/64 | 123/128 | 5 |
| 4  | 4/4 | 15/16 | 30/32 | 62/64 | 123/128 | 4 |
| 5  | 4/4 | 15/16 | 30/32 | 62/64 | 123/128 | 1 |

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

n=1 from the sweep above; n>1 = `full_load_tok_s` from the full-load reruns (this log and the 1n log, 2026-10-11). Same-config factor (4 nodes / 1 node):

| Config                       | n=1                  | n=4                    | n=8                    | n=16                   | n=32                   |
|------------------------------|----------------------|------------------------|------------------------|------------------------|------------------------|
| none                         | x2.86 (36.0 / 12.6)  | x2.31 (113.9 / 49.3)   | x2.24 (211.2 / 94.4)   | x2.05 (322.6 / 157.2)  | x1.83 (449.2 / 245.7)  |
| DSpark g7, float32 SSM       | x2.15 (103.8 / 48.3) | x1.66 (213.0 / 128.4)  | x1.38 (284.3 / 205.4)  | x1.19 (312.6 / 261.9)  | x1.66 (464.7 / 279.1) ¹ |
| DSpark g7, bfloat16 SSM      | x2.16 (96.0 / 44.5)  | x1.81 (262.0 / 145.1)  | x1.18 (284.7 / 240.4)  | x0.88 (334.4 / 378.2)  | x1.02 (520.2 / 512.1)  |

¹ Single node float32 g7 runs 15 slots, so its n=32 is the rate at 15 running.

MTP 3/1/4 and DSpark g5 were not rerun on the single node. Four nodes give about x2.2 for single streams; with bfloat16 SSM on both sides one node matches four from n=16 on.

Speculation gain on TP=4 (full load, DSpark g7 over the control): float32 x1.87 n=4, x1.35 n=8, x0.97 n=16, x1.03 n=32; bfloat16 x2.30, x1.35, x1.04, x1.16; n=1 x2.7 (sweep, cases 1 and 5). Single node (bfloat16 preset over control): x2.94, x2.55, x2.41, x2.08. The gain shrinks fast with batch size on TP=4: the target verify is cheap and per-step overhead (draft + inter-node sync) dominates, and the control scales almost linearly with n.

Ranking on TP=4 at full load (float32 SSM, one run each, about 20 % run-to-run spread): n=4 g5 244 > g7 213 > MTP 192 > control 114; n=8 MTP 299, g5 292, g7 284 tied, control 211; n=16 MTP 331, control 323, g7 313 tied, g5 285; n=32 g7 465 and control 449 tied, g5 426, MTP 373. At n=1 DSpark g7 (93-104) > g5 (93) > MTP (70) > control (36). MTP is below the control at n=32 (x0.83).

Drift control (case 5 vs 1): n=1 -10 %; the 4 questions include the first request's warm-up, read single-digit percent as noise.

## Decision

- Keep DSpark gamma 7 (RadixArk draft), now with bfloat16 SSM (profile). It leads at n=1, and with bfloat16 also at n=4 (262) and n=32 (520); at n=8/16 all speculative configs and the control are within the spread. Healthy accept len (4.3 median, 5.2 at high concurrency), no graph fallbacks, no warnings beyond the KV-scale line.
- MTP (EAGLE 3/1/4) is a valid fallback but below the control at n=32; gamma 5 shows no consistent advantage over g7 (ahead at n=4, behind at n=16/32).
- At n>=16 speculation on TP=4 gains at most x1.16 at full load, and a single TP=1 node with bfloat16 SSM reaches 378 tok/s at n=16 and 512 at n=32 (4 nodes: 334 and 520) while leaving three Sparks free. If the workload is mostly concurrent, a TP=1 instance (profile header suggests `sglang_instances`) is the better use of hardware; for single-stream latency TP=4 + DSpark g7 is x2.2 faster.
- FP8 KV scales not applied on 0.5.21 (cluster and single node): still open for the p51 / PR #31220 path; unaffected quality at this sample size.
- Header update proposal (not applied): item C resolved with the numbers above; item D re-measured on TP=4 (x2.7 at n=1; full load x1.0 at n=16, x1.0-1.2 at n=32).

## Full-load rerun with the streaming harness (2026-10-11)

All n>1 levels re-measured with `sglang-gsm8k run` (streams by default) and `analyze` -> `full_load_tok_s` (token rate while >= 90 % of the client slots are busy). The old `concurrent_peak_tok_s` was 1.09-1.56x full_load in these runs, so no n>1 peak values are listed in this log. Profile edits deployed one config after another, cluster idle-checked (3 min without POSTs/running requests) before each deploy, profile restored afterwards. Levels n4/n8/n16/n32 with 16/32/64/128 questions, 20 s pause, max-tokens 8192, temperature 0; n1 not rerun (exact at n=1). `max_running_requests` 32 in every case, 0 decode batches with `cuda graph: False`. Head log POSTs = 240 of 240 expected, except DSpark g7 float32 (241, one foreign request).

full_load tok/s:

| config                              | bench (UTC) | n4    | n8    | n16   | n32   | engine n32 | accept len | queue max |
|-------------------------------------|-------------|-------|-------|-------|-------|-----------:|-----------:|----------:|
| DSpark g7, bfloat16 SSM (= profile) | 14:47-14:56 | 262.0 | 284.7 | 334.4 | 520.2 | 454        | 4.38       | 3         |
| DSpark g7, float32 SSM              | 16:57-17:06 | 213.0 | 284.3 | 312.6 | 464.7 | 411        | 4.27       | 2         |
| DSpark g5 (draft 6), float32 SSM    | 17:14-17:24 | 244.0 | 292.2 | 285.0 | 425.6 | 392        | 3.86       | 3         |
| MTP EAGLE 3/1/4, float32 SSM        | 17:32-17:41 | 191.7 | 298.6 | 330.7 | 373.4 | 342        | 3.05       | 2         |
| none, float32 SSM                   | 17:49-18:03 | 113.9 | 211.2 | 322.6 | 449.2 | 438        | -          | 1         |

Best 10 s window / engine max, tok/s. Best 10 s window = highest 10 s streamed token rate of the level (character-weighted timeline; the token-exact timeline via `continuous_usage_stats` came after these runs). Engine max = highest head-log `gen throughput` in the level's time window, logged decode batches in parentheses.

| config                              | n4             | n8              | n16             | n32             |
|-------------------------------------|----------------|-----------------|-----------------|-----------------|
| DSpark g7, bfloat16 SSM (= profile) | 287 / 330 (20) | 308 / 310 (36)  | 380 / 353 (39)  | 618 / 542 (54)  |
| DSpark g7, float32 SSM              | 254 / 269 (17) | 321 / 323 (11)  | 355 / 333 (57)  | 513 / 490 (55)  |
| DSpark g5 (draft 6), float32 SSM    | 261 / 269 (20) | 334 / 328 (37)  | 328 / 322 (54)  | 474 / 462 (59)  |
| MTP EAGLE 3/1/4, float32 SSM        | 211 / 226 (21) | 331 / 350 (22)  | 376 / 385 (40)  | 456 / 474 (76)  |
| none, float32 SSM                   | 130 / 128 (68) | 224 / 233 (215) | 358 / 377 (162) | 500 / 518 (232) |

- GSM8K (n4/n8/n16/n32): 15/16 and 31/32 everywhere (none: +1 truncation at n8); n16 61-63/64; n32 122/128 everywhere with 3-4 truncations at 8192 tokens.
- Engine n32 = head-log `gen throughput` mean at >= 29 running; full_load is 2-10 % above it for none and MTP, 9-18 % for DSpark at n8-n32 (DSpark g7 bf16 n4: 9 % below).
- Speculation pays at low concurrency (DSpark g7 bf16 vs none x2.3 at n4) and fades at n32 (520 vs 449, +16 %; MTP 3/1/4 falls below none there). bfloat16 vs float32 at g7: +23 % n4, 0 % n8, +7 % n16, +12 % n32.
- Comparison with the single node (from n16 on one node with bfloat16 SSM matches four): `SINGLE_CLUSTER_COMPARE.md`.
