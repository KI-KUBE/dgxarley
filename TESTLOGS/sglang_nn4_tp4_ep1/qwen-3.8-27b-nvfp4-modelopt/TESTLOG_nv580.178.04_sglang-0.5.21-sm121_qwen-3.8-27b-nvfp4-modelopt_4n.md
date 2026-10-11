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

One rollout per case. Bench via `sglang-gsm8k` through `kubectl port-forward svc/sglang` (chat API, GSM8K test split, temperature 0, max_tokens 8192), levels n=1/4/8/16/32 with the first 4/16/32/64/128 questions, 20 s pause between levels. Peak = harness `concurrent_peak_tok_s` (sum of per-request tok/s). Accept len = median of `accept len` over head `Decode batch` lines inside the bench window (the head logs only a sample of batches, 138 to 253 per case).

Differences to the single-node method (keep in mind when comparing): question counts at n=4/8/16 are 16/32/64 here vs 8/16/32 there; TP=4 vs TP=1; `mem_fraction_static` 0.6 vs 0.80; `max_running_requests` 32 vs 48; mamba SSM dtype default here vs float32 there; this role does not set `SGLANG_RAGGED_VERIFY_MODE=static` or `--speculative-draft-attention-backend`.

## Matrix

| #  | Spec config                         | n=1 | n=4   | n=8   | n=16  | n=32  | accept len median (ceiling) | max_running_requests | max_mamba_cache_size |
|----|-------------------------------------|----:|------:|------:|------:|------:|----------------------------:|---------------------:|---------------------:|
| 1  | DSpark g7 (draft 8, = profile)      | 103.8 | 293.3 | 355.5 | 428.7 | 587.6 | 4.25 (8)                    | 32                   | 280                  |
| 2  | DSpark g5 (block 5, draft 6)        | 93.3  | 279.7 | 345.1 | 394.6 | 555.4 | 3.92 (6)                    | 32                   | 328                  |
| 3  | MTP (EAGLE) 3 / 1 / 4               | 69.7  | 214.5 | 358.5 | 389.8 | 479.6 | 3.02 (4)                    | 32                   | 405                  |
| 4  | none (control)                      | 36.0  | 126.8 | 230.3 | 360.8 | 552.5 | -                           | 32                   | 754                  |
| 5  | DSpark g7 repeat (drift control)    | 93.2  | 294.7 | 341.3 | 423.5 | 563.0 | 4.30 (8)                    | 32                   | 276                  |

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

Same-config factor (4 node peak / 1 node peak):

| Config                | n=1  | n=4  | n=8  | n=16 |
|-----------------------|-----:|-----:|-----:|-----:|
| none                  | x2.86 (36.0 / 12.6) | x2.57 (126.8 / 49.4) | x2.44 (230.3 / 94.2) | x2.17 (360.8 / 166.3) |
| MTP 3/1/4             | x2.80 (69.7 / 24.9) | x2.28 (214.5 / 94.1) | x2.25 (358.5 / 159.6) | x1.53 (389.8 / 255.2) |
| DSpark g5             | x2.26 (93.3 / 41.3) | x2.21 (279.7 / 126.4) | x1.65 (345.1 / 209.0) | x1.23 (394.6 / 320.6) |
| DSpark g7 (case 1)    | x2.15 (103.8 / 48.3) | x2.11 (293.3 / 138.8) | x1.53 (355.5 / 231.7) | x1.27 (428.7 / 336.9) |

Best config per level, 4 node vs best 1 node: n=1 DSpark g7 103.8 vs 48.3 = x2.15; n=4 DSpark g7 293.3 vs 138.8 = x2.11; n=8 MTP 358.5 (DSpark g7 355.5, within noise) vs 231.7 = x1.55; n=16 DSpark g7 428.7 vs 336.9 = x1.27. Four nodes give about x2 for latency-bound single streams and little more than x1.3 at n=16 with a speculative draft.

Speculation gain on TP=4 (DSpark g7 mean of cases 1 and 5 over control): n=1 x2.7, n=4 x2.3, n=8 x1.5, n=16 x1.2, n=32 x1.05 (575 vs 552). Single node was x3.8 / x2.8 / x2.5 / x2.0 at 1/4/8/16. The gain shrinks fast with batch size: at TP=4 the target verify is cheap and per-step overhead (draft + inter-node sync) dominates, and the control scales almost linearly with n.

Ranking on TP=4: DSpark g7 > DSpark g5 > MTP at n=1..4 and n=16; at n=8 the three are within noise (341 to 359); at n=32 DSpark g7 (556 to 588) is ahead of g5 (555), the control (553) and MTP (480). MTP is below the control at n=32 (x0.87).

Drift control (case 5 vs 1): n=1 -10 %, n=4 +0.5 %, n=8 -4 %, n=16 -1 %, n=32 -4 %. The n=1 difference is 4 questions of which the first request pays warm-up; read single-digit percent as noise.

## Decision

- Keep the profile as shipped (DSpark gamma 7, RadixArk draft). It is best or tied at every level n=1..32 on TP=4, healthy accept len (4.3 median, 5.2 at high concurrency), no graph fallbacks, no warnings beyond the KV-scale line.
- MTP (EAGLE 3/1/4) is a valid fallback but worse than the control at n=32; gamma 5 offers no advantage (it is -3 to -10 % vs g7).
- Worth a look (not tested here): at n>=16 the speculative gain on TP=4 is only x1.2 and at n=32 about zero, and the single-node TP=1 deployment of this 20.6 GB model reaches 337 tok/s at n=16 (vs 429 on 4 nodes, x1.27) while leaving three Sparks free. If the workload is mostly concurrent, a TP=1 instance (profile header suggests `sglang_instances`) is the better use of hardware; for single-stream latency TP=4 + DSpark g7 is x2.1 faster.
- FP8 KV scales not applied on 0.5.21 (cluster and single node): still open for the p51 / PR #31220 path; unaffected quality at this sample size.
- Header update proposal (not applied): item C resolved with the numbers above; item D re-measured on TP=4 (x2.7 at n=1, x1.2 at n=16, about x1.0 at n=32).
