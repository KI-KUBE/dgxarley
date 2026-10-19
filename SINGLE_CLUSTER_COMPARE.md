# Single node vs 4-node cluster: engine throughput

n=1 = single-stream tok/s (harness, one request at a time). n>1 = highest `gen throughput` (tok/s) the SGLang engine logged during the level (re-measured 2026-10-12, see "Method" below).
Slots = SGLang `max_running_requests` after the mamba-cache fit; where it is below n, the rest queues and the cell is the rate at that many running.
SGLang 0.5.21-sm121, GSM8K via chat API (`sglang-gsm8k`), temperature 0. Bold = best per model and column.

| Model                                 | Setup                | Spec config                                                           | Slots | n=1            | n=4     | n=8     | n=16    | n=32    |
|---------------------------------------|----------------------|-----------------------------------------------------------------------|------:|---------------:|--------:|--------:|--------:|--------:|
| **Qwen3.8-27B** (NVFP4)               | 1 node, TP1          | none, float32 SSM, cache 222 ¹                                        |    44 |           12.6 |      50 |      93 |     164 |     257 |
|                                       | 1 node, TP1          | DSpark gamma 7 RadixArk, float32 SSM                                  |    15 |           48.3 |     153 |     244 |     329 |     322 |
|                                       | 1 node, TP1          | DSpark gamma 7 RedHat (R0, hand-converted config), float32 SSM        |    15 |           49.3 |     152 |     233 |     320 |     319 |
|                                       | 1 node, TP1          | DSpark gamma 7 RadixArk, bfloat16 SSM, cache 152 ²                    |    30 |           45.1 |     178 |     293 | **439** |     558 |
|                                       | 1 node, TP1          | DSpark gamma 7, bfloat16 SSM, cache 160, CG bs 32 (= preset defaults) |    32 |           44.5 |     188 |     305 |     424 |     553 |
|                                       | 4 nodes, TP4         | none                                                                  |    32 |           36.0 |     129 |     235 |     381 | **559** |
|                                       | 4 nodes, TP4         | MTP 3/1/4                                                             |    32 |           69.7 |     217 | **344** |     368 |     399 |
|                                       | 4 nodes, TP4         | DSpark gamma 5                                                        |    32 |           93.3 |     270 |     331 |     323 |     437 |
|                                       | 4 nodes, TP4         | DSpark gamma 7, float32 SSM ³                                         |    32 | **93.2-103.8** |     263 |     309 |     346 |     504 |
|                                       | 4 nodes, TP4         | DSpark gamma 7, bfloat16 SSM (= profile)                              |    32 |           96.0 | **292** |     304 |     330 |     517 |
| **Qwen3.8-Flash-Next** (nvidia NVFP4) | 1 node, TP1          | none                                                                  |    25 |           16.3 |      56 |     106 |     169 |     201 |
|                                       | 1 node, TP1          | MTP 3/1/4, float32 SSM                                                |     9 |           31.7 |      81 |     124 |     147 |     154 |
|                                       | 1 node, TP1          | MTP adaptive, float32 SSM                                             |    11 |           31.3 |      77 |     125 |     135 |     167 |
|                                       | 1 node, TP1          | MTP 3/1/4, bfloat16 SSM, cache 76                                     |    19 |           34.2 |      83 |     158 |     246 |     230 |
|                                       | 1 node, TP1          | MTP 3/1/4, bfloat16 SSM, cache 128 ⁴                                  |    32 |           32.6 |      98 |     148 |     208 |     285 |
|                                       | 4 nodes, TP4 EP4     | none, float32 SSM                                                     |    16 |           34.6 |     124 |     206 |     351 |     354 |
|                                       | 4 nodes, TP4 EP4     | MTP 2/1/3 fixed, float32 SSM                                          |    16 |    62.8-68.6 ⁵ |     202 |     306 |     400 |     372 |
|                                       | 4 nodes, TP4 EP4     | MTP adaptive, float32 SSM                                             |    16 |           74.6 |     215 |     302 |     430 |     396 |
|                                       | 4 nodes, TP4 EP4     | MTP adaptive, bfloat16 SSM (= profile)                                |    32 |       **75.8** | **231** | **331** | **476** | **646** |
| **Nemotron-3-Super-120B** (NVFP4)     | 1 node, TP1 (spark5) | MTPv2 4/1/5, float32 SSM, cache 64                                    |    12 |           32.1 |      78 |     107 |     127 |     133 |
|                                       | 1 node, TP1 (spark5) | MTPv2 4/1/5, bfloat16 SSM, cache 128 (= preset)                       |    25 |           32.2 |      81 |     119 |     178 |     236 |
|                                       | 4 nodes, TP4 EP4     | MTPv2 3/1/4                                                           |    32 |           70.3 |     195 |     254 |     355 |     412 |
|                                       | 4 nodes, TP4 EP4     | MTPv2 5/1/6 ⁶                                                         |    32 |           74.1 |     192 |     267 |     299 |     404 |
|                                       | 4 nodes, TP4 EP4     | MTPv2 4/1/5, float32 SSM                                              |    32 |       **82.7** |     193 |     253 |     327 |     432 |
|                                       | 4 nodes, TP4 EP4     | MTPv2 4/1/5, bfloat16 SSM (= profile)                                 |    32 |           82.1 | **211** | **287** | **358** | **485** |

¹ No speculation leaves room for 44 mamba slots; the preset's CUDA graphs stop at batch 16, so 61 of 1990 logged decode batches ran eager.
² `--max-mamba-cache-size 152` with the then preset `--cuda-graph-max-bs-decode 16`: 12 of 301 logged decode batches ran eager.
³ n=1 = range of case 1 and repeat case 5 of the 2026-10-10 sweep.
⁴ `--max-mamba-cache-size 128` (4 slots per request -> 32). Mamba state 14.2 GB (cache 76: 8.5 GB) out of the same static pool, so the KV pool shrinks from 386k to 164k tokens. vs cache 76: +18 % n4, -6 % n8, -15 % n16, +24 % n32. GSM8K needs only a few % of the KV pool; long-context load would hit the 164k pool. Not adopted as preset default.
⁵ Range of reps 1/2 of the 2026-10-10 sweep.
⁶ Listed as "5/1/7" before 2026-10-11: with topk 1 SGLang rewrites draft tokens to steps+1, so it always ran 5/1/6.

## Cluster rerun 2026-10-19 (dual-rail, 4 NCCL channels)

Same method as 2026-10-12, but NCCL over both QSFP rails and `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS` = 4 (was 16 by default), see `NCCL_DUALRAIL_QSFP_TEST.md`. No power cycle before the run. n=1 = single-stream tok/s of this run (4 questions, Flash-Next 8). In progress; "pending" cells are not measured yet. Last column: n=32 against the 2026-10-12 cell above.

| Model                                 | Setup            | Spec config                              | Slots |     n=1 |     n=4 |     n=8 |    n=16 |    n=32 | n=32 vs 10-12 |
|---------------------------------------|------------------|------------------------------------------|------:|--------:|--------:|--------:|--------:|--------:|--------------:|
| **Qwen3.8-Flash-Next** (nvidia NVFP4) | 4 nodes, TP4 EP4 | MTP adaptive, bfloat16 SSM (= profile)   |    32 |    83.5 |     254 |     356 |     504 |     762 |         +18 % |
|                                       | 4 nodes, TP4 EP4 | MTP adaptive, float32 SSM                |       | pending | pending | pending | pending | pending |               |
|                                       | 4 nodes, TP4 EP4 | MTP 2/1/3 fixed, float32 SSM             |       | pending | pending | pending | pending | pending |               |
|                                       | 4 nodes, TP4 EP4 | none, float32 SSM                        |       | pending | pending | pending | pending | pending |               |
| **Qwen3.8-27B** (NVFP4)               | 4 nodes, TP4     | DSpark gamma 7, bfloat16 SSM (= profile) |       | pending | pending | pending | pending | pending |               |
|                                       | 4 nodes, TP4     | DSpark gamma 7, float32 SSM              |       | pending | pending | pending | pending | pending |               |
|                                       | 4 nodes, TP4     | DSpark gamma 5, float32 SSM              |       | pending | pending | pending | pending | pending |               |
|                                       | 4 nodes, TP4     | MTP 3/1/4, float32 SSM                   |       | pending | pending | pending | pending | pending |               |
|                                       | 4 nodes, TP4     | none, float32 SSM                        |       | pending | pending | pending | pending | pending |               |
| **Nemotron-3-Super-120B** (NVFP4)     | 4 nodes, TP4 EP4 | MTPv2 4/1/5, bfloat16 SSM (= profile)    |       | pending | pending | pending | pending | pending |               |
|                                       | 4 nodes, TP4 EP4 | MTPv2 4/1/5, float32 SSM                 |       | pending | pending | pending | pending | pending |               |
|                                       | 4 nodes, TP4 EP4 | MTPv2 3/1/4, float32 SSM                 |       | pending | pending | pending | pending | pending |               |
|                                       | 4 nodes, TP4 EP4 | MTPv2 5/1/6, float32 SSM                 |       | pending | pending | pending | pending | pending |               |

Flash-Next profile n=4 to n=32: run of 2026-10-19 03:09-03:21 UTC (`NCCL_DUALRAIL_QSFP_TEST.md`).

## Method (2026-10-12)

- All n>1 cells measured on 2026-10-12 after a power cycle of all Sparks. Single node: spark5, `scripts/run_standalone.sh` with the preset of the row plus env overrides and `--decode-log-interval 10`, server in the foreground of a tmux session. Cluster: one rollout per config with only that config's profile keys changed, cluster idle-checked before each deploy, default profile redeployed afterwards.
- Load: `sglang-gsm8k run` at n=4/8/16/32 with 8/16/32/64 questions (single node), 16/32/64/128 (cluster 27B, Nemotron) or 32/64/128/256 (cluster Flash-Next); max_tokens 8192, 20 s pause between levels.
- Throughput: highest `gen throughput` of the `Decode batch` lines in the level's time window (server log; cluster log interval 40 decode steps, single node 10). SGLang sums every generated token between two log lines and divides by the elapsed wall time, so each line is a measured rate over a few seconds, readable by anyone from the server log. Client-side harness rates are not used.
- n=1 cells are the single-stream rates of the 2026-10-10/11 runs (one request, no concurrency).
- Checks: cluster head-log POSTs = expected in 11 of 13 cases (Flash-Next profile 481 of 480, 27B profile 241 of 240); 0 eager decode batches except footnotes ¹ and ². GSM8K 61-64/64 (single node n=32), 122-125/128 and 250-253/256 (cluster n=32), at most 5 truncations at 8192 tokens per level; no accuracy difference between configs is visible at these counts.
- Per-config details: testlogs listed under "Sources".

## Observations

- 27B 1 node vs 4 nodes (DSpark gamma 7, bfloat16 SSM both sides): x2.2 at n=1 (44.5 -> 96.0), x1.55 at n=4 (188 -> 292), x1.0 at n=8 (305 -> 304), x0.78 at n=16 (424 -> 330), x0.93 at n=32 (553 -> 517). From n=8 on one Spark with bfloat16 SSM is as fast as four or faster. bfloat16 SSM on the single node: +23 % n=4, +25 % n=8, +29 % n=16, +72 % n=32 over float32 (15 slots); on TP4 -5 to +11 %.
- 27B on 4 nodes: speculation pays only up to n=8 (gamma 7 over the control x2.0-2.3 at n=4, x1.3 at n=8); from n=16 on the control is faster (381 vs 330-346 at n=16, 559 vs 504-517 at n=32). MTP 3/1/4 is the slowest speculative config at n=32 (399).
- Flash-Next 1 node vs 4 nodes (MTP, best per side): x2.2 at n=1 (34.2 -> 75.8), x2.4 at n=4 (98 -> 231), x2.1 at n=8 (158 -> 331), x1.9 at n=16 (246 -> 476), x2.3 at n=32 (285 -> 646). On the single node float32 caps MTP at 9-11 slots and MTP falls below no-MTP from n=16 on (147-167 vs 169-201); bfloat16 SSM lifts n=32 to 230-285. On TP4 the profile (bfloat16, 32 slots) leads from n=16 on (476 vs 351-430, n=32 646 vs 354-396); the float32 configs run 16 slots and stop scaling at n=16.
- Nemotron Super 1 node vs 4 nodes (MTPv2 4/1/5, bfloat16 SSM both sides): x2.5 at n=1 (32.2 -> 82.1), x2.6 at n=4 (81 -> 211), x2.4 at n=8 (119 -> 287), x2.0 at n=16 (178 -> 358), x2.1 at n=32 (236 -> 485). The cluster gain stays at x2 or more at every level, unlike the 27B. bfloat16 SSM on the single node: +3 % n=4, +10 % n=8, +41 % n=16, +78 % n=32 (12 -> 25 slots); on TP4 +9 to +13 %. On TP4 at n=32: 4/1/5 bf16 485 > 4/1/5 float32 432 > 3/1/4 412 and 5/1/6 404.
- 1 node, ranking: the 27B with DSpark leads at every level (n=4 188 vs 98 Flash-Next vs 81 Nemotron, n=32 558 vs 285 vs 236). Without speculation the 27B is unusable for single streams (12.6 tok/s).
- 4 nodes, ranking (best config per model): n=1 27B (93-104) > Nemotron (83) > Flash-Next (76); n=4 27B (292) > Flash-Next (231) > Nemotron (211); n=8 27B (344) ~ Flash-Next (331) > Nemotron (287); n=16 Flash-Next (476) > 27B (381) > Nemotron (358); n=32 Flash-Next (646) > 27B (559) > Nemotron (485).
- Day-to-day: against the 2026-10-11 engine logs (before the power cycle) most cluster configs moved by less than 15 % at n=32, but the Flash-Next profile went from 435 to 646 and 27B MTP 3/1/4 from 474 to 399. Cause not investigated. Differences below about 15 % between configs are not a ranking.

## Sources

- `TESTLOGS/standalone_nn1_tp1/qwen-3.8-27b-nvfp4-modelopt/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-27b-nvfp4-modelopt_1n.md`
- `TESTLOGS/standalone_nn1_tp1/qwen-3.8-flash-next-nvfp4/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-flash-next-nvfp4_1n.md`
- `TESTLOGS/standalone_nn1_tp1/nemotron-3-super-120b-a12b-nvfp4/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_nemotron-3-super-120b-a12b-nvfp4_1n.md`
- `TESTLOGS/sglang_nn4_tp4_ep4/qwen-3.8-flash-next-nvfp4/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-flash-next-nvfp4_4n.md`
- `TESTLOGS/sglang_nn4_tp4_ep1/qwen-3.8-27b-nvfp4-modelopt/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-27b-nvfp4-modelopt_4n.md`
- `TESTLOGS/sglang_nn4_tp4_ep1/nemotron-3-super-120b-a12b-nvfp4/TESTLOG_sglang-0.5.21-sm121_nemotron-3-super-120b-a12b-nvfp4-mtpv2_4n.md`
