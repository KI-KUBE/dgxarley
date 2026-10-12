# Single node vs 4-node cluster: full-load throughput

n=1 = single-stream tok/s (harness `concurrent_peak_tok_s`, exact at n=1). n>1 = `full_load_tok_s`: streamed token rate while at least 90 % of the client slots are busy (all n>1 cells re-measured 2026-10-11, see "Rerun" below).
Slots = SGLang `max_running_requests` after the mamba-cache fit; where it is below n, the rest queues and the cell is the rate at that many running.
Best measured configuration per setup, SGLang 0.5.21-sm121, GSM8K via chat API (`sglang-gsm8k`), temperature 0.

| Model                                 | Setup                | Spec config                                                           | Slots |            n=1 |       n=4 |       n=8 |      n=16 |      n=32 |
|---------------------------------------|----------------------|-----------------------------------------------------------------------|------:|---------------:|----------:|----------:|----------:|----------:|
| **Qwen3.8-27B** (NVFP4)               | 1 node, TP1          | none, float32 SSM, cache 222 ¹                                        |    44 |           12.6 |      49.3 |      94.4 |     157.2 |     245.7 |
|                                       | 1 node, TP1          | DSpark gamma 7 RadixArk, float32 SSM                                  |    15 |           48.3 |     128.4 |     205.4 |     261.9 |     279.1 |
|                                       | 1 node, TP1          | DSpark gamma 7 RedHat (R0, hand-converted config), float32 SSM        |    15 |           49.3 |     120.3 |     191.6 |     267.3 |     280.6 |
|                                       | 1 node, TP1          | DSpark gamma 7 RadixArk, bfloat16 SSM, cache 152 ²                    |    30 |           45.1 |     143.1 |     239.8 |     372.9 |     499.0 |
|                                       | 1 node, TP1          | DSpark gamma 7, bfloat16 SSM, cache 160, CG bs 32 (= preset defaults) |    32 |           44.5 |     145.1 |     240.4 | **378.2** | **512.1** |
|                                       | 4 nodes, TP4         | none                                                                  |    32 |           36.0 |     113.9 |     211.2 |     322.6 |     449.2 |
|                                       | 4 nodes, TP4         | MTP 3/1/4                                                             |    32 |           69.7 |     191.7 | **298.6** |     330.7 |     373.4 |
|                                       | 4 nodes, TP4         | DSpark gamma 5                                                        |    32 |           93.3 |     244.0 |     292.2 |     285.0 |     425.6 |
|                                       | 4 nodes, TP4         | DSpark gamma 7, float32 SSM ³                                         |    32 | **93.2-103.8** |     213.0 |     284.3 |     312.6 |     464.7 |
|                                       | 4 nodes, TP4         | DSpark gamma 7, bfloat16 SSM (= profile)                              |    32 |           96.0 | **262.0** |     284.7 |     334.4 |     520.2 |
| **Qwen3.8-Flash-Next** (nvidia NVFP4) | 1 node, TP1          | none                                                                  |    25 |           16.3 |      48.8 |      77.0 |     110.9 |     116.5 |
|                                       | 1 node, TP1          | MTP 3/1/4, float32 SSM                                                |     9 |           31.7 |      63.9 |     106.8 |     110.6 |     115.8 |
|                                       | 1 node, TP1          | MTP adaptive, float32 SSM                                             |    11 |           31.3 |      62.0 |     103.2 |     120.5 |     121.0 |
|                                       | 1 node, TP1          | MTP 3/1/4, bfloat16 SSM, cache 76                                     |    19 |       **34.2** |      68.2 | **116.8** |     138.4 |     166.7 |
|                                       | 1 node, TP1          | MTP 3/1/4, bfloat16 SSM, cache 128 ⁴                                  |    32 |           32.6 |  **71.2** |     109.4 | **155.2** | **183.0** |
|                                       | 4 nodes, TP4 EP4     | none, float32 SSM                                                     |    16 |           34.6 |     106.9 |     162.9 |     249.7 |     243.8 |
|                                       | 4 nodes, TP4 EP4     | MTP 2/1/3 fixed, float32 SSM                                          |    16 |    62.8-68.6 ⁵ |     155.9 |     220.8 | **283.0** |     267.5 |
|                                       | 4 nodes, TP4 EP4     | MTP adaptive, float32 SSM                                             |    16 |           74.6 | **183.2** | **234.6** |     270.1 |     272.4 |
|                                       | 4 nodes, TP4 EP4     | MTP adaptive, bfloat16 SSM (= profile)                                |    32 |       **75.8** |     155.3 |     204.5 |     280.7 | **300.1** |
| **Nemotron-3-Super-120B** (NVFP4)     | 1 node, TP1 (spark5) | MTPv2 4/1/5, float32 SSM, cache 64                                    |    12 |           32.1 |      63.6 |      84.5 |     110.2 |     108.4 |
|                                       | 1 node, TP1 (spark5) | MTPv2 4/1/5, bfloat16 SSM, cache 128 (= preset)                       |    25 |           32.2 |      67.7 |      91.5 |     146.6 |     189.0 |
|                                       | 4 nodes, TP4 EP4     | MTPv2 3/1/4                                                           |    32 |           70.3 |     154.9 |     203.5 |     283.2 |     307.1 |
|                                       | 4 nodes, TP4 EP4     | MTPv2 5/1/6 ⁶                                                         |    32 |           74.1 |     145.6 |     192.0 |     273.5 |     351.5 |
|                                       | 4 nodes, TP4 EP4     | MTPv2 4/1/5, float32 SSM                                              |    32 |       **82.7** |     157.0 |     213.6 |     245.3 |     378.0 |
|                                       | 4 nodes, TP4 EP4     | MTPv2 4/1/5, bfloat16 SSM (= profile)                                 |    32 |           82.1 | **158.2** | **219.9** | **299.5** | **400.5** |

¹ No speculation leaves room for 44 mamba slots; the preset's CUDA graphs stop at batch 16, so 16 decode batches ran eager at n=32 (lower bound).
² `--max-mamba-cache-size 152` with the then preset `--cuda-graph-max-bs-decode 16`: 3 eager decode batches above 16 running.
³ n=1 = range of case 1 and repeat case 5 of the 2026-10-10 sweep.
⁴ `--max-mamba-cache-size 128` (4 slots per request -> 32). Mamba state 14.2 GB (cache 76: 8.5 GB) out of the same static pool, so the KV pool shrinks from 388k to 160k tokens. vs cache 76: +4 % n=4, -6 % n=8, +12 % n=16, +10 % n=32, i.e. 13 more slots buy about 10 % at n>=16. GSM8K needs only a few % of the KV pool; long-context load would hit the 160k pool. Not adopted as preset default.
⁵ Range of reps 1/2 of the 2026-10-10 sweep.
⁶ Listed as "5/1/7" before 2026-10-11: with topk 1 SGLang rewrites draft tokens to steps+1, so it always ran 5/1/6.

## Best 10 s window / engine max

Same runs as the n>1 cells above, tok/s. Best 10 s window = highest 10 s streamed token rate of the level (character-weighted timeline; the token-exact timeline via `continuous_usage_stats` came after these runs). Engine max = highest head-log `gen throughput` in the level's time window.

| Model                 | Setup   | Spec config                                 | n=4       | n=8       | n=16      | n=32      |
|-----------------------|---------|---------------------------------------------|----------:|----------:|----------:|----------:|
| Qwen3.8-27B           | 1 node  | none, float32 SSM, cache 222                |   57 / 50 |  106 / 93 | 194 / 162 | 314 / 257 |
|                       | 1 node  | DSpark g7 RadixArk, float32 SSM             | 137 / 127 | 219 / 209 | 286 / 278 | 308 / 289 |
|                       | 1 node  | DSpark g7 RedHat R0, float32 SSM            | 132 / 114 | 208 / 197 | 310 / 277 | 320 / 279 |
|                       | 1 node  | DSpark g7 RadixArk, bfloat16 SSM, cache 152 | 150 / 142 | 265 / 253 | 408 / 390 | 575 / 515 |
|                       | 1 node  | DSpark g7, bfloat16 SSM, cache 160 (preset) | 160 / 157 | 254 / 239 | 405 / 386 | 587 / 501 |
|                       | 4 nodes | none                                        | 130 / 128 | 224 / 233 | 358 / 377 | 500 / 518 |
|                       | 4 nodes | MTP 3/1/4                                   | 211 / 226 | 331 / 350 | 376 / 385 | 456 / 474 |
|                       | 4 nodes | DSpark g5                                   | 261 / 269 | 334 / 328 | 328 / 322 | 474 / 462 |
|                       | 4 nodes | DSpark g7, float32 SSM                      | 254 / 269 | 321 / 323 | 355 / 333 | 513 / 490 |
|                       | 4 nodes | DSpark g7, bfloat16 SSM (profile)           | 287 / 330 | 308 / 310 | 380 / 353 | 618 / 542 |
| Qwen3.8-Flash-Next    | 1 node  | none                                        |   54 / 52 |  100 / 85 | 170 / 137 | 175 / 135 |
|                       | 1 node  | MTP 3/1/4, float32 SSM                      |   74 / 70 | 127 / 111 | 144 / 122 | 152 / 129 |
|                       | 1 node  | MTP adaptive, float32 SSM                   |   74 / 69 | 131 / 104 | 163 / 140 | 171 / 132 |
|                       | 1 node  | MTP 3/1/4, bfloat16 SSM, cache 76           |   79 / 81 | 136 / 125 | 172 / 148 | 248 / 173 |
|                       | 1 node  | MTP 3/1/4, bfloat16 SSM, cache 128          |   84 / 80 | 123 / 122 | 233 / 171 | 228 / 214 |
|                       | 4 nodes | none, float32 SSM                           | 125 / 122 | 197 / 211 | 291 / 309 | 331 / 329 |
|                       | 4 nodes | MTP 2/1/3 fixed, float32 SSM                | 188 / 202 | 280 / 290 | 421 / 431 | 389 / 435 |
|                       | 4 nodes | MTP adaptive, float32 SSM                   | 209 / 216 | 292 / 320 | 378 / 376 | 381 / 416 |
|                       | 4 nodes | MTP adaptive, bfloat16 SSM (profile)        | 181 / 221 | 303 / 297 | 418 / 413 | 452 / 435 |
| Nemotron-3-Super-120B | 1 node  | MTPv2 4/1/5, float32 SSM                    |   73 / 66 |  108 / 91 | 131 / 111 | 136 / 116 |
|                       | 1 node  | MTPv2 4/1/5, bfloat16 SSM (preset)          |   77 / 71 |  111 / 97 | 182 / 151 | 226 / 196 |
|                       | 4 nodes | MTPv2 3/1/4                                 | 180 / 181 | 262 / 251 | 349 / 340 | 463 / 412 |
|                       | 4 nodes | MTPv2 5/1/6                                 | 176 / 178 | 255 / 255 | 343 / 316 | 476 / 446 |
|                       | 4 nodes | MTPv2 4/1/5, float32 SSM                    | 182 / 193 | 258 / 264 | 331 / 318 | 513 / 460 |
|                       | 4 nodes | MTPv2 4/1/5, bfloat16 SSM (profile)         | 179 / 198 | 283 / 281 | 377 / 371 | 524 / 493 |

## Rerun 2026-10-11

Every n>1 cell was re-measured on one day with the streaming harness (`sglang-gsm8k run`, streams by default; `analyze` -> `full_load_tok_s`, `sustained_10s_tok_s`). n=1 cells are the older single-stream values (one request, no overlap, not affected by the peak bias).
- Questions per level: single node 8/16/32/64 at n=4/8/16/32; cluster 16/32/64/128 (27B, Nemotron) and 32/64/128/256 (Flash-Next). max_tokens 8192.
- Single node: spark5 `scripts/run_standalone.sh` with the preset of the row plus env overrides. Cluster: model profile edits deployed from a scratch copy of the repo; the cluster is back on the Flash-Next profile.
- Cluster head log: POST count = expected in 12 of 13 cases (27B gamma 7 float32: 241 of 240, one foreign request), 0 eager decode batches in all 13. Single node: 0 eager decode batches except footnotes ¹ and ².
- GSM8K: 61-64/64, 30-32/32, 14-16/16, 8/8 single node; 122-126/128, 250-252/256 cluster; at most 4 truncations at 8192 tokens per level. No accuracy difference between configs is visible at these counts.
- Engine cross-check (head-log `gen throughput`, decode batches at >= 90 % of min(n, slots) running, within the level's time window): cluster full_load is 0-10 % above the engine mean for no-spec and MTP/EAGLE (e.g. Flash-Next adaptive bf16 n=32 300 vs 293, 27B none 449 vs 438, Nemotron 4/1/5 bf16 401 vs 387) and 10-25 % above for 27B DSpark (gamma 7 bf16 n=32 520 vs 454). The single node logs a decode batch only every 40 steps, so the spec runs have 1-5 samples per level (some covering the ramp); where there are 9-16 samples (no spec, capped runs) full_load is 1-15 % above engine.
- The old `concurrent_peak_tok_s` of the same runs is 1.0-1.3x full_load where all requests run at once and up to 1.6-2.1x where slots < n (Flash-Next float32 MTP, Nemotron float32, 27B float32 DSpark at n=32): the peak averaged the queue away. Old peak values are in git history (commit 547975f and earlier).

## Observations

- 27B 1 node vs 4 nodes (DSpark gamma 7, bfloat16 SSM both sides): x2.2-2.3 at n=1 (44.5 -> 96.0-103.8), x1.8 at n=4 (145 -> 262), x1.2 at n=8 (240 -> 285-299 best cluster config), x0.88 at n=16 (378 -> 334), x1.0 at n=32 (512 -> 520). From n=16 on one Spark with bfloat16 SSM matches four. On the single node bfloat16 SSM is worth +13 % at n=4 up to +83 % at n=32 over float32 (15 slots), on TP4 +0-23 %.
- Flash-Next 1 node vs 4 nodes (MTP, best per side): x2.2 at n=1 (34.2 -> 75.8), x2.6 at n=4 (71 -> 183), x2.0 at n=8 (117 -> 235), x1.8 at n=16 (155 -> 283), x1.6 at n=32 (183 -> 300). On the single node float32 caps MTP at 9-11 slots, so n=16 and n=32 stay at 111-121 tok/s; bfloat16 SSM lifts n=32 to 167-183. On TP4 float32 adaptive (16 slots) and bfloat16 adaptive (32 slots) are within the spread at every level (n=4/8 float32 15-18 % higher, n=16/32 bfloat16 4-10 % higher), the slots above 16 add little.
- Nemotron Super 1 node vs 4 nodes (MTPv2 4/1/5, bfloat16 SSM both sides): x2.6 at n=1 (32.2 -> 82.1), x2.3 at n=4 (68 -> 158), x2.4 at n=8 (92 -> 220), x2.0 at n=16 (147 -> 300), x2.1 at n=32 (189 -> 401). The cluster gain stays at about x2 or more at every level, unlike the 27B. bfloat16 SSM on the single node: +6 % n=4, +8 % n=8, +33 % n=16, +74 % n=32 (12 -> 25 slots); on TP4 +1-22 %.
- 1 node, ranking: the 27B with DSpark leads at every level (n=4 145 vs 71 Flash-Next vs 68 Nemotron, n=32 512 vs 183 vs 189). Flash-Next and Nemotron are within 6 % of each other except n=8 (Flash-Next +27 %). Without speculation the 27B is unusable (12.6 tok/s per stream).
- 4 nodes, ranking: n=1 27B (93-104) > Nemotron (82) > Flash-Next (76); n=4 27B (262) > Flash-Next (155-183) = Nemotron (158); n=8 27B (285-299) > Flash-Next (205-235) = Nemotron (220); n=16 all three within 20 % (27B 313-334, Nemotron 300, Flash-Next 270-283); n=32 27B (520) > Nemotron (401) > Flash-Next (300). The engine cross-check shrinks the 27B lead at n=32 to 454 vs 387 (+17 %), the order holds.
- Spec config on 4 nodes: 27B without speculation reaches 449 at n=32, above MTP 3/1/4 (373) and close to DSpark gamma 5/7 (426-520); at n=32 speculation is worth at most +16 % there, at n=4 x2.3. Nemotron 4/1/5 > 5/1/6 > 3/1/4 at n=32 (378-401 > 352 > 307), at n<=16 the three are within the spread.
- Run-to-run spread: configs that should be close differ by up to 20 % (Flash-Next adaptive float32 vs bf16 at n=4, Nemotron 4/1/5 float32 at n=16 245 vs 3/1/4 283 and 5/1/6 274). Differences below about 20 % are not a ranking.
- full_load window length: as short as 11-17 s for the 27B single node at n=4/8, 20-400 s elsewhere; the short windows carry the most noise.

## Sources

- `TESTLOGS/standalone_nn1_tp1/qwen-3.8-27b-nvfp4-modelopt/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-27b-nvfp4-modelopt_1n.md`
- `TESTLOGS/standalone_nn1_tp1/qwen-3.8-flash-next-nvfp4/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-flash-next-nvfp4_1n.md`
- `TESTLOGS/sglang_nn4_tp4_ep4/qwen-3.8-flash-next-nvfp4/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-flash-next-nvfp4_4n.md`
- `TESTLOGS/sglang_nn4_tp4_ep1/qwen-3.8-27b-nvfp4-modelopt/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-27b-nvfp4-modelopt_4n.md`
- `TESTLOGS/sglang_nn4_tp4_ep1/nemotron-3-super-120b-a12b-nvfp4/TESTLOG_sglang-0.5.21-sm121_nemotron-3-super-120b-a12b-nvfp4-mtpv2_4n.md`
- n=1 values and the older peak tables are in those testlogs; the n>1 values above supersede their peak numbers.
