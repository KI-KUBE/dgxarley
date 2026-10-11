# Single node vs 4-node cluster: peak throughput

Peak tok/s = harness `concurrent_peak_tok_s` (sum of concurrent per-request tok/s; n=1 = single-stream tok/s).
Best measured configuration per setup, SGLang 0.5.21-sm121, GSM8K via chat API (`sglang-gsm8k`), temperature 0.

| Model | Setup | Spec config | n=1 | n=4 | n=8 | n=16 | n=32 |
|---|---|---|---:|---:|---:|---:|---:|
| **Qwen3.8-27B** (NVFP4) | 1 node, TP1 | none | 12.6 | 49.4 | 94.2 | 166.3 | - |
| | 1 node, TP1 | DSpark gamma 7 RadixArk (= profile) | 48.3 | 138.8 | 231.7 | 336.9 ¹ | - |
| | 1 node, TP1 | DSpark gamma 7 RedHat (R0, hand-converted config) | 49.3 | 147.3 | 228.6 | 346.1 ¹ | - |
| | 1 node, TP1 | DSpark gamma 7 RadixArk, bfloat16 SSM | 45.1 | **160.6** | **272.7** | **437.2** ⁷ | 569.1 ⁷ |
| | 1 node, TP1 | DSpark gamma 7, bfloat16 SSM, cache 160, CG bs 32 (= preset defaults) | 44.5 | 157.6 | 262.7 | 437.0 | **598.6** ⁹ |
| | 4 nodes, TP4 | none | 36.0 | 126.8 | 230.3 | 360.8 | 552.5 |
| | 4 nodes, TP4 | MTP 3/1/4 | 69.7 | 214.5 | **358.5** | 389.8 | 479.6 |
| | 4 nodes, TP4 | DSpark gamma 5 | 93.3 | 279.7 | 345.1 | 394.6 | 555.4 |
| | 4 nodes, TP4 | **DSpark gamma 7** (= profile, case 1 / repeat case 5) | **93.2-103.8** | **293.3-294.7** | 341.3-355.5 | **423.5-428.7** | **563.0-587.6** |
| | 4 nodes, TP4 | DSpark gamma 7, bfloat16 SSM, max_running 32 ¹⁰ | 96.0 | 265.2 | **388.4** | 363.5 | 562.7 |
| **Qwen3.8-Flash-Next** (nvidia NVFP4) | 1 node, TP1 | none | 16.3 | 51.6 | 85.4 | 119.3 | 133.9 |
| | 1 node, TP1 | MTP 3/1/4 | 31.7 | 83.6 | 130.8 | 188.1 ² | 239.7 ² |
| | 1 node, TP1 | MTP adaptive | 31.3 | 82.1 | 128.3 | 165.4 ² | 246.8 ² |
| | 1 node, TP1 | MTP 3/1/4, bfloat16 SSM | **34.2** | **89.6** | **141.7** | **186.2** ⁸ | **208.4** ⁸ |
| | 4 nodes, TP4 EP4 | none | 34.6 | 116.0 | 187.9 | 284.5 | - |
| | 4 nodes, TP4 EP4 | MTP 2/1/3 (shipped, rep 3) | 62.8-68.6 ³ | 152.6-178.8 ³ | 276.9 | 419.5 | - |
| | 4 nodes, TP4 EP4 | MTP adaptive (B rep 2) | 74.6 | **212.9** | **317.2** | **469.7** | - |
| | 4 nodes, TP4 EP4 | MTP adaptive, bfloat16 SSM, max_running 32 ¹¹ | **75.8** | 201.6 | 300.9 | 367.0 | **412.3** |
| **Nemotron-3-Super-120B** (NVFP4) | 1 node, TP1 (spark5) | MTPv2 4/1/5, float32 SSM | 32.1 | 73.6 | 107.0 | 145.8 ⁴ | 178.9 ⁴ |
| | 1 node, TP1 (spark5) | MTPv2 4/1/5, bfloat16 SSM | 32.2 | 77.8 | 114.3 | 173.8 | 219.3 ⁶ |
| | 4 nodes, TP4 EP4 | MTPv2 3/1/4 (first sweep) | 70.3 | 201.2 | 267.3 | 306.0 | 328.1 |
| | 4 nodes, TP4 EP4 | MTPv2 5/1/7 | 74.1 | 200.2 | 257.3 | 342.6 | 431.8 |
| | 4 nodes, TP4 EP4 | **MTPv2 4/1/5** (sweep winner) | **82.7** | **202.6** ⁵ | **277.4** | **377.2** | **513.2** |
| | 4 nodes, TP4 EP4 | MTPv2 4/1/5, bfloat16 SSM ¹² | 82.1 | **207.8** | 266.3 | **419.3** | **541.8** |

¹ gamma 7 caps `max_running_requests` at 15 (mamba state cache), one request queued at n=16.
² With MTP the effective concurrency is only 9-11 (mamba cache), the rest queues; queue-limited numbers.
³ Range of reps 1/2; rep 3 did not run n=1/n=4.
⁴ `max_running_requests` capped at 12 by the mamba state cache (`--max-mamba-cache-size 64`, 5 slots per request, float32 SSM state, `mem_fraction_static 0.90`); n=16/32 queue. With the auto-sized cache (24 slots) the cap was 4.
⁵ Measured separately 2026-10-11 (testlog Run 6, 16 questions); drift anchors in that run: n=1 85.9 (+3.9 %), n=8 299.4 (+7.9 %) against the sweep.
⁶ bfloat16 SSM state with `--max-mamba-cache-size 128` caps `max_running_requests` at 25 (same cache memory as float32 x 64); only n=32 queues.
⁷ bfloat16 SSM, `--max-mamba-cache-size 152` -> `max_running_requests` 30. The preset's `--cuda-graph-max-bs-decode 16` (MAX_RUNNING default) makes batches above 16 run eager (3 of 129 logged decode batches, at 17/28/30 running), so n=16/32 are a lower bound.
⁸ bfloat16 SSM, `--max-mamba-cache-size 76` -> `max_running_requests` 19 (float32 auto-fit: 9); n=32 still queues.
⁹ Preset defaults since 2026-10-11 (`--max-mamba-cache-size 160 --mamba-ssm-dtype bfloat16 --cuda-graph-max-bs-decode 32`): `max_running_requests` 32, 0 eager decode batches, accept len mean 4.22, probe 0/16 collapsed, tool calls 4/4, GSM8K 4/4, 8/8, 15/16, 31/32, 62/64 (misses id 12 and 45, as in the run above). Only n=32 moves (+5 %, within noise); the eager batches above 16 cost little.
¹⁰ 2026-10-11 rerun after the profile got `mamba_ssm_dtype: bfloat16` + `max_running_requests: 32` (same levels/questions as the float32 sweep). 32 running, `#queue-req` max 3, 0 eager decode batches, accept len mean 4.28, probe 0/16 collapsed, tool calls 4/4, GSM8K 4/4, 15/16, 30/32, 62/64, 122/128. Mixed vs float32 (n=8 higher, n=4/n=16 lower): noise, as expected, since the mamba cache was never the limit at TP4 (34.6 GB still free after boot). Engine gen throughput at 28-32 running: mean 421, max 539 tok/s.
¹¹ 2026-10-11, profile with `mamba_ssm_dtype: bfloat16` + `max_running_requests: 32` (was 16), adaptive MTP as shipped; levels 8/32/64/128/256 questions at n=1/4/8/16/32 (same as the 10-10 A/B, n=32 new). 32 running, `#queue-req` max 2, 0 eager decode batches, accept len mean 2.96 (B: 3.1), probe 0/16 collapsed, tool calls 4/4, GSM8K 8/8, 31/32, 63/64, 124/128, 249/256 (+3 truncations). No foreign requests during the bench. n=1..16 within the 10-10 run-to-run spread (A varied up to x1.59 at n=16); n=32 412 tok/s. Engine gen throughput at 28-32 running: mean 299, max 406 tok/s (14-27 running: mean 283), so the extra slots above 16 add little on TP4. Only 8.4 GB free after boot.
¹² 2026-10-11, profile with `mamba_ssm_dtype: bfloat16` (cache 160 slots, `max_running_requests` 32), same levels as the sweep (4/16/32/64/128 questions). 32 running, `#queue-req` max 2, 0 eager decode batches, accept len mean 4.04 (sweep 4.02), probe 0/16 collapsed, tool calls 4/4, GSM8K 4/4, 16/16, 31/32, 63/64, 124/128, no truncations. 260 POSTs in the head log = exactly the probe plus bench, no foreign requests. 15.45 GB free after boot (sweep 15.5). vs float32: -1 % n=1, +3 % n=4, -4 % n=8, +11 % n=16, +6 % n=32, within the run-to-run spread; engine gen throughput at 28-32 running mean 396, max 505 tok/s (float32 sweep 352 / 429).

Nemotron single-node run (2026-10-11): `scripts/run_standalone.sh PRESET=nemotron3_super` (MEM_FRACTION=0.90, extra arg `--max-mamba-cache-size 64`), image 0.5.21-sm121, context 262144, KV fp8 (728k tokens). Levels n=1/4/8/16/32 with 4/16/32/64/128 questions, max_tokens 8192. GSM8K 4/4, 16/16, 32/32, 64/64, 126/128 (2 wrong answers, no truncations), no char runs, 0 eager decode batches, accept len 4.0-4.2 at every batch size (cluster: 4.0-4.1).

bfloat16 SSM rerun (same day, same preset plus `--mamba-ssm-dtype bfloat16 --max-mamba-cache-size 128`, now the preset defaults): KV 693k tokens, GSM8K 4/4, 15/16, 31/32, 64/64, 125/128 (+1 truncation at 8192 tokens); the n=4/n=8 miss is the usual id 12. Accept len 4.0-4.3 at every batch size (float32: 4.0-4.2), 0 eager decode batches, no char runs. bfloat16 does not hurt MTP; the gain is concurrency: +19 % at n=16, +23 % at n=32, n<=8 within noise.

Qwen3.8 bfloat16 SSM reruns (2026-10-11, spark5, same image/presets as the 2026-10-10 single-node logs, same question counts per level 4/8/16/32/64, n=32 new for the 27B; collapse probe as in the Flash-Next testlog):
- Flash-Next MTP 3/1/4 (`MAMBA_SSM_DTYPE=bfloat16 MAX_RUNNING=32 --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 76`): GSM8K 4/4, 8/8, 15/16, 31/32, 63/64 (only miss id 12, as in every earlier case), probe 0/16 collapsed, tool calls 4/4, accept len mean 3.03 (float32 3.04), 0 eager decode batches. The preset's "non-float32 breaks spec decode" claim does not reproduce. Peak vs float32 3/1/4: +8 % n=1, +7 % n=4, +8 % n=8, -1 % n=16, -13 % n=32 (both n=16/32 are queue-limited; float32 ran 9 concurrent, bf16 19, yet n=32 is lower, so read n>=16 as noise).
- 27B DSpark gamma 7 (`MAMBA_SSM_DTYPE=bfloat16 --max-mamba-cache-size 152`): GSM8K 4/4, 8/8, 15/16, 30/32 (+1 truncation), 62/64, probe 0/16 collapsed, tool calls 4/4, accept len mean 4.25 (float32 4.31), FP8 KV scale warning still present (known, unchanged). Peak vs float32 D7: -7 % n=1, +16 % n=4, +18 % n=8, +30 % n=16; n=32 569 tok/s is new.

## Observations

- Flash-Next 1 node vs 4 nodes (MTP, best per side; single node = bfloat16 SSM 3/1/4): x2.2 at n=1 (34.2 -> 74.6-75.8), x2.2-2.4 at n=4 (89.6 -> 201.6-212.9), x2.1-2.2 at n=8 (141.7 -> 300.9-317.2), x2.0-2.5 at n=16 (186 -> 367-470; the single node runs at most 19 concurrent, the rest queues). At n=32 the cluster (412) is x2.0 over the queue-limited single node (208); engine gen throughput at full load is 299 tok/s (cluster, 28-32 running) vs 143 (single node, 17-19 running).
- 27B 1 node vs 4 nodes (DSpark gamma 7): x2.1 at n=1 and n=4, x1.5 at n=8, x1.27 at n=16 (float32 single node); against the bfloat16 single node (preset defaults) the cluster is only x1.1 at n=8 (263 vs 341-356), x1.0 at n=16 (437 vs 424-429) and x0.96-1.0 at n=32 (599 vs 563-588). For concurrent load one Spark with bfloat16 SSM matches four.
- 27B single node vs Flash-Next single node: the dense 27B with DSpark is faster at every concurrency (n=1 48 vs 32, n=16 337 vs 188; with bfloat16 SSM n=16 437 vs 186, n=32 569 vs 208). Without speculation it is unusable (12.6 tok/s per stream).
- Nemotron Super 1 node vs 4 nodes (MTPv2 4/1/5; single node = bfloat16 SSM): x2.6 at n=1 (32.2 -> 82.7), x2.6 at n=4 (77.8 -> 202.6), x2.4 at n=8 (114.3 -> 277.4), x2.2 at n=16 (173.8 -> 377.2), x2.3 at n=32 (219.3 -> 513.2, single node capped at 25 concurrent). The cluster gain stays above x2 at every level, unlike the 27B. At n=1 the single node equals Flash-Next single node (32 vs 34) and is well below the dense 27B with DSpark (45-48).
- 4 nodes, ranking by level: n=1 27B (93-104) > Nemotron (83) > Flash-Next (75-76); n=4 27B (265-295) > Flash-Next (202-213) = Nemotron (203); n=8 27B (341-388) > Flash-Next (301-317) > Nemotron (277), adjacent gaps about 10-20 %; n=16 all three within the run-to-run spread (Flash-Next 367-470, 27B 364-429, Nemotron 377); n=32 27B (563-588) > Nemotron (513) > Flash-Next (412), confirmed by the engine gen throughput at 28-32 running (27B 340-421, Nemotron 352, Flash-Next 299 tok/s mean).
- n=32 caveat: `concurrent_peak_tok_s` sums each request's lifetime-average rate over the overlapping set. At n=32 concurrency collapses after the first wave (27B single node: 32 running for ~30 s, then 15, then 2-6 for minutes), and the long-tail requests that run nearly alone carry high average rates into that sum, so n=32 peaks overstate the instantaneous throughput at full concurrency. Cross-check with the engine's own `gen throughput` at 28-32 running requests: 27B single node (bf16, preset defaults) ~350 tok/s mean (max 530, 3 samples), 4-node TP4 ~340-410 mean (max 470-495, 5-6 samples). The single node did run 32 concurrent there (no cap queueing, `#queue-req` only 1-3 transiently). Same check for the other two at their concurrency cap: Flash-Next cluster 299 mean (peak 412), single node 143 at 17-19 running (peak 208); Nemotron cluster 352 mean at 28-32 running (8 samples, 2026-10-08 sweep head log; peak 513), single node 169 at 23-25 running (bf16, peak 219) and 109 at 10-12 running (float32, peak 179). The peak overstates the engine rate by about 30-70 % at n=32 on every setup, so compare n=32 across setups via these engine numbers. Question counts also differ at n=32 (single node 64, cluster 128; Flash-Next cluster 256).
- Run-to-run spread at n>=8 is large (Flash-Next same config up to x1.59 at n=16; Nemotron sweep vs A/B up to 20 %). Differences below about 20 % are not a ranking.

## Sources

- `TESTLOGS/standalone_nn1_tp1/qwen-3.8-27b-nvfp4-modelopt/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-27b-nvfp4-modelopt_1n.md`
- `TESTLOGS/standalone_nn1_tp1/qwen-3.8-flash-next-nvfp4/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-flash-next-nvfp4_1n.md`
- `TESTLOGS/sglang_nn4_tp4_ep4/qwen-3.8-flash-next-nvfp4/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-flash-next-nvfp4_4n.md`
- `TESTLOGS/sglang_nn4_tp4_ep1/qwen-3.8-27b-nvfp4-modelopt/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-27b-nvfp4-modelopt_4n.md`
- `TESTLOGS/sglang_nn4_tp4_ep1/nemotron-3-super-120b-a12b-nvfp4/TESTLOG_sglang-0.5.21-sm121_nemotron-3-super-120b-a12b-nvfp4-mtpv2_4n.md`
