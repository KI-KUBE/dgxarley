# Single node vs 4-node cluster: peak throughput

Peak tok/s = harness `concurrent_peak_tok_s` (sum of concurrent per-request tok/s; n=1 = single-stream tok/s).
Best measured configuration per setup, SGLang 0.5.21-sm121, GSM8K via chat API (`sglang-gsm8k`), temperature 0.

| Model | Setup | Spec config | n=1 | n=4 | n=8 | n=16 | n=32 |
|---|---|---|---:|---:|---:|---:|---:|
| **Qwen3.8-27B** (NVFP4) | 1 node, TP1 | none | 12.6 | 49.4 | 94.2 | 166.3 | - |
| | 1 node, TP1 | DSpark gamma 7 RadixArk (= profile) | 48.3 | 138.8 | 231.7 | 336.9 ¹ | - |
| | 1 node, TP1 | DSpark gamma 7 RedHat (R0, hand-converted config) | 49.3 | 147.3 | 228.6 | **346.1** ¹ | - |
| | 4 nodes | - | *not measured* | | | | |
| **Qwen3.8-Flash-Next** (nvidia NVFP4) | 1 node, TP1 | none | 16.3 | 51.6 | 85.4 | 119.3 | 133.9 |
| | 1 node, TP1 | MTP 3/1/4 | 31.7 | 83.6 | 130.8 | 188.1 ² | 239.7 ² |
| | 1 node, TP1 | MTP adaptive | 31.3 | 82.1 | 128.3 | 165.4 ² | **246.8** ² |
| | 4 nodes, TP4 EP4 | none | 34.6 | 116.0 | 187.9 | 284.5 | - |
| | 4 nodes, TP4 EP4 | MTP 2/1/3 (shipped, rep 3) | 62.8-68.6 ³ | 152.6-178.8 ³ | 276.9 | 419.5 | - |
| | 4 nodes, TP4 EP4 | MTP adaptive (B rep 2) | **74.6** | **212.9** | **317.2** | **469.7** | - |
| **Nemotron-3-Super-120B** (NVFP4) | 1 node, TP1 (spark5) | MTPv2 4/1/5, float32 SSM | 32.1 | 73.6 | 107.0 | 145.8 ⁴ | 178.9 ⁴ |
| | 1 node, TP1 (spark5) | MTPv2 4/1/5, bfloat16 SSM | 32.2 | 77.8 | 114.3 | 173.8 | 219.3 ⁶ |
| | 4 nodes, TP4 EP4 | MTPv2 3/1/4 (first sweep) | 70.3 | 201.2 | 267.3 | 306.0 | 328.1 |
| | 4 nodes, TP4 EP4 | MTPv2 5/1/7 | 74.1 | 200.2 | 257.3 | 342.6 | 431.8 |
| | 4 nodes, TP4 EP4 | **MTPv2 4/1/5** (sweep winner) | **82.7** | **202.6** ⁵ | **277.4** | **377.2** | **513.2** |

¹ gamma 7 caps `max_running_requests` at 15 (mamba state cache), one request queued at n=16.
² With MTP the effective concurrency is only 9-11 (mamba cache), the rest queues; queue-limited numbers.
³ Range of reps 1/2; rep 3 did not run n=1/n=4.
⁴ `max_running_requests` capped at 12 by the mamba state cache (`--max-mamba-cache-size 64`, 5 slots per request, float32 SSM state, `mem_fraction_static 0.90`); n=16/32 queue. With the auto-sized cache (24 slots) the cap was 4.
⁵ Measured separately 2026-10-11 (testlog Run 6, 16 questions); drift anchors in that run: n=1 85.9 (+3.9 %), n=8 299.4 (+7.9 %) against the sweep.
⁶ bfloat16 SSM state with `--max-mamba-cache-size 128` caps `max_running_requests` at 25 (same cache memory as float32 x 64); only n=32 queues.

Nemotron single-node run (2026-10-11): `scripts/run_standalone.sh PRESET=nemotron3_super` (MEM_FRACTION=0.90, extra arg `--max-mamba-cache-size 64`), image 0.5.21-sm121, context 262144, KV fp8 (728k tokens). Levels n=1/4/8/16/32 with 4/16/32/64/128 questions, max_tokens 8192. GSM8K 4/4, 16/16, 32/32, 64/64, 126/128 (2 wrong answers, no truncations), no char runs, 0 eager decode batches, accept len 4.0-4.2 at every batch size (cluster: 4.0-4.1).

bfloat16 SSM rerun (same day, same preset plus `--mamba-ssm-dtype bfloat16 --max-mamba-cache-size 128`, now the preset defaults): KV 693k tokens, GSM8K 4/4, 15/16, 31/32, 64/64, 125/128 (+1 truncation at 8192 tokens); the n=4/n=8 miss is the usual id 12. Accept len 4.0-4.3 at every batch size (float32: 4.0-4.2), 0 eager decode batches, no char runs. bfloat16 does not hurt MTP; the gain is concurrency: +19 % at n=16, +23 % at n=32, n<=8 within noise.

## Observations

- Flash-Next 1 node vs 4 nodes (MTP): about x2.2 at n=1 (32 -> 69-75), about x2.2-2.5 at n=16 (188 -> 420-470; the single node only runs 9-11 concurrent).
- 27B single node vs Flash-Next single node: the dense 27B with DSpark is faster at every concurrency (n=1 48 vs 32, n=16 337 vs 188). Without speculation it is unusable (12.6 tok/s per stream).
- Nemotron Super 1 node vs 4 nodes (MTPv2 4/1/5): x2.6 at n=1 (32.1 -> 82.7), x2.8 at n=4 (73.6 -> 202.6), x2.6 at n=8 (107 -> 277). At n=1 the single node equals Flash-Next single node (32 vs 32) and is well below the dense 27B with DSpark (48).
- Nemotron Super 4 nodes leads at n=1 (82.7); at n=16 it is within noise of Flash-Next (377 vs 420-470). Only Nemotron has n=32 data (513).
- Run-to-run spread at n>=8 is large (Flash-Next same config up to x1.59 at n=16; Nemotron sweep vs A/B up to 20 %). Differences below about 20 % are not a ranking.

## Sources

- `TESTLOGS/standalone_nn1_tp1/qwen-3.8-27b-nvfp4-modelopt/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-27b-nvfp4-modelopt_1n.md`
- `TESTLOGS/standalone_nn1_tp1/qwen-3.8-flash-next-nvfp4/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-flash-next-nvfp4_1n.md`
- `TESTLOGS/sglang_nn4_tp4_ep4/qwen-3.8-flash-next-nvfp4/TESTLOG_nv580.178.04_sglang-0.5.21-sm121_qwen-3.8-flash-next-nvfp4_4n.md`
- `TESTLOGS/sglang_nn4_tp4_ep1/nemotron-3-super-120b-a12b-nvfp4/TESTLOG_sglang-0.5.21-sm121_nemotron-3-super-120b-a12b-nvfp4-mtpv2_4n.md`
