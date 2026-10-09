# SGLang Test Log: NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 + MTPv2 draft head, 4 Nodes, TP=4 EP=4, 0.5.21-sm121

> **STATUS: COMPLETE, executed 2026-10-08 (19:38 to 19:45 UTC).** GSM8K-chat concurrency sweep n=1..32,
> metrics only. Zero errors, zero truncations, zero restarts, `#queue-req` 0 throughout.
> Peak throughput 328 tok/s at n=32, 70 tok/s at n=1, accept len 3.1 to 3.6.

## Environment

| Component            | Value                                                                                                                                 |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| GPU                  | NVIDIA GB10 (SM121), 4 nodes (spark1 head, spark2-4 workers)                                                                          |
| Image                | `xomoxcc/dgx-spark-sglang:0.5.21-sm121`                                                                                               |
| Model                | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (`NemotronHForCausalLM`, Mamba2 + MoE + attn hybrid, reasoning model)                |
| Parallelism          | TP=4, EP=4                                                                                                                            |
| KV cache             | fp8, 20.9M tokens                                                                                                                     |
| Context              | 524288                                                                                                                                |
| mem_fraction_static  | 0.75                                                                                                                                  |
| max_running_requests | 32                                                                                                                                    |
| Mamba cache          | 160 slots                                                                                                                             |
| Speculative decoding | EAGLE, steps 3 / topk 1 / draft tokens 4                                                                                              |
| Draft model          | `nvidia/Nemotron-3-Super-120B-A12B-BF16-MTPv2` (separately distributed MTPv2 head, loaded as `NemotronHForCausalLMMTP`, 1.54 GB/rank) |
| Head pod             | `sglang-head-5fcd5798b7-ng5tm`, container `sglang`                                                                                    |
| Runtime patches      | run at start, not relevant to this log                                                                                                |

## Boot facts (supplied by operator)

- Target load 103 s, draft load 96 s.
- CUDA graph capture: verify 8.7 s, draft decode 5.4 s, draft extend 1.1 s.
- Fired up 19:12:23 UTC, about 5 min after container start.
- Head restart count 0 before and after the sweep; worker-1 shows 2 old restarts from before the head started, unchanged.

## Sweep design

- Harness: `sglang-gsm8k run`, chat API via port-forward `svc/sglang` on local 38081.
- Levels n = 1, 2, 4, 8, 16, 32, sequential, 30 s pause between levels, 2n questions per level (always the first 2n of the test split).
- `--max-tokens 8192 --temperature 0 --timeout 900 --dispatch-stop-frac 1.0`.
- Metrics only. The sample is far too small (2 to 64 questions, easy leading questions) for accuracy to mean anything.
- Scheduler stats parsed from the head log "Decode batch" lines (logged every ~192 steps, so sparse) within each level's window, taking lines where `#running-req` equals the maximum seen.
- Raw data: `sweep_n<N>.jsonl`, `times.txt`, `head.log` in the session scratchpad `gsm8k_sweep_nemotron/`.

## Results

| n  | questions | wall s | agg tok/s | peak tok/s | mean per-req tok/s | sched median tok/s @n (samples) | accept len median | lat p50 s | lat max s | trunc | correct/scored |
|----|-----------|--------|-----------|------------|--------------------|---------------------------------|-------------------|-----------|-----------|-------|----------------|
| 1  | 2         | 6.6    | 68.9      | 70.3       | 66.5               | 71.5 (2)                        | 3.20              | 5.4       | 5.4       | 0     | 2/2            |
| 2  | 4         | 13.6   | 108.2     | 123.1      | 55.2               | 118.4 (5)                       | 3.35              | 7.6       | 10.5      | 0     | 4/4            |
| 4  | 8         | 26.2   | 134.8     | 201.2      | 45.8               | 173.6 (5)                       | 3.49              | 9.7       | 16.3      | 0     | 8/8            |
| 8  | 16        | 32.4   | 197.5     | 267.3      | 28.8               | 188.4 (3)                       | 3.47              | 13.2      | 26.4      | 0     | 16/16          |
| 16 | 32        | 56.0   | 221.5     | 306.0      | 16.3               | 212.2 (4)                       | 3.39              | 22.5      | 55.0      | 0     | 32/32          |
| 32 | 64        | 101.9  | 245.5     | 328.1      | 9.2                | 150.0 (2)                       | 3.55              | 41.1      | 92.7      | 0     | 64/64          |

Notes on the columns: peak = concurrent peak tok/s from the harness (sum of per-request rates while overlapping); mean per-req = completion tokens / latency averaged over requests; scheduler samples are "Decode batch" lines with `#running-req` equal to n (gen throughput 0.1 first-line warmup values excluded).
Max scheduler gen throughput seen at running == n: 73.2 (n=1), 122.0 (2), 186.6 (4), 226.5 (8), 258.7 (16), 207.0 (32).

## Scheduler and accept length

- `#queue-req` stayed 0 in every Decode batch line.
- Accept len median 3.2 to 3.55 across levels (max possible with 4 draft tokens is 4), accept rate 0.71 to 0.86 at n=1.
- The n=32 scheduler median (150, 2 samples) is low-sample and sits below the n=16 value; the harness peak (328) is the better figure at that level, since n=32 spends much of its window in ramp-up and tail with fewer than 32 running.

## Anomalies

- None: no 5xx, no tracebacks, no "Scheduler hit", no `#queue-req` > 0, no NaN. Zero runs of 20+ identical characters. No truncations.
- The harness flagged "odd_nonascii" on some responses from n=2 upward; these are ordinary typographic characters in the math prose (e.g. curly apostrophes), no garbage was seen.
- The first Decode-batch line of a request burst shows gen throughput 0.1 (timer artifact, excluded).

## Comparison

- Against the GLM-5.3-Flash sweep from earlier today (peak n1 50, n2 71, n4 122, n8 133, n16 172, n32 210; accept len ~4.0): Nemotron peaks are higher at every level (70, 123, 201, 267, 306, 328, i.e. 1.4x at n=1 up to 2.0x at n=8) with a lower accept len (3.2 to 3.55 vs ~4.0).
- Against the profile's built-in-MTP-head reference (n=1 54 tok/s, n=8 peak 199.7, accept ~2.7, measured 2026-06-16 on an older image): MTPv2 head on 0.5.21 gives n=1 about 70 tok/s (+30%), n=8 peak 267 (+34%), accept len about 3.4 vs 2.7. Different image, EP and question set, so treat as indicative.

---

# Run 2: built-in MTP head, steps 3 / draft 4

Executed 2026-10-08, sweep 20:08 to 20:15 UTC (CEST 22:08 to 22:15), head pod `sglang-head-f778ddd7b-jht22`, same image `0.5.21-sm121`, same sweep design as above (port-forward on local 38082).

## Config delta vs the MTPv2 reference

- `speculative_draft_model_path: ""` and `speculative_draft_model_quantization` removed. Launch line carries `--speculative-algo EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` and no `--speculative-draft-model-path`.

## Boot facts

- Target load 126.5 s (`NemotronHForCausalLM`, mem usage 20.99 GB); built-in head load 20.2 s (`NemotronHForCausalLMMTP`, 1.77 GB/rank, `quant=modelopt_mixed`, taken from the target checkpoint).
- Mamba cache 160 slots: ssm_state 6.29 GB, intermediate_ssm_state 5.16 GB, intermediate_conv_window 0.04 GB.
- Memory pool end avail mem 30.24 / 25.07 GB; graph capture: verify 8.7 s (1.10 GB), draft decode 5.6 s (0.52 GB), draft extend 1.1 s (0.59 GB); avail mem after capture 16.27 GB.
- Fired up 20:07:45, about 4 min after container start. No tracebacks.

## Results

| n  | questions | wall s | agg tok/s | peak tok/s | mean per-req tok/s | sched median tok/s @n (samples) | accept len median | lat p50 s | lat max s | trunc | correct/scored |
|----|-----------|--------|-----------|------------|--------------------|---------------------------------|-------------------|-----------|-----------|-------|----------------|
| 1  | 2         | 9.3    | 64.4      | 67.0       | 65.1               | 64.8 (4)                        | 3.22              | 6.1       | 6.1       | 0     | 2/2            |
| 2  | 4         | 18.3   | 95.6      | 123.6      | 53.9               | 101.5 (5)                       | 3.24              | 8.1       | 13.2      | 0     | 4/4            |
| 4  | 8         | 25.4   | 126.4     | 174.3      | 40.7               | 154.2 (3)                       | 3.41              | 9.8       | 20.1      | 0     | 8/8            |
| 8  | 16        | 34.7   | 189.2     | 236.8      | 25.9               | 192.1 (5)                       | 3.31              | 15.4      | 29.8      | 0     | 15/16          |
| 16 | 32        | 49.9   | 227.7     | 287.4      | 16.7               | 312.6 (3)                       | 3.36              | 18.5      | 49.4      | 0     | 31/32          |
| 32 | 64        | 101.6  | 235.3     | 301.3      | 8.9                | 80.7 (1)                        | 3.40              | 38.5      | 90.7      | 0     | 63/64          |

Sweep window start/end (UTC) per level is in the scratchpad `gsm8k_sweep_nemotron_A/times.txt`.

## Scheduler stats

- `#queue-req` 0 in every Decode batch line; accept len median 3.2 to 3.4 (min seen 2.83 at n=1 warmup), versus 3.2 to 3.55 for MTPv2 3/4.
- Scheduler sample counts are small (1 to 5 per level), the n=32 scheduler median (1 sample, maxrun 31) is not usable; use the harness peak.

## Anomalies

- 3 wrong answers across 126 (question id 12 at n=8 and n=16, id 45 at n=32), all `finish=stop` with a plain arithmetic slip, no truncation, no garbage characters. The MTPv2 3/4 reference had 126/126. Greedy decoding with speculation is not batch-invariant, so single-question flips are not meaningful at this sample size, but note A is the only run with more than one miss.
- No 5xx, no tracebacks, no restarts.

---

# Run 3: MTPv2, steps 5 / draft 7

Executed 2026-10-08, sweep 20:21 to 20:27 UTC (CEST 22:21 to 22:27), head pod `sglang-head-6b66fff948-5szxv`, same image, same sweep design.

## Config delta vs the MTPv2 reference

- `speculative_num_steps: 5`, `speculative_num_draft_tokens: 7` (topk 1, MTPv2 draft with `unquant`). Launch line confirmed: `--speculative-num-steps 5 --speculative-num-draft-tokens 7 --speculative-draft-model-path nvidia/Nemotron-3-Super-120B-A12B-BF16-MTPv2 --speculative-draft-model-quantization unquant`.

## Boot facts

- Target load 127.3 s (20.99 -> 21.04 GB), MTPv2 draft load 10.8 s (1.36 GB/rank, `NemotronHForCausalLMMTP`).
- Mamba cache 160 slots: ssm_state 6.29 GB, intermediate_ssm_state 7.73 GB (up from 5.16 GB), intermediate_conv_window 0.05 GB.
- Memory pool end avail mem 30.05 / 25.26 GB; graph capture: verify 9.6 s (1.04 GB), draft decode 6.5 s (1.19 GB), draft extend 1.4 s (0.84 GB); avail mem after capture 15.12 GB (vs 16.27 GB for 3/4).
- Fired up 20:20:38, about 4.5 min after container start. No tracebacks.

## Results

| n  | questions | wall s | agg tok/s | peak tok/s | mean per-req tok/s | sched median tok/s @n (samples) | accept len median | lat p50 s | lat max s | trunc | correct/scored |
|----|-----------|--------|-----------|------------|--------------------|---------------------------------|-------------------|-----------|-----------|-------|----------------|
| 1  | 2         | 7.4    | 73.4      | 74.1       | 72.6               | 77.4 (3)                        | 4.40              | 5.6       | 5.6       | 0     | 2/2            |
| 2  | 4         | 16.3   | 104.3     | 133.3      | 59.0               | 122.6 (3)                       | 4.48              | 6.2       | 11.9      | 0     | 4/4            |
| 4  | 8         | 24.0   | 138.4     | 200.2      | 46.2               | 173.3 (3)                       | 4.43              | 9.0       | 15.8      | 0     | 8/8            |
| 8  | 16        | 33.9   | 194.4     | 257.3      | 28.0               | 162.1 (3)                       | 4.45              | 15.5      | 29.8      | 0     | 16/16          |
| 16 | 32        | 50.6   | 243.4     | 342.6      | 18.0               | 177.2 (2)                       | 4.56              | 20.6      | 48.4      | 0     | 32/32          |
| 32 | 64        | 75.4   | 325.8     | 431.8      | 12.1               | 59.4 (1)                        | 4.93              | 30.3      | 64.7      | 0     | 63/64          |

## Scheduler stats

- `#queue-req` 0 throughout; accept len median 4.4 to 4.9 (max possible 7), rising with batch size.
- Scheduler gen-throughput samples are sparse and often catch ramp lines (e.g. 56.7 at n=16, 59.4 at n=32); the harness peak is the reliable figure.

## Anomalies

- One wrong answer (id 12 at n=32, plain slip, `finish=stop`); no truncations, no tracebacks, no restarts.
- n=8 peak (257.3) is below the 3/4 reference (267.3) despite +1.0 accept len: at that batch size the extra verify width costs more than the accepted tokens return.

---

# Three-way comparison (same image 0.5.21-sm121, TP4/EP4)

| n  | peak tok/s ref (MTPv2 3/4) | peak A (built-in 3/4) | peak B (MTPv2 5/7) | agg ref | agg A | agg B | accept ref | accept A | accept B |
|----|----------------------------|-----------------------|--------------------|---------|-------|-------|------------|----------|----------|
| 1  | 70.3                       | 67.0                  | 74.1               | 68.9    | 64.4  | 73.4  | 3.20       | 3.22     | 4.40     |
| 2  | 123.1                      | 123.6                 | 133.3              | 108.2   | 95.6  | 104.3 | 3.35       | 3.24     | 4.48     |
| 4  | 201.2                      | 174.3                 | 200.2              | 134.8   | 126.4 | 138.4 | 3.49       | 3.41     | 4.43     |
| 8  | 267.3                      | 236.8                 | 257.3              | 197.5   | 189.2 | 194.4 | 3.47       | 3.31     | 4.45     |
| 16 | 306.0                      | 287.4                 | 342.6              | 221.5   | 227.7 | 243.4 | 3.39       | 3.36     | 4.56     |
| 32 | 328.1                      | 301.3                 | 431.8              | 245.5   | 235.3 | 325.8 | 3.55       | 3.40     | 4.93     |

## Conclusion

Image effect: A (built-in head, 3/4, image 0.5.21) against the June built-in-head reference on 0.5.13 (n1 54, n8 peak 199.7, accept 2.7) gives n1 67 (+24%), n8 236.8 (+19%) and accept 3.2 to 3.4 (vs 2.7), so most of the gain over June comes from the image (and EP4/question set), not from the MTPv2 head. Draft-head effect: MTPv2 3/4 against A at identical steps is +5% at n=1, +15% at n=4, +13% at n=8, +6% at n=16, +9% at n=32 on peak tok/s, with accept len up only about 0.1 to 0.15 (3.2 to 3.55 vs 3.2 to 3.4), a modest but consistent gain, partly within run-to-run noise at n<=2 (peaks tie at n=2). Draft-depth effect: MTPv2 5/7 against MTPv2 3/4 raises accept len by about 1.0 (4.4 to 4.9) and wins at n=1 (+5%), n=2 (+8%), n=16 (+12%) and n=32 (+32%, agg +33%), ties at n=4 (200 vs 201) and loses at n=8 (257 vs 267, -4%), so deeper drafting pays off where the batch is large or tiny and costs at mid batch; sample counts are small (one sweep per config, 2n questions), so the n=8 deficit is within plausible noise but the n=16/n=32 wins are large. Memory cost of 5/7 is about 1.1 GB/rank less free memory after graph capture (15.1 vs 16.3 GB). The profile therefore stays on MTPv2 3/4 under the n=8 AND n=16 criterion; 5/7 is worth a repeat with more questions if the target load is mostly n>=16.

---

# Run 4: steps x topk x draft sweep

Executed 2026-10-08 21:16 to 22:38 UTC (CEST 23:16 to 00:38), image `xomoxcc/dgx-spark-sglang:0.5.21-sm121`, TP4/EP4, MTPv2 draft head with `unquant`, one Ansible rollout per case. Bench: 4 / 32 / 64 / 128 questions at n = 1 / 8 / 16 / 32 (first N of the full GSM8K test split), `--max-tokens 8192 --temperature 0`, 20 s between levels. Raw data: scratchpad `gsm8k_mtpsweep/caseNN/` (per-level JSONL, `head.log`, `times.txt`, `status.txt`), Ansible logs `ansible_mtpsweep_caseNN.log`.

## Constraint findings (image 0.5.21-sm121, `/usr/local/lib/python3.12/dist-packages/sglang/srt`)

- `arg_groups/speculative_hook.py:1144-1153`: with `speculative_eagle_topk == 1` and `speculative_num_draft_tokens != speculative_num_steps + 1`, SGLang logs a warning and rewrites draft tokens to `steps + 1`. A chain sweep therefore has exactly one legal draft count per step count.
- Consequence for Run 3: the "5/1/7" point of Run 3 ran as 5/1/6 (boot-log `server_args` of this sweep show the rewrite is applied; Run 3 had draft extend graphs with `num_tokens_per_req=6` and the 7.73 GB intermediate_ssm cache that case 03 reproduces at draft 6). Run 3 results are the 5/1/6 point, and the planned case 06 (5/1/7) is identical to case 03 and was skipped.
- `arg_groups/speculative_hook.py:948-951` (MTP-family branch, not this EAGLE path): "MTP requires --speculative-eagle-topk 1" and verification width must equal steps + 1; it does not apply to `speculative_algo: EAGLE`, which is what the profile uses.
- `speculative/eagle_worker_v2.py:177-179`: topk must be 1 only when `speculative_use_rejection_sampling` is on (chain sampler). Otherwise the v2 worker has a tree path (`fast_topk`, line 824).
- `layers/attention/hybrid_linear_attn_backend.py:75,271,700,886`: the hybrid linear-attention (mamba) backend has explicit `topk > 1` tree-verify handling, so no NemotronH restriction on topk > 1 exists.
- `arg_groups/mamba_hook.py:123`: `mamba_track_interval (256) >= speculative_num_draft_tokens`, not binding for any case here.
- `arg_groups/speculative_hook.py:1156-1170`: topk > 1 with page_size > 1 needs flashinfer, fa3 or triton; page_size is 1 and the backend is flashinfer, so topk 2 is legal.

## Cases

Peak = concurrent peak tok/s from `analyze`, agg = aggregate tok/s. Accept = median accept len at `#running-req == n`. Free mem = avail mem after the last CUDA graph capture (draft extend), ssm cache = `intermediate_ssm_state_cache` from the "Mamba Cache is allocated" line.

| case | steps/topk/draft | peak n1 | peak n8 | peak n16 | peak n32 | agg n8 | agg n16 | agg n32 | accept n8 | accept n32 | free mem GB | ssm cache GB | correct/scored | status                                          |
|------|------------------|---------|---------|----------|----------|--------|---------|---------|-----------|------------|-------------|--------------|----------------|-------------------------------------------------|
| 01   | 3/1/4            | 77.9    | 246.3   | 364.8    | 450.0    | 193.5  | 204.3   | 291.2   | 3.48      | 3.50       | 16.15       | 5.16         | 225/228        | ok                                              |
| 02   | 4/1/5            | 82.7    | 277.4   | 377.2    | 513.2    | 212.8  | 266.1   | 291.7   | 4.02      | 4.11       | 15.50       | 6.45         | 224/228        | ok                                              |
| 03   | 5/1/6            | 85.2    | 274.5   | 328.5    | 402.2    | 206.3  | 252.8   | 212.5   | 4.54      | 4.57       | 14.96       | 7.73         | 224/228        | ok                                              |
| 04   | 6/1/7            | 79.6    | 235.4   | 319.4    | 454.5    | 199.0  | 234.5   | 243.8   | 4.79      | 4.94       | 14.50       | 9.02         | 224/228        | ok                                              |
| 05   | 7/1/8            | -       | -       | -        | -        | -      | -       | -       | -         | -          | -           | -            | -              | skipped (stop rule)                             |
| 06   | 5/1/7            | -       | -       | -        | -        | -      | -       | -       | -         | -          | -           | -            | -              | skipped (identical to 03, rewritten to draft 6) |
| 07   | 3/2/6            | 70.7    | 199.3   | 276.6    | 424.7    | 171.9  | 210.9   | 205.5   | 3.71      | 3.72       | 14.79       | 7.73         | 222/228        | ok                                              |
| 08   | 5/2/8            | 82.8    | 264.3   | 325.5    | 414.2    | 194.8  | 229.4   | 260.8   | 4.88      | 4.88       | 13.19       | 10.31        | 222/228        | ok                                              |

Latency (p50 / max seconds) and mean per-request tok/s at n = 8 / 16 / 32:

| case | n8 p50/max  | n16 p50/max | n32 p50/max  | mean per-req tok/s n8 / n16 / n32 |
|------|-------------|-------------|--------------|-----------------------------------|
| 01   | 12.0 / 25.6 | 21.1 / 78.5 | 31.7 / 92.9  | 27.1 / 18.3 / 11.5                |
| 02   | 11.3 / 28.8 | 18.6 / 41.3 | 29.7 / 80.1  | 29.2 / 19.2 / 12.6                |
| 03   | 11.2 / 27.8 | 20.9 / 53.8 | 37.5 / 179.1 | 28.2 / 17.8 / 10.3                |
| 04   | 12.7 / 33.8 | 21.4 / 81.8 | 31.7 / 104.6 | 26.8 / 17.4 / 11.2                |
| 07   | 13.8 / 44.1 | 25.3 / 59.3 | 36.4 / 141.9 | 23.0 / 14.8 / 10.2                |
| 08   | 13.2 / 36.5 | 23.2 / 57.9 | 37.3 / 96.4  | 26.7 / 16.5 / 10.2                |

Case 01 peak n1 / n2 are 4-question samples, so the n1 column is within about 10 % noise.

## Scheduler stats

- `#queue-req` was 0 at every level of every case. All cases reached "fired up" 4.0 to 5.2 min after the rollout (inits fast, caches warm).
- Scheduler gen-throughput medians at `#running-req == n` (max in parentheses), n8 / n16 / n32: 01 210.4 (257.4) / 309.5 (330.7) / 325.6 (394.3); 02 211.4 (253.4) / 268.3 (291.7) / 374.5 (428.9); 03 210.0 (268.9) / 288.9 (319.0) / 345.7 (355.4); 04 216.5 (247.0) / 246.8 (306.5) / 321.5 (354.0); 07 183.1 (227.8) / 214.9 (268.8) / 298.2 (361.2); 08 199.2 (231.6) / 251.6 (303.6) / 271.5 (372.6). Sample counts per level are only 4 to 10 (log interval 40 decode batches), so these are indicative; the harness peak is the primary figure.
- Accept len rises about 0.5 per extra step (3.5, 4.0, 4.5, 4.8 to 4.9 for 3, 4, 5, 6 steps at topk 1), but the verify width grows with it, so throughput peaks at 4/1/5.
- topk 2 buys accept len (3/2/6: 3.7 vs 3.5 for 3/1/4; 5/2/8: 4.9 vs 4.5 for 5/1/6) but the wider tree costs more than it returns: 3/2/6 is 19 % slower than 3/1/4 at n8 and 24 % slower at n16; 5/2/8 is 4 % slower than 5/1/6 at n8, about equal at n16.

## Anomalies

- Case 03 n=32: one request (id 119) ran away to the 8192-token limit (`finish=length`, empty content, 28.7k chars of reasoning, 159 s) and stretched the level to 179 s max latency; this is why 03's n32 agg (212.5) and peak (402.2) are low. Id 119 is also wrong in every other case (stop, wrong answer), so it is a hard question, not a speculation fault. It is the only truncation in 1368 requests.
- Wrong answers are plain misreads, all `finish=stop`: ids 100 and 119 (n32) are wrong in all cases, id 12 and 93 flip between cases (greedy decoding with speculation is not batch-invariant). Correct counts are 222 to 225 of 228 everywhere, so the quality gate is met: no char runs >= 20, `reasoning_content` populated for all 1368 requests, no 5xx.
- Peak n16 at 3/1/4 is 364.8 here versus 306.0 in the 2n-question reference; peak figures depend strongly on sample size and which requests overlap, so compare only within this run.
- No crashes, no tracebacks, no pod restarts beyond the rollout-driven ones.

## Conclusion

Winner for n <= 8 and for n >= 16 is the same: MTPv2 4/1/5 (steps 4, topk 1, draft 5). It has the best peak at n8 (277.4, +13 % over 3/1/4, +1 % over 5/1/6), n16 (377.2, +3 % over 3/1/4, +15 % over 5/1/6) and n32 (513.2, +14 % over 3/1/4), the best agg at n8 and n16 and the best latency p50 at n16 and n32 (n8 p50 11.3 vs 11.2 for 5/1/6), with 15.5 GB free after capture. 5/1/6 wins only n1 (85.2 vs 82.7, within noise) and 6/1/7 regresses at n8 and n16 versus its predecessor, so the stop rule ended the topk-1 chain after case 04 (case 05 skipped). topk 2 does not pay off on this hybrid model. The profile is set to 4/1/5.

## Run 5: max_total_tokens cap + Omni companion co-located (2026-10-09)

Head `sglang-head-6497fb4f88-p5ccj`, MTPv2 4/1/5, launch now with `--max-total-tokens 16777216`, Omni vision instance booted at the same time. Details in the Omni testlog section "Co-located vision companion next to Super (2026-10-09)".

- Load weight end 31.47 GB (122.8 s), draft 2.15 GB; mamba 160 slots (12.9 GB incl. intermediates); KV fp8 10182073 tokens (K 9.71 + V 9.71 GB, draft 1.21 + 1.21 GB), i.e. the auto-fit bound below the 16.78M cap; Memory pool end avail 23.14 GB; graphs 8.9 + 5.7 + 1.0 s; final available_gpu_mem 14.61 GB. GPU process memory 61.6 GB per rank.
- GSM8K n=8, 32 questions: alone 32/32, peak 289.4 tok/s (reference 277.4), agg 219.8, p50 11.0 s. In parallel with Omni n=8: 31/32, peak 293.8, agg 130.8, p50 16.0 s (Omni phase overlapped only partly).
- 0 restarts, no Scheduler/OOM errors.
