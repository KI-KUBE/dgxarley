# SGLang Test Log: NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 + MTPv2 draft head, 4 Nodes, TP=4 EP=4, 0.5.21-sm121

> **STATUS: COMPLETE, executed 2026-10-08 (19:38 to 19:45 UTC).** GSM8K-chat concurrency sweep n=1..32,
> metrics only. Zero errors, zero truncations, zero restarts, `#queue-req` 0 throughout.
> 70 tok/s at n=1, accept len 3.1 to 3.6. Throughput at n>1: Run 8.

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

| n  | questions | wall s | agg tok/s | n1 tok/s   | mean per-req tok/s | sched median tok/s @n (samples) | accept len median | lat p50 s | lat max s | trunc | correct/scored |
|----|-----------|--------|-----------|------------|--------------------|---------------------------------|-------------------|-----------|-----------|-------|----------------|
| 1  | 2         | 6.6    | 68.9      | 70.3       | 66.5               | 71.5 (2)                        | 3.20              | 5.4       | 5.4       | 0     | 2/2            |
| 2  | 4         | 13.6   | 108.2     | -          | 55.2               | 118.4 (5)                       | 3.35              | 7.6       | 10.5      | 0     | 4/4            |
| 4  | 8         | 26.2   | 134.8     | -          | 45.8               | 173.6 (5)                       | 3.49              | 9.7       | 16.3      | 0     | 8/8            |
| 8  | 16        | 32.4   | 197.5     | -          | 28.8               | 188.4 (3)                       | 3.47              | 13.2      | 26.4      | 0     | 16/16          |
| 16 | 32        | 56.0   | 221.5     | -          | 16.3               | 212.2 (4)                       | 3.39              | 22.5      | 55.0      | 0     | 32/32          |
| 32 | 64        | 101.9  | 245.5     | -          | 9.2                | 150.0 (2)                       | 3.55              | 41.1      | 92.7      | 0     | 64/64          |

Notes on the columns: n1 tok/s = single-stream tok/s (n>1 peak values removed, see Run 8); mean per-req = completion tokens / latency averaged over requests; scheduler samples are "Decode batch" lines with `#running-req` equal to n (gen throughput 0.1 first-line warmup values excluded).
Max scheduler gen throughput seen at running == n: 73.2 (n=1), 122.0 (2), 186.6 (4), 226.5 (8), 258.7 (16), 207.0 (32).

## Scheduler and accept length

- `#queue-req` stayed 0 in every Decode batch line.
- Accept len median 3.2 to 3.55 across levels (max possible with 4 draft tokens is 4), accept rate 0.71 to 0.86 at n=1.
- The n=32 scheduler median (150, 2 samples) is low-sample and sits below the n=16 value; throughput at n>1: Run 8.

## Anomalies

- None: no 5xx, no tracebacks, no "Scheduler hit", no `#queue-req` > 0, no NaN. Zero runs of 20+ identical characters. No truncations.
- The harness flagged "odd_nonascii" on some responses from n=2 upward; these are ordinary typographic characters in the math prose (e.g. curly apostrophes), no garbage was seen.
- The first Decode-batch line of a request burst shows gen throughput 0.1 (timer artifact, excluded).

## Comparison

- Against the GLM-5.3-Flash sweep from earlier today (n1 50 tok/s, accept len ~4.0): Nemotron n1 70 tok/s (1.4x) with a lower accept len (3.2 to 3.55 vs ~4.0).
- Against the profile's built-in-MTP-head reference (n=1 54 tok/s, accept ~2.7, measured 2026-06-16 on an older image): MTPv2 head on 0.5.21 gives n=1 about 70 tok/s (+30%), accept len about 3.4 vs 2.7. Different image, EP and question set, so treat as indicative.

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

| n  | questions | wall s | agg tok/s | n1 tok/s   | mean per-req tok/s | sched median tok/s @n (samples) | accept len median | lat p50 s | lat max s | trunc | correct/scored |
|----|-----------|--------|-----------|------------|--------------------|---------------------------------|-------------------|-----------|-----------|-------|----------------|
| 1  | 2         | 9.3    | 64.4      | 67.0       | 65.1               | 64.8 (4)                        | 3.22              | 6.1       | 6.1       | 0     | 2/2            |
| 2  | 4         | 18.3   | 95.6      | -          | 53.9               | 101.5 (5)                       | 3.24              | 8.1       | 13.2      | 0     | 4/4            |
| 4  | 8         | 25.4   | 126.4     | -          | 40.7               | 154.2 (3)                       | 3.41              | 9.8       | 20.1      | 0     | 8/8            |
| 8  | 16        | 34.7   | 189.2     | -          | 25.9               | 192.1 (5)                       | 3.31              | 15.4      | 29.8      | 0     | 15/16          |
| 16 | 32        | 49.9   | 227.7     | -          | 16.7               | 312.6 (3)                       | 3.36              | 18.5      | 49.4      | 0     | 31/32          |
| 32 | 64        | 101.6  | 235.3     | -          | 8.9                | 80.7 (1)                        | 3.40              | 38.5      | 90.7      | 0     | 63/64          |

Sweep window start/end (UTC) per level is in the scratchpad `gsm8k_sweep_nemotron_A/times.txt`.

## Scheduler stats

- `#queue-req` 0 in every Decode batch line; accept len median 3.2 to 3.4 (min seen 2.83 at n=1 warmup), versus 3.2 to 3.55 for MTPv2 3/4.
- Scheduler sample counts are small (1 to 5 per level), the n=32 scheduler median (1 sample, maxrun 31) is not usable.

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

| n  | questions | wall s | agg tok/s | n1 tok/s   | mean per-req tok/s | sched median tok/s @n (samples) | accept len median | lat p50 s | lat max s | trunc | correct/scored |
|----|-----------|--------|-----------|------------|--------------------|---------------------------------|-------------------|-----------|-----------|-------|----------------|
| 1  | 2         | 7.4    | 73.4      | 74.1       | 72.6               | 77.4 (3)                        | 4.40              | 5.6       | 5.6       | 0     | 2/2            |
| 2  | 4         | 16.3   | 104.3     | -          | 59.0               | 122.6 (3)                       | 4.48              | 6.2       | 11.9      | 0     | 4/4            |
| 4  | 8         | 24.0   | 138.4     | -          | 46.2               | 173.3 (3)                       | 4.43              | 9.0       | 15.8      | 0     | 8/8            |
| 8  | 16        | 33.9   | 194.4     | -          | 28.0               | 162.1 (3)                       | 4.45              | 15.5      | 29.8      | 0     | 16/16          |
| 16 | 32        | 50.6   | 243.4     | -          | 18.0               | 177.2 (2)                       | 4.56              | 20.6      | 48.4      | 0     | 32/32          |
| 32 | 64        | 75.4   | 325.8     | -          | 12.1               | 59.4 (1)                        | 4.93              | 30.3      | 64.7      | 0     | 63/64          |

## Scheduler stats

- `#queue-req` 0 throughout; accept len median 4.4 to 4.9 (max possible 7), rising with batch size.
- Scheduler gen-throughput samples are sparse and often catch ramp lines (e.g. 56.7 at n=16, 59.4 at n=32).

## Anomalies

- One wrong answer (id 12 at n=32, plain slip, `finish=stop`); no truncations, no tracebacks, no restarts.

---

# Three-way comparison (same image 0.5.21-sm121, TP4/EP4)

| n  | agg ref | agg A | agg B | accept ref | accept A | accept B |
|----|---------|-------|-------|------------|----------|----------|
| 1  | 68.9    | 64.4  | 73.4  | 3.20       | 3.22     | 4.40     |
| 2  | 108.2   | 95.6  | 104.3 | 3.35       | 3.24     | 4.48     |
| 4  | 134.8   | 126.4 | 138.4 | 3.49       | 3.41     | 4.43     |
| 8  | 197.5   | 189.2 | 194.4 | 3.47       | 3.31     | 4.45     |
| 16 | 221.5   | 227.7 | 243.4 | 3.39       | 3.36     | 4.56     |
| 32 | 245.5   | 235.3 | 325.8 | 3.55       | 3.40     | 4.93     |

n=1 single-stream tok/s: ref 70.3, A 67.0, B 74.1.

## Conclusion

Image effect: A (built-in head, 3/4, image 0.5.21) against the June built-in-head reference on 0.5.13 (n1 54, accept 2.7) gives n1 67 (+24%) and accept 3.2 to 3.4 (vs 2.7), so most of the gain over June comes from the image (and EP4/question set), not from the MTPv2 head. Draft-head effect: MTPv2 3/4 against A at identical steps is +5% at n=1 with accept len up only about 0.1 to 0.15 (3.2 to 3.55 vs 3.2 to 3.4). Draft-depth effect: MTPv2 5/7 (ran as 5/1/6) against MTPv2 3/4 raises accept len by about 1.0 (4.4 to 4.9) and n=1 by 5%. Memory cost of 5/7 is about 1.1 GB/rank less free memory after graph capture (15.1 vs 16.3 GB). The n>1 comparison of these configs is in Run 4 (scheduler) and Run 8 (full load).

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

peak n1 = single-stream tok/s, agg = aggregate tok/s (total tokens / wall time). Accept = median accept len at `#running-req == n`. Free mem = avail mem after the last CUDA graph capture (draft extend), ssm cache = `intermediate_ssm_state_cache` from the "Mamba Cache is allocated" line.

| case | steps/topk/draft | peak n1 | agg n8 | agg n16 | agg n32 | accept n8 | accept n32 | free mem GB | ssm cache GB | correct/scored | status                                          |
|------|------------------|---------|--------|---------|---------|-----------|------------|-------------|--------------|----------------|-------------------------------------------------|
| 01   | 3/1/4            | 77.9    | 193.5  | 204.3   | 291.2   | 3.48      | 3.50       | 16.15       | 5.16         | 225/228        | ok                                              |
| 02   | 4/1/5            | 82.7    | 212.8  | 266.1   | 291.7   | 4.02      | 4.11       | 15.50       | 6.45         | 224/228        | ok                                              |
| 03   | 5/1/6            | 85.2    | 206.3  | 252.8   | 212.5   | 4.54      | 4.57       | 14.96       | 7.73         | 224/228        | ok                                              |
| 04   | 6/1/7            | 79.6    | 199.0  | 234.5   | 243.8   | 4.79      | 4.94       | 14.50       | 9.02         | 224/228        | ok                                              |
| 05   | 7/1/8            | -       | -      | -       | -       | -         | -          | -           | -            | -              | skipped (stop rule)                             |
| 06   | 5/1/7            | -       | -      | -       | -       | -         | -          | -           | -            | -              | skipped (identical to 03, rewritten to draft 6) |
| 07   | 3/2/6            | 70.7    | 171.9  | 210.9   | 205.5   | 3.71      | 3.72       | 14.79       | 7.73         | 222/228        | ok                                              |
| 08   | 5/2/8            | 82.8    | 194.8  | 229.4   | 260.8   | 4.88      | 4.88       | 13.19       | 10.31        | 222/228        | ok                                              |

Latency (p50 / max seconds) and mean per-request tok/s at n = 8 / 16 / 32:

| case | n8 p50/max  | n16 p50/max | n32 p50/max  | mean per-req tok/s n8 / n16 / n32 |
|------|-------------|-------------|--------------|-----------------------------------|
| 01   | 12.0 / 25.6 | 21.1 / 78.5 | 31.7 / 92.9  | 27.1 / 18.3 / 11.5                |
| 02   | 11.3 / 28.8 | 18.6 / 41.3 | 29.7 / 80.1  | 29.2 / 19.2 / 12.6                |
| 03   | 11.2 / 27.8 | 20.9 / 53.8 | 37.5 / 179.1 | 28.2 / 17.8 / 10.3                |
| 04   | 12.7 / 33.8 | 21.4 / 81.8 | 31.7 / 104.6 | 26.8 / 17.4 / 11.2                |
| 07   | 13.8 / 44.1 | 25.3 / 59.3 | 36.4 / 141.9 | 23.0 / 14.8 / 10.2                |
| 08   | 13.2 / 36.5 | 23.2 / 57.9 | 37.3 / 96.4  | 26.7 / 16.5 / 10.2                |

Case 01 peak n1 is a 4-question sample, so the n1 column is within about 10 % noise.

## Scheduler stats

- `#queue-req` was 0 at every level of every case. All cases reached "fired up" 4.0 to 5.2 min after the rollout (inits fast, caches warm).
- Scheduler gen-throughput medians at `#running-req == n` (max in parentheses), n8 / n16 / n32: 01 210.4 / 309.5 / 325.6; 02 211.4 / 268.3 / 374.5; 03 210.0 / 288.9 / 345.7; 04 216.5 / 246.8 / 321.5; 07 183.1 / 214.9 / 298.2; 08 199.2 / 251.6 / 271.5. Sample counts per level are only 4 to 10 (log interval 40 decode batches), so these are indicative.
- Accept len rises about 0.5 per extra step (3.5, 4.0, 4.5, 4.8 to 4.9 for 3, 4, 5, 6 steps at topk 1), but the verify width grows with it, so throughput peaks at 4/1/5.
- topk 2 buys accept len (3/2/6: 3.7 vs 3.5 for 3/1/4; 5/2/8: 4.9 vs 4.5 for 5/1/6) but the wider tree costs more than it returns: scheduler median 3/2/6 vs 3/1/4 is 13 % lower at n8 and 31 % lower at n16; 5/2/8 vs 5/1/6 5 % lower at n8, 13 % lower at n16.

## Anomalies

- Case 03 n=32: one request (id 119) ran away to the 8192-token limit (`finish=length`, empty content, 28.7k chars of reasoning, 159 s) and stretched the level to 179 s max latency; this is why 03's n32 agg (212.5) is low. Id 119 is also wrong in every other case (stop, wrong answer), so it is a hard question, not a speculation fault. It is the only truncation in 1368 requests.
- Wrong answers are plain misreads, all `finish=stop`: ids 100 and 119 (n32) are wrong in all cases, id 12 and 93 flip between cases (greedy decoding with speculation is not batch-invariant). Correct counts are 222 to 225 of 228 everywhere, so the quality gate is met: no char runs >= 20, `reasoning_content` populated for all 1368 requests, no 5xx.
- No crashes, no tracebacks, no pod restarts beyond the rollout-driven ones.

## Conclusion

MTPv2 4/1/5 (steps 4, topk 1, draft 5) is the pick: highest scheduler median at n32 (374.5 vs 325.6 for 3/1/4 and 345.7 for 5/1/6), at n8 all topk-1 cases are within 3 % (210-217), at n16 3/1/4 leads (309.5 vs 268.3), with 4 to 10 samples per level. Run 8 (engine max, 2026-10-12): at n32 4/1/5 bf16 485, 4/1/5 float32 432, 3/1/4 412, 5/1/6 404; at n<=16 the float32 configs are within about 10 % of each other. 4/1/5 also has the best agg at n8 and n16 and the best latency p50 at n16 and n32, with 15.5 GB free after capture. 5/1/6 wins only n1 (85.2 vs 82.7, within noise) and 6/1/7 has the lowest topk-1 scheduler median at n16 and n32, so the stop rule ended the topk-1 chain after case 04 (case 05 skipped). topk 2 does not pay off on this hybrid model. The profile is set to 4/1/5.

## Run 5: max_total_tokens cap + Omni companion co-located (2026-10-09)

Head `sglang-head-6497fb4f88-p5ccj`, MTPv2 4/1/5, launch now with `--max-total-tokens 16777216`, Omni vision instance booted at the same time. Details in the Omni testlog section "Co-located vision companion next to Super (2026-10-09)".

- Load weight end 31.47 GB (122.8 s), draft 2.15 GB; mamba 160 slots (12.9 GB incl. intermediates); KV fp8 10182073 tokens (K 9.71 + V 9.71 GB, draft 1.21 + 1.21 GB), i.e. the auto-fit bound below the 16.78M cap; Memory pool end avail 23.14 GB; graphs 8.9 + 5.7 + 1.0 s; final available_gpu_mem 14.61 GB. GPU process memory 61.6 GB per rank.
- GSM8K n=8, 32 questions: alone 32/32, agg 219.8 tok/s, p50 11.0 s. In parallel with Omni n=8: 31/32, agg 130.8, p50 16.0 s (Omni phase overlapped only partly).
- 0 restarts, no Scheduler/OOM errors.

### Run 5b: cap lowered to 9437184 (2026-10-09)

Head `sglang-head-869849cdd7-lfd9t`, launch `--max-total-tokens 9437184`; only the Super head and workers rolled, Omni pods untouched.

- Mamba 160 slots (conv 0.09 + ssm 6.29 + intermediate ssm 6.45 + intermediate conv 0.04 GB); KV fp8 9437184 tokens (K 9.00 + V 9.00 GB, draft 1.13 + 1.13 GB); `available_gpu_mem` 16.00 GB after capture (was 14.61).
- GPU process per rank: Super 59,959 MiB (was 61,992), Omni 10,907 MiB.
- Per node (MemAvailable / Shmem / AnonPages, GB): spark1 15.96 / 15.80 / 13.24; spark2 19.43 / 15.80 / 10.16; spark3 18.86 / 15.80 / 10.15; spark4 19.45 / 15.80 / 10.18. `free -g` available 15 / 18 / 17 / 18.
- The cap bound this time (9.44M is below the 10.18M auto-fit); smoke "17*23" returned 391 with reasoning_content, 0 restarts.

## Run 6: n=4 fill-in (2026-10-11)

Current profile (MTPv2 4/1/5, TP4/EP4, `--max-total-tokens` cap as in Run 5b), head `sglang-head-57cb9d8bb-d6t97`, redeployed with `ansible-playbook k8s_dgx.yml --tags sglang -e sglang_model=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`; ready 4.0 min after the rollout. Same GSM8K harness (temperature 0, max-tokens 8192), levels run back to back with a 20 s pause. n1 is the drift anchor against the 2026-10-08 sweep (case 02). n4/n8 throughput: Run 8.

| level | questions | n1 tok/s                   | agg tok/s | accept (median) | correct/scored | p50 / max s |
|-------|-----------|----------------------------|-----------|-----------------|----------------|-------------|
| n1    | 4         | 85.9 (10-08: 82.7, +3.9 %) | 81.6      | 4.10            | 4/4            | 4.7 / 5.6   |
| n4    | 16        | -                          | 166.1     | 4.10            | 15/16          | 8.9 / 21.7  |
| n8    | 32        | -                          | 241.7     | 4.13            | 31/32          | 10.6 / 30.5 |

- Accept len is the median over decode batches with `#running-req` equal to the level (8 to 10 samples per level, so indicative). 0 decode batches with `cuda graph: False` in any level window.
- The single wrong answer at n4 and n8 is id 12 (pred 12, gt 13, `finish=stop`), which also flips between cases in the 10-08 sweep. No char runs, no failures other than that, no truncation, no restarts.
- The n1 drift anchor sits 3.9 % above the 10-08 value, within the sample-size noise.
- Smoke "17*23" returned 391 with reasoning_content before the runs.

## Run 7: bfloat16 SSM state (2026-10-11)

Profile now sets `mamba_ssm_dtype: "bfloat16"` (cache 160 slots, MTPv2 4/1/5, TP4/EP4), head `sglang-head-7569c95448-5z92v`, ready 3.5 min after the rollout (12:30 to 12:35 UTC). Levels and question counts as in the Run 4 sweep plus n4 (4/16/32/64/128), 20 s pause, max-tokens 8192, temperature 0. Cluster idle before the deploy; the head log has exactly 260 POSTs (16 probe + 244 bench), so no foreign traffic.

| level | questions | n1 tok/s            | agg tok/s | correct/scored |
|-------|-----------|---------------------|-----------|----------------|
| n1    | 4         | 82.1 (float32 82.7) | 77.8      | 4/4            |
| n4    | 16        | -                   | 158.4     | 16/16          |
| n8    | 32        | -                   | 229.2     | 31/32          |
| n16   | 64        | -                   | 219.4     | 63/64          |
| n32   | 128       | -                   | 340.0     | 124/128        |

- Boot: `mamba_ssm_dtype` bfloat16, `max_running_requests` 32, `max_total_num_tokens` 23.27M, 15.45 GB free after capture (float32 15.5 GB). FP8 KV "no scaling factors" warning twice (target + draft), as before.
- 32 running at n32, `#queue-req` max 2, 0 decode batches with `cuda graph: False` (the 187 such lines are all prefill batches), accept len mean 4.04 over 101 logged decode batches (sweep 4.02). Collapse probe 0/16, tool calls 4/4, no truncation, no errors.
- Engine gen throughput max at 28-32 running: 505 (float32 sweep head log: 429).
- Verdict: bfloat16 is neutral to slightly positive on TP4 (n1 -1 %, engine max at 28-32 running +18 %; engine max in Run 8: +9 % n4, +13 % n8, +9 % n16, +12 % n32); keep it. Comparison table: `SINGLE_CLUSTER_COMPARE.md`.

## Run 8: engine-log rerun (2026-10-12)

The four MTPv2 configs re-run on 2026-10-12 after a power cycle of all Sparks, one rollout per config from a scratch copy of the repo with only the config's profile keys changed, cluster idle-checked (3 min without POSTs/running requests) before each deploy. Levels n4/n8/n16/n32 with 16/32/64/128 questions, 20 s pause, max-tokens 8192, temperature 0, harness `sglang-gsm8k` (token-exact streaming via `continuous_usage_stats`); n1 not rerun. Throughput comes from the head log (`Decode batch` lines, log interval 40 decode steps): each cell is the highest `gen throughput` (tok/s) in the level's time window. SGLang sums every generated token between two log lines and divides by the elapsed wall time, so each line is a measured rate over a few seconds. These numbers replace the client-side figures of 2026-10-11. The old `concurrent_peak_tok_s` n>1 values were removed from Runs 1-7 because they overstate full load.

engine max tok/s:

| config                          | slots | bench (UTC) | n4  | n8  | n16 | n32 | accept len |
|---------------------------------|------:|-------------|-----|-----|-----|-----|-----------:|
| 4/1/5, bfloat16 SSM (= profile) |    32 | 10:34-10:42 | 211 | 287 | 358 | 485 |       4.04 |
| 4/1/5, float32 SSM              |    32 | 13:44-13:53 | 193 | 253 | 327 | 432 |       4.05 |
| 3/1/4, float32 SSM              |    32 | 14:02-14:10 | 195 | 254 | 355 | 412 |       3.47 |
| 5/1/6, float32 SSM              |    32 | 14:19-14:28 | 192 | 267 | 299 | 404 |       4.54 |

- GSM8K (n4/n8/n16/n32): 15-16/16, 31-32/32, 63-64/64, 123-125/128 (4/1/5 bf16 +1 truncation). Head log POSTs = 240 of 240 expected in every case, 0 decode batches with `cuda graph: False`, `max_running_requests` 32 in all four.
- n32: 4/1/5 bf16 485 > 4/1/5 float32 432 > 3/1/4 412 and 5/1/6 404. bfloat16 vs float32 at 4/1/5: +9 % n4, +13 % n8, +9 % n16, +12 % n32; keep bfloat16.
- Day-to-day: engine max at n32 vs 2026-10-11 (before a power cycle of all Sparks): 4/1/5 bf16 485 vs 493, 4/1/5 float32 432 vs 460, 3/1/4 412 vs 412, 5/1/6 404 vs 446.
- "5/1/7" in Runs 3/4 ran as 5/1/6: with topk 1 SGLang rewrites draft tokens to steps+1.
- Comparison with the single node and the other models: `SINGLE_CLUSTER_COMPARE.md`.

## Run 9: rerun with dual-rail and 4 NCCL channels (2026-10-19)

The four MTPv2 configs of Run 8 re-run on 2026-10-19 with the same method (one rollout per config from a scratch copy of the repo with only the config's profile keys changed, idle check before each deploy, levels n1/n4/n8/n16/n32 with 4/16/32/64/128 questions, 20 s pause, max-tokens 8192, temperature 0, harness `sglang-gsm8k`, engine max from the head log). Changed since Run 8: NCCL over both QSFP rails (`NCCL_IB_HCA` with both VFs) and `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS` = 4 instead of the 16-channel default (now the role default, see `NCCL_DUALRAIL_QSFP_TEST.md`). Engine settings otherwise identical (checked in `server_args` per config). No power cycle before the run. n1 = single-stream tok/s of this run (Run 8 did not rerun n1; the value in parentheses is from Run 7, float32 configs from Run 4).

engine max tok/s (Run 8 in parentheses):

| config                          | slots | bench (UTC) | n1          | n4        | n8        | n16       | n32       | accept len (level medians) |
|---------------------------------|------:|-------------|-------------|-----------|-----------|-----------|-----------|---------------------------:|
| 4/1/5, bfloat16 SSM (= profile) |    32 | 11:55-12:01 | 83.8 (82.1) | 202 (211) | 307 (287) | 482 (358) | 704 (485) |                  4.08-4.14 |
| 4/1/5, float32 SSM              |    32 | 12:09-12:16 | 84.5 (82.7) | 201 (193) | 290 (253) | 421 (327) | 613 (432) |                  4.02-4.15 |
| 3/1/4, float32 SSM              |    32 | 12:22-12:29 | 79.2 (77.9) | 199 (195) | 278 (254) | 395 (355) | 581 (412) |                  3.45-3.52 |
| 5/1/6, float32 SSM              |    32 | 12:37-12:44 | 84.5 (85.2) | 205 (192) | 304 (267) | 444 (299) | 599 (404) |                  4.40-4.60 |

- Checks (all finished configs): head log POSTs = questions in every level (240 per config), 0 decode batches with `cuda graph: False`, `#running-req` reached n in every level (32 at n32, no queue). GSM8K profile 4/4, 15/16, 32/32, 63/64, 124/128, no truncation; 4/1/5 float32 4/4, 16/16, 32/32, 63/64, 125/128, 1 truncation; 3/1/4 float32 4/4, 16/16, 32/32, 64/64, 125/128; 5/1/6 float32 4/4, 16/16, 32/32, 62/64, 124/128.
- Against Run 8 (n4/n8/n16/n32): profile -4/+7/+35/+45 %, 4/1/5 float32 +4/+15/+29/+42 %, 3/1/4 float32 +2/+9/+11/+41 %, 5/1/6 float32 +7/+14/+48/+48 %; n1 within 2 % of Run 4/7 for all four (5/1/6 -1 %). Both: within noise up to n8 (float32 n8 +15 % at the edge), a clear gain from n16 on, the same pattern as the Qwen3.8-27B TP=4 rerun (where the speculative configs gained up to x2.4).
- n32 ranking: profile 704 > 4/1/5 float32 613 > 5/1/6 float32 599 ≈ 3/1/4 float32 581 (Run 8: 485 > 432 > 412 ≈ 404); the float32 configs are within noise of each other at n32. 5/1/6 has the highest accept len (4.4-4.6) and the best float32 n16 (444).
- bfloat16 vs float32 SSM (4/1/5): 0 % n4, +6 % n8, +14 % n16, +15 % n32 (Run 8: +9 to +13 %); keep bfloat16.
- Factor 4 nodes / 1 node (4/1/5 bfloat16 on both sides, 1n values from the spark5 log): x2.6 n1 (83.8 / 32.2), x2.5 n4 (202 / 81), x2.6 n8 (307 / 119), x2.7 n16 (482 / 178), x3.0 n32 (704 / 236); in Run 8 it fell to x2.0-2.1 at n16/n32.
- Comparison with the other models: `SINGLE_CLUSTER_COMPARE.md`, section "Cluster rerun 2026-10-19".
