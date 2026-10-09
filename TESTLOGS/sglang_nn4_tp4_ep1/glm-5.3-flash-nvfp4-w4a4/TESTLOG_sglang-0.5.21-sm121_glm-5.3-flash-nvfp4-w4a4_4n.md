# SGLang Test Log: GLM-5.3-Flash NVFP4 W4A4, 4 Nodes, TP=4 EP=1, 0.5.21-sm121 (GSM8K chat-API run)

## Environment and config

| Component                    | Value                                                                                |
|------------------------------|--------------------------------------------------------------------------------------|
| GPU                          | NVIDIA GB10 (SM121), 4 nodes (spark1-4), 1 GPU each                                  |
| Driver                       | not recorded in this run (file name therefore has no driver tag)                     |
| Image                        | `xomoxcc/dgx-spark-sglang:0.5.21-sm121`                                              |
| Model                        | `vroomfondel/GLM-5.3-Flash-NVFP4-W4A4` (reasoning model, `--reasoning-parser glm45`) |
| Parallelism                  | TP=4, EP=1                                                                           |
| Attention                    | tilelang DSA, prefill + decode                                                       |
| KV cache                     | bf16, page size 64                                                                   |
| dsa_paged_mqa_logits_backend | torch                                                                                |
| MoE runner                   | flashinfer_cutlass                                                                   |
| Speculative decoding         | off                                                                                  |
| mem_fraction_static          | 0.80                                                                                 |
| Server limits                | max_running_requests 32, chunked_prefill_size 8192                                   |
| Runtime patches              | p30, p35, p68, p69, p70                                                              |

## Run parameters

- Date: 2026-10-08, 17:48 to 18:02 CEST. Read-only access via kubectl port-forward to svc/sglang.
- Custom chat-API harness, now `scripts/debughelper/gsm8k_chat_harness.py` (stock sglang GSM8K harness unusable: needs py3.12 `urlopen(cafile=)` and the raw completion endpoint). Dataset: GSM8K test split (openai/grade-school-math test.jsonl, 1319 questions), questions dispatched in file order.
- Prompt: question + "Solve step by step. The final line of your answer must be exactly `#### <number>`". Parse `#### N`, fallback last number in content.
- max_tokens 8192, temperature 0, concurrency 8, per-request HTTP timeout 600 s.
- Budget: stop dispatch at 13.5 min, hard stop 15 min. Scoring numeric (1e-2 relative or 1e-6 absolute); empty content or finish_reason length counts as truncated.
- Raw data: `/tmp/claude-1000/-home-thiess-pythondev-workspace-dgxarley/5561cf3b-496a-4367-8f7f-04e22c4a2b50/scratchpad/gsm8k/results.jsonl` (one record per request), `/tmp/claude-1000/-home-thiess-pythondev-workspace-dgxarley/5561cf3b-496a-4367-8f7f-04e22c4a2b50/scratchpad/gsm8k/head.log` (head log, --since=16m).

## Accuracy

| Outcome                      | Count |
|------------------------------|-------|
| n finished (incl. error)     | 317   |
| correct                      | 312   |
| wrong                        | 4     |
| truncated                    | 0     |
| error (client timeout 600 s) | 1     |
| unfinished at hard stop      | 0     |

- Accuracy over scored answers (316): 312/316 = 98.7 percent. Over all 317 finished: 98.4 percent. Exact-integer matches: 311 (one correct answer was a numerically equal non-identical rendering).
- finish_reason: stop for all 316 answered requests, no length truncation, every answer contained a `####` line.
- Reasoning present in reasoning_content for all 316 (max 4566 chars); the server usage block gives no separate reasoning token count, so reasoning tokens are not reported. Mean completion tokens per request: 239 (max 1483).

## Latency and throughput

- Latency per request (wall, s): p50 16.8, p95 36.1; the largest successful one was 109.4 s; the errored request hit the 600 s timeout.
- Aggregate: 75619 completion tokens / 843.7 s = 89.6 tok/s (wall includes the 600 s timeout tail of the last in-flight request, so this understates steady state).
- Concurrent peak (max over time of the sum of per-request tok/s of in-flight requests): 112.8 tok/s.
- Scheduler log "Decode batch ... gen throughput (token/s)" (273 lines in the window): min 1.11, median 100.75, max 130.16. #running-req: 8 in 246 lines, 7 in 10, 1 in 13 (idle ramp-down), 5/6/9 once or twice each. #queue-req stayed 0. Single-request decode at the tail: ~31 tok/s.

## Example failures (all four wrong answers are plausible model reasoning slips)

| Question id (0-based test index) | Expected | Got      | Tail of content                                                     |
|----------------------------------|----------|----------|---------------------------------------------------------------------|
| 12                               | 13       | 12       | "...After 12 years, Carlos has earned back the $90 cost... #### 12" |
| 119                              | 95200    | 99076.92 | "...56000 + 43076.92 = \$99,076.92 ... #### 99076.92"               |
| 255                              | 192      | 176      | "20 + 2 = 22 cows per stall. ... 8 x 22 = 176 cows. #### 176"       |
| 245                              | 7        | 14       | "Each aquarium has 14 fish, and twice 14 is 28... #### 14"          |
| 89 (error)                       | 24       | none     | client TimeoutError after 600.1 s, no response                      |

## Anomalies

- No NaN, no garbage text, no runs of the same character of 20 or more in any content. Non-ASCII characters in content are only legitimate math/typography symbols (x, minus, division, arrow, check mark, dash, euro).
- No tracebacks and no 5xx in the head log for the window; port-forward stayed up.
- Request 89 never returned within 600 s. The server showed no abort or error in the log and #running-req stayed at 8 meanwhile. At ~13 tok/s per request under concurrency 8, 600 s is roughly 7800 tokens, so this is consistent with a runaway generation near the 8192 cap, but this was not verified (no server-side per-request trace). Treat as an open item: a single runaway generation per ~317 requests.

---

# Run 2: MTP/EAGLE speculative decoding enabled

## Config delta vs run 1

- Speculative decoding on: built-in MTP head (same checkpoint as draft), EAGLE steps 5 / topk 1 / draft tokens 6.
- Patches p30/p35/p42 (fixed)/p68/p69/p70 (run 1 had no p42).
- Everything else identical (TP4, tilelang DSA, bf16 KV, page 64, torch logits backend, flashinfer_cutlass, mem 0.80). Head pod sglang-head-84dfb74db9-h7wvd, up since 16:31:56 UTC, no restarts during the run.
- Same harness, dataset order, question ids and parameters as run 1 (concurrency 8, max_tokens 8192, temperature 0, 13.5 min dispatch stop, 600 s timeout). Run 18:33 to 18:47 CEST.
- Raw data: `/tmp/claude-1000/-home-thiess-pythondev-workspace-dgxarley/5561cf3b-496a-4367-8f7f-04e22c4a2b50/scratchpad/gsm8k/results_mtp.jsonl`, `/tmp/claude-1000/-home-thiess-pythondev-workspace-dgxarley/5561cf3b-496a-4367-8f7f-04e22c4a2b50/scratchpad/gsm8k/head_mtp.log`.

## Results

- n = 375 finished: 363 correct, 12 wrong, 0 truncated, 0 errors, 0 unfinished. Accuracy 96.8 percent (exact-integer matches 363). All finish_reason stop.
- Latency: p50 14.8 s, p95 33.2 s, max 144.6 s. No request timed out.
- Aggregate: 89081 completion tokens / 823.0 s = 108.2 tok/s. Concurrent peak: 140.3 tok/s. Mean completion 238 tokens per request.
- Scheduler log (76 Decode batch lines, the log is sampled sparsely): gen throughput min 1.36, median 107.5, max 146.8 tok/s. At #running-req 8 (63 lines): median 107.8, min 6.26, max 146.8. #running-req: 8 in 63 lines, 7 in 4, 1 in 7, 6 in 1, 3 in 1.
- Accept len (76 lines): min 2.30, median 4.14, max 4.59. At #running-req 8: min 3.53, median 4.14, max 4.55.
- Head log: 0 tracebacks, 0 5xx.

## Comparison

| Metric                              | Run 1 (spec off)                     | Run 2 (MTP/EAGLE 5/1/6)   |
|-------------------------------------|--------------------------------------|---------------------------|
| n finished                          | 317                                  | 375                       |
| Correct / wrong / truncated / error | 312 / 4 / 0 / 1                      | 363 / 12 / 0 / 0          |
| Accuracy (scored answers)           | 98.7 percent                         | 96.8 percent              |
| Latency p50 / p95 (s)               | 16.8 / 36.1                          | 14.8 / 33.2               |
| Aggregate tok/s                     | 89.6 (wall incl. 600 s timeout tail) | 108.2                     |
| Concurrent peak tok/s               | 112.8                                | 140.3                     |
| Scheduler median / max tok/s        | 100.8 / 130.2                        | 107.5 / 146.8             |
| Accept len (median)                 | n/a                                  | 4.14 (min 2.30, max 4.59) |
| Questions finished in ~13.5 min     | 317                                  | 375 (+18 percent)         |

- Same-question check on the 317 ids both runs finished: run 1 312 correct, run 2 308 correct (counting run 1's timeout as not correct). Only in run 1 wrong: 89 (timeout), 245. Only in run 2 wrong: 7, 37, 62, 111, 209, 267. Questions 12, 119 and 255 are wrong in both runs. Outputs are not identical at temperature 0, as expected with speculative verification.
- Over the first 317 questions run 2 has 9 wrong against 4 wrong plus 1 timeout in run 1 (308 vs 312 correct). With these sample sizes a difference of this size is within noise but leans toward a small accuracy cost; a larger run would be needed to settle it.
- Throughput gain is modest (about 20 percent aggregate, 24 percent peak) despite accept len above 4, consistent with concurrency 8 already filling the batch and verify cost growing with batch size.

## Example failures (run 2)

| Question id | Expected | Got | Tail of content                             |
|-------------|----------|-----|---------------------------------------------|
| 7           | 160      | 120 | "40 + 20 + 60 = 120 minutes #### 120"       |
| 37          | 2        | 0   | "...all his toys were sold.) #### 0"        |
| 209         | 145      | 70  | "16800 / 240 = 70 #### 70"                  |
| 267         | 91       | 51  | "Total = 28 + 23 = 51 minutes. #### 51"     |
| 342         | 87       | 63  | "Total outfits: 24 + 48 + 15 = 63. #### 63" |

## Anomalies (run 2)

- No NaN, no garbage text, no runs of the same character of 20 or more. Non-ASCII characters are legitimate math/typography symbols (x, minus, division, arrow, check mark, dash, euro). No "!" runs.
- All wrong answers are arithmetic or reading-comprehension slips with coherent text, no sign of degenerate speculative output.
- No request hung this time (run 1's 600 s timeout did not recur).

---

# Run 3: full GSM8K test split (1319 questions), MTP/EAGLE enabled

## Config

- Identical to run 2: image `xomoxcc/dgx-spark-sglang:0.5.21-sm121`, TP=4 EP=1, MTP/EAGLE steps 5 / topk 1 / draft tokens 6, tilelang DSA prefill + decode, bf16 KV, page 64, dsa_paged_mqa_logits_backend torch, flashinfer_cutlass MoE, mem_fraction_static 0.80, max_running_requests 32, patches p30/p35/p42/p68/p69/p70. Head pod sglang-head-84dfb74db9-h7wvd, 0 restarts before and after the run.

## Run parameters

- Date: 2026-10-08, 18:59 to 19:52 CEST (16:59:33 to ~17:51:45 UTC). Read-only access via kubectl port-forward to svc/sglang (stayed up).
- Same harness and prompt as runs 1 and 2, all 1319 questions dispatched in file order. Concurrency 8, max_tokens 8192, temperature 0, per-request HTTP timeout 900 s, budget 90 min with dispatch-stop-frac 1.0. Wall 3129.3 s (52.2 min), 0 unfinished workers.
- Raw data: `/tmp/claude-1000/-home-thiess-pythondev-workspace-dgxarley/5561cf3b-496a-4367-8f7f-04e22c4a2b50/scratchpad/gsm8k_full/results_full_mtp.jsonl`, head log `/tmp/claude-1000/-home-thiess-pythondev-workspace-dgxarley/5561cf3b-496a-4367-8f7f-04e22c4a2b50/scratchpad/gsm8k_full/head_full.log`.

## Accuracy

| Outcome                                         | Count |
|-------------------------------------------------|-------|
| n finished                                      | 1319  |
| correct                                         | 1271  |
| wrong                                           | 45    |
| truncated (finish_reason length, empty content) | 3     |
| error                                           | 0     |
| unfinished                                      | 0     |

- Accuracy over scored answers (1316): 1271/1316 = 96.58 percent. Over all 1319: 96.36 percent. Exact-integer matches: 1271 (all correct answers).
- Truncated ids: 62, 119, 1119. Each hit the 8192-token cap inside reasoning_content (24.6k, 26.1k and 30.0k chars) without producing a final answer. Their latencies (609 s, 616 s, 519 s) are the long tail of the latency distribution.
- Mean completion tokens per request: 264.7 (349099 total).

## Latency and throughput

- Latency per request (wall, s): p50 14.4, p95 36.7, max 616.0 (a truncated request). Longest non-truncated: 320.6 s (id 916, 4617 tokens).
- Aggregate: 349099 completion tokens / 3129.3 s = 111.6 tok/s. Concurrent peak (sum of per-request tok/s of in-flight requests): 164.1 tok/s.
- Scheduler log "Decode batch" lines in the run window (268 lines, sampled): gen throughput min 0.64, median 109.2, max 176.5 tok/s. At #running-req 8 (240 lines): min 0.64, median 108.9, max 176.5. #running-req: 8 in 240 lines, 7 in 26, 2 in 1, 1 in 1. #queue-req stayed 0.
- Accept len (268 lines): min 3.69, median 4.18, max 5.62. At #running-req 8: min 3.69, median 4.18, max 4.57.
- Head log over the window: 0 tracebacks, 0 "Scheduler hit", all 1319 POST /v1/chat/completions returned 200 (0 5xx).

## Example failures

| Question id | Expected | Got  | Tail of content                                                          |
|-------------|----------|------|--------------------------------------------------------------------------|
| 12          | 13       | 12   | "...After 12 years, the tree has paid back its planting cost... #### 12" |
| 102         | 26       | 24   | "2 glasses x 2 days = 4 glasses ... 20 + 4 = 24 #### 24"                 |
| 143         | 16       | 13   | "Total cost = $2 + $3 + $3 + $4 + $1 (tax) = $13 #### 13"                |
| 357         | 2        | 3    | "$90 / 30 = 3 ... Morgan can invite 3 friends #### 3"                    |
| 403         | 81       | 135  | "...Mel will save 135 kilowatts of electric energy in 30 days. #### 135" |
| 255         | 192      | 176  | "8 x 22 = 176 #### 176"                                                  |
| 62          | 25000    | none | truncated at 8192 tokens, content empty                                  |
| 119         | 95200    | none | truncated at 8192 tokens, content empty                                  |

## Anomalies

- No runs of 20 or more identical characters in any content. Non-ASCII characters in content are only math/typography symbols (x, minus, division, arrow, check mark, em/en dash, degree, euro, cent, a few emoji marks), no garbage and no "!" runs.
- The only degenerate behaviour is 3 runaway reasoning generations (0.23 percent of requests) that exhausted the token cap, counted as truncated and not as wrong. Run 1's single 600 s timeout was the same failure type.
- All wrong answers are coherent arithmetic or reading-comprehension slips. Throughput remained stable for 52 min, no degradation over time.

## Comparison with runs 1 and 2 (shared ids)

| Metric                | Run 1 (spec off, 317) | Run 2 (MTP, 375) | Run 3 (MTP, 1319) |
|-----------------------|-----------------------|------------------|-------------------|
| Accuracy (scored)     | 98.7 percent          | 96.8 percent     | 96.6 percent      |
| Latency p50 / p95 (s) | 16.8 / 36.1           | 14.8 / 33.2      | 14.4 / 36.7       |
| Aggregate tok/s       | 89.6                  | 108.2            | 111.6             |
| Concurrent peak tok/s | 112.8                 | 140.3            | 164.1             |
| Accept len median     | n/a                   | 4.14             | 4.18              |

- Run 3 vs run 2 on the 375 shared ids: 364 vs 363 correct, wrong in both: 12, 37, 255, 342, 353, 368. Run 3 reproduces run 2 closely (same config, near-identical accuracy and accept len).
- Run 3 vs run 1 on the 317 shared ids: 310 vs 312 correct (run 1's timeout counted as not correct), wrong in both: 12, 255. The gap is 2 questions, so the earlier suspicion of an MTP accuracy cost is not supported at a meaningful level on this evidence, but the full-set MTP accuracy of 96.6 percent sits below run 1's 98.7 percent on its (smaller, easier-ordered prefix) sample; a full spec-off run would be needed for a like-for-like number.
- Throughput gain of MTP holds at scale: 111.6 tok/s aggregate vs 89.6 (run 1, understated by its timeout tail).

## Run 4: concurrency sweep (metrics only)

Config: same as runs 2 and 3 (model `vroomfondel/GLM-5.3-Flash-NVFP4-W4A4`, TP=4, MTP/EAGLE 5/1/6 on, tilelang DSA, bf16 KV, image 0.5.21-sm121). Harness `scripts/debughelper/gsm8k_chat_harness.py run`, temperature 0, max_tokens 8192, one fixed concurrency per level, started after the full run 3 had finished and the scheduler had been idle for 2 min.

Design: per level n in 1, 2, 4, 8, 16, 32 the first 2n GSM8K test questions (`--limit 2n`), 30 s pause between levels. Accuracy at these sample sizes is not statistically meaningful; this is a throughput measurement. Head pod restarts before and after: 0 / 0.

| n  | questions | wall s | agg tok/s | peak tok/s | mean per-req tok/s | sched median tok/s @n | accept len median | lat p50 s | lat max s | trunc | correct/scored |
|----|-----------|--------|-----------|------------|--------------------|-----------------------|-------------------|-----------|-----------|-------|----------------|
| 1  | 2         | 5.8    | 48.4      | 50.5       | 48.1               | 33.5 (max 49.0)       | 4.10              | 2.9       | 3.3       | 0     | 2/2            |
| 2  | 4         | 12.2   | 61.8      | 70.5       | 33.5               | 36.2 (max 63.8)       | 3.88              | 4.9       | 8.7       | 0     | 4/4            |
| 4  | 8         | 16.6   | 93.4      | 121.8      | 29.5               | 65.2 (max 121.3)      | 4.02              | 6.7       | 9.7       | 0     | 7/8            |
| 8  | 16        | 35.8   | 107.5     | 132.5      | 16.4               | 69.3 (max 108.9)      | 4.04              | 13.2      | 24.6      | 0     | 15/16          |
| 16 | 32        | 54.2   | 137.3     | 171.9      | 9.9                | 74.0 (max 139.9)      | 4.01              | 21.6      | 46.9      | 0     | 31/32          |
| 32 | 64        | 119.2  | 125.3     | 210.2      | 5.2                | 59.6 (single sample)  | 4.12              | 47.4      | 99.7      | 0     | 60/64          |

Notes:
- Aggregate tok/s = total completion tokens / harness wall (includes the ramp-up and the tail where fewer than n requests are in flight, so it understates steady state at small question counts). Peak tok/s is the harness `concurrent_peak_tok_s` (sum of per-request tok/s over concurrently running requests).
- Scheduler columns come from head-log "Decode batch" lines with `#running-req == n`. The server logs only every ~192 decode steps, so there are only 1 to 2 samples per level; the scheduler medians are indicative only. `#queue-req` stayed 0 in every window, including n=32 (max_running_requests 32). Full token usage stayed 0.00 at every level (short GSM8K contexts).
- Aggregate throughput scales 48 -> 137 tok/s up to n=16 and is flat to slightly lower at n=32 (125 tok/s) for this short run; the n=32 wall is dominated by one 2-wave tail (64 questions on 32 slots).
- MTP accept length is stable at about 4.0 across all levels.

Anomalies:
- First attempt of levels 16 and 32 was invalid: my local `kubectl port-forward` died during level 16 (8 connection errors in n=16, all 32 requests of n=32 refused). Those files are kept as `sweep_n16_pfdied.jsonl` and `sweep_n32_pfdied.jsonl` and are not used. Levels 16 and 32 were re-run with a supervised port-forward loop; levels 1 to 8 are from the first pass. No server-side error: one FastAPIDeprecationWarning line (ORJSONResponse, `create_error_response`) at 20:01:45 around the aborted client connections, no tracebacks, no "Scheduler hit", no 5xx.
- The "Decode batch ... #running-req: 1" lines after the sweep belong to the other agent's traffic; a few windows contain 2 to 6 decode lines with a different running count (ramp-up/tail), excluded from the @n columns.

Raw data: `<scratchpad>/gsm8k_sweep/sweep_n{1,2,4,8,16,32}.jsonl` (scratchpad `/tmp/claude-1000/-home-thiess-pythondev-workspace-dgxarley/5561cf3b-496a-4367-8f7f-04e22c4a2b50/scratchpad`), head log slice `headlog.txt` in the same dir.
