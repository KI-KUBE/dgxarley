# SGLang Test Log: Nemotron-3 Nano Omni 30B-A3B-Reasoning-NVFP4, 4 Nodes, TP=4 EP=1, v0.5.21-sm121 (throughput re-matrix)

## Environment

| Component | Value                                                                       |
|-----------|-----------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121), 128 GB unified per node                                |
| Nodes     | spark1 (head/rank0), spark2, spark3, spark4 (1 GB10 each)                   |
| Image     | `xomoxcc/dgx-spark-sglang:0.5.21-sm121` (cluster default, no profile pin)   |
| Model     | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`                       |
| Transport | RoCE via SR-IOV VF                                                          |
| Parallel  | tp=4, pp=1, ep=1 (ep=4 retested in case 02)                                 |
| Date      | 2026-10-08                                                                  |

Profile: `roles/k8s_dgx/model_profiles/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4.yml`
Predecessor: `TESTLOG_nv580.159_sglang-0.5.13-sm121_nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4_4n.md` (2026-06-25; case 02 profile default: n=1 90.1, n=8 peak 437.9, n=16 660.8).

Method: one Ansible rollout per case (`ansible-playbook k8s_dgx.yml --tags sglang`), bench via `sglang-gsm8k` (chat API, GSM8K, temperature 0, max_tokens 8192), levels n=1/4/8/16 with 4/8/16/32 questions, 20 s pause between levels. Peak = harness `concurrent_peak_tok_s` (sum of concurrent per-request tok/s). The checkpoint was pre-warmed on JuiceFS (100 % on all four nodes), so boots are not load-bound.

## Matrix (6 cases: 5 run, 1 skipped)

| #  | Delta vs case 01                                   | Status      | n=1 tok/s | n=4 peak | n=8 peak | n=16 peak | Output  |
|----|----------------------------------------------------|-------------|----------:|---------:|---------:|----------:|---------|
| 01 | none (profile as is)                               | ok          | 118.5     | 329.8    | 507.3    | 769.5     | clean   |
| 02 | `ep_size: 4`                                       | ok (slower) | 110.3     | 309.2    | 465.8    | 753.1     | clean   |
| 03 | `mamba_backend: flashinfer`                        | ok (slower) | 114.7     | 293.0    | 485.0    | 734.3     | clean   |
| 04 | `mamba_ssm_dtype: bfloat16`                        | ok (neutral)| 116.1     | 312.0    | 511.5    | 793.2     | clean   |
| 05 | `ep_size: 4` + `mamba_backend: flashinfer`         | skipped: 03 did not win | - | - | - | - | 03 below baseline at n8 and n16 |
| 06 | `cuda_graph_backend_prefill: breakable`            | ok (no-op, neutral) | 110.6 | 340.6 | 483.7 | 762.7 | clean   |

Cases 03, 05 and 06 were skipped in the first run (2026-10-08) because: `sglang_launch.sh` and `sglang_instance.yml` have no `mamba_backend` env/flag and no `cuda_graph_backend_prefill` variable (only `disable_piecewise_cuda_graph`, which maps to `--cuda-graph-backend-prefill disabled`; `disable_piecewise_cuda_graph: false` would leave the SGLang default, not an explicit `breakable`). Boot logs confirm the effective `mamba_backend: triton` and `cuda_graph_backend_prefill: disabled`. The plumbing was added afterwards and cases 03 and 06 were run on 2026-10-09 (see the addendum below); 05 stays skipped.

### Detail per level

Decode columns are from the head log "Decode batch" lines at `#running-req == n` inside the level window (samples in brackets). `#queue-req` was 0 everywhere. Mamba usage peak 0.01, full token usage peak 0.00.

| Case | n  | aggregate tok/s | mean per-req tok/s | latency p50 / max (s) | correct | decode median / max tok/s (samples) |
|------|----|----------------:|-------------------:|----------------------:|--------:|------------------------------------:|
| 01   | 1  |           116.5 |              115.6 |             3.2 / 8.7 |     4/4 |                  119.3 / 121.2 (50) |
| 01   | 4  |           240.1 |               76.2 |            6.2 / 12.1 |     8/8 |                  321.3 / 335.4 (18) |
| 01   | 8  |           335.4 |               59.7 |            8.4 / 19.3 |   15/16 |                  514.0 / 526.3 (17) |
| 01   | 16 |           652.0 |               46.3 |            8.4 / 22.6 |   31/32 |                  729.0 / 780.3 (17) |
| 02   | 1  |           107.4 |              107.5 |            3.9 / 10.4 |     4/4 |                  111.1 / 115.5 (55) |
| 02   | 4  |           224.7 |               70.7 |            4.6 / 16.6 |     8/8 |                  296.8 / 300.1 (13) |
| 02   | 8  |           343.2 |               54.8 |            7.9 / 19.2 |   15/16 |                  471.3 / 489.3 (15) |
| 02   | 16 |           547.4 |               44.9 |            8.7 / 28.5 |   32/32 |                  699.8 / 750.3 (18) |
| 04   | 1  |            98.8 |              101.8 |            3.9 / 10.4 |     4/4 |                  119.0 / 122.4 (49) |
| 04   | 4  |           237.5 |               67.4 |            5.6 / 15.3 |     8/8 |                  245.7 / 337.0 (18) |
| 04   | 8  |           389.4 |               60.2 |            7.2 / 19.2 |   15/16 |                  526.8 / 541.8 (16) |
| 04   | 16 |           607.4 |               44.6 |            8.2 / 23.4 |   31/32 |                  704.0 / 802.6 (18) |
| 03   | 1  |            95.1 |               95.6 |   3.3 / 7.7 (p95 7.7) |     4/4 |                  108.5 / 120.3 (38) |
| 03   | 4  |           197.9 |               64.5 | 7.4 / 15.1 (p95 15.1) |     8/8 |                  223.3 / 332.0 (19) |
| 03   | 8  |           321.1 |               53.5 | 7.9 / 18.8 (p95 18.8) |   16/16 |                  411.5 / 517.9 (16) |
| 03   | 16 |           566.6 |               40.6 | 9.1 / 23.5 (p95 22.4) |   31/32 |                  676.4 / 772.0 (17) |
| 06   | 1  |           107.0 |              108.0 | 2.8 / 14.3 (p95 14.3) |     4/4 |                  117.5 / 120.6 (59) |
| 06   | 4  |           238.5 |               78.1 | 5.9 / 10.3 (p95 10.3) |     8/8 |                  330.0 / 332.1 (17) |
| 06   | 8  |           380.2 |               58.5 | 7.7 / 24.1 (p95 24.1) |   15/16 |                  503.1 / 517.8 (19) |
| 06   | 16 |           617.7 |               46.1 | 8.7 / 24.8 (p95 23.2) |   31/32 |                  715.9 / 779.1 (19) |

Short runs (17 to 30 s per level): differences of a few percent between cases are within run-to-run noise.

## Boot facts (case 01, head log)

- Runtime patch p59 applied: `Patched nemotron_h.py: NemotronH VL/Omni wrapper: overrides dispatch + llm_config resolution` (head log line 81).
- Effective server_args: tp 4, ep 1, mamba_backend triton, mamba_ssm_dtype None (model default float32), mamba_radix_cache_strategy auto, mamba_full_memory_ratio 0.9, cuda_graph decode backend full, bs [1,2,4,8,12,16,24,32], prefill backend disabled, max_running_requests 32.
- `Mamba Cache is allocated. max_mamba_cache_size: 2211, conv_state size: 0.44GB, ssm_state size: 24.84GB`
- `KV Cache is allocated. dtype: float8_e4m3fn, #tokens: 19644177, K size: 14.05 GB, V size: 14.05 GB`; avail mem 42.45 GB after pools, 35.27 GB after graph capture.
- Decode graph capture 6.01 s (0.73 GB). Tree cache: UnifiedRadixCache, components FULL + MAMBA, hybrid_ssm=True.
- Boot to "fired up": about 3 min from head start with warm cache.
- Multimodal smoke (PNG 128x128 red circle on white, data URL): content `A red circle`, reasoning_content populated and sensible. Image path works on 0.5.21.

## Case notes

- **02 (ep_size 4):** the 0.5.13 crash in `flashinfer_backend.py` `init_cuda_graph_state` does not reproduce. Boots without any traceback (ranks logged as `TP0 EP0`), mamba pool 2244, KV 19.9M tokens, graph capture 6.12 s. Serves clean, 32/32 at n=16. Throughput is lower than ep=1: n=1 -7 %, n=4 -6 %, n=8 -8 %, n=16 -2 %. EP gives no win on this 3B-active model; keep ep=1.
- **04 (mamba_ssm_dtype bfloat16):** `max_mamba_cache_size` doubles (2211 to 4348, ssm_state 24.42 GB), but `max_running_requests` stays 32 (explicit env clamp), so the larger pool is unused at n<=16. n=8 peak +0.8 %, n=16 peak +3.1 %, decode median at n=8 +2.5 %: below the 5 % adoption threshold. Quality gate passed, see below. Not adopted.

## Quality gate

| Check                           | 01                                                                                                         | 02           | 04           |
|---------------------------------|------------------------------------------------------------------------------------------------------------|--------------|--------------|
| GSM8K correct n=8 / n=16        | 15/16, 31/32                                                                                               | 15/16, 32/32 | 15/16, 31/32 |
| Same-char runs >= 20            | none                                                                                                       | none         | none         |
| `content` starts with `<think>` | 0                                                                                                          | 0            | 0            |
| `reasoning_content` populated   | all requests                                                                                               | all          | all          |
| Odd non-ASCII                   | only typographic: U+202F, U+2011, U+2013, U+2019, U+201C/D, U+2192, U+2212, U+00D7, U+00F7, U+2026, U+2248 | same set     | same set     |
| Truncations / errors            | 0                                                                                                          | 0            | 0            |

The only wrong answer in 01/04 (and in 02 at n=8) is question id 12 (gt 13, answered 12, finish_reason stop), an off-by-one reading of the question, identical across cases; case 02 n=16 happened to answer it correctly. Tails of 3 answers per case end in a clean `#### <number>` line with coherent arithmetic. bfloat16 SSM state shows no quality loss at this sample size (small: 60 answers).

## Results and decision

- 0.5.21-sm121 baseline is clearly faster than 0.5.13-sm121: n=1 118.5 vs 90.1 (+32 %), n=8 peak 507.3 vs 437.9 (+16 %), n=16 769.5 vs 660.8 (+16 %).
- No tested variant beats case 01 at n=8 by >= 5 %: profile keys unchanged (ep_size 1, mamba_ssm_dtype unset). Only header notes were refreshed (AUDIO note, EP=4 retest, image note).
- 2026-10-09 addendum: `mamba_backend flashinfer` (-4.4 % n=8, -4.6 % n=16) and `cuda_graph_backend_prefill breakable` (no-op, prefill graphs stay off on this hybrid model) do not beat the baseline; profile keys stay absent. Open: audio input untested.
- Cluster left serving this model on the case-01 profile state.

## Matrix addendum 2026-10-09 (cases 03/05/06 via new mamba_backend / cuda_graph_backend_prefill plumbing)

Same method, levels and harness options as the 2026-10-08 run (n=1/4/8/16 with 4/8/16/32 questions, 20 s pauses). Image `xomoxcc/dgx-spark-sglang:0.5.21-sm121`, one rollout per case from the case-01 profile state plus one key.

### Boot facts

- **03:** env `SGLANG_MAMBA_BACKEND=flashinfer` present in the head container env dump, launch line contains `--mamba-backend flashinfer`, effective `server_args` `mamba_backend: 'flashinfer'`, log line `Successfully imported FlashInfer mamba module`. Mamba pool 2230 (ssm_state 25.06 GB), KV 19.82M tokens, decode graph capture 5.98 s (0.71 GB), avail mem 34.20 GB before and 33.49 GB after capture (case 01: 35.27 GB after).
- **06:** env `SGLANG_CUDA_GRAPH_BACKEND_PREFILL=breakable`, launch line contains `--cuda-graph-backend-prefill breakable`, `server_args` `cuda_graph_backend_prefill: 'breakable'` (literal accepted by the image). Mamba pool 2225 (ssm_state 25.00 GB), KV 19.77M tokens, capture 6.03 s (0.85 GB), avail mem 35.69 GB before and 34.84 GB after capture. The head then logs `Disable prefill CUDA graph because some layers do not apply Standard GQA` and every prefill batch reports `cuda graph: False`: the explicit backend is overridden at runtime on this hybrid Mamba model, so the case is functionally identical to baseline.
- Both: 0 restarts, no traceback, `Scheduler hit` absent, patch ConfigMap `sglang-patch-scripts` has no `p71_*` key.

### Results

| Case |   n=1 |   n=4 |   n=8 |  n=16 | vs 01 at n=8 / n=16 | ok (n=1/4/8/16)                       | quality |
|------|------:|------:|------:|------:|---------------------|---------------------------------------|---------|
| 01   | 118.5 | 329.8 | 507.3 | 769.5 | baseline            | 4/8/16/32 requests, 4/8/15/31 correct | clean   |
| 03   | 114.7 | 293.0 | 485.0 | 734.3 | -4.4 % / -4.6 %     | 4/8/16/32 requests, 4/8/16/31 correct | clean   |
| 06   | 110.6 | 340.6 | 483.7 | 762.7 | -4.7 % / -0.9 %     | 4/8/16/32 requests, 4/8/15/31 correct | clean   |

Aggregate tok/s n=8 / n=16: 03 321.1 / 566.6, 06 380.2 / 617.7 (01: 335.4 / 652.0). Quality gate for both: no same-char runs >= 20, `reasoning_content` populated for all requests, no `<think>` in content, 0 truncations or errors, only typographic non-ASCII. The single wrong answers are the usual off-by-one question id 12 (or, in 03 n=8, none).

### Anomalies

- 03 per-token decode is visibly slower at low concurrency (median at running==1 108.5 vs 119.3 in case 01; running==4 223.3 vs 321.3), consistent with the lower n=1 and n=4 peaks. The flashinfer Mamba2 SSM kernel is not faster than triton on GB10 here.
- 06 shows no difference beyond run-to-run noise (runs are 15 to 25 s per level); its n=4 peak (340.6) is above baseline, n=8 below, which is noise, not signal.

### Conclusion

Neither knob beats the baseline by the 5 % adoption threshold at both n=8 and n=16, so no profile key is set (commented one-liners with the measurements were added next to `ep_size`). Case 05 was skipped (03 did not win; ep_size 4 had already lost on its own in case 02). The new plumbing works end to end (profile key to env to launch flag to `server_args`). The cluster was switched back to Nemotron-3 Super afterwards.

## Co-located vision companion next to Super (2026-10-09)

Rollout `ansible-playbook k8s_dgx.yml --tags sglang` (rc 0, 0 failed, 14 changed, 47 s). Both heads rolled at once (Super because `--max-total-tokens 16777216` was added, Omni new). Image `xomoxcc/dgx-spark-sglang:0.5.21-sm121`. `sglang-patch-scripts` has no `p71_*` key. No plumbing fix needed: all four per-instance overrides landed in the vision launch line, and the default instance did not pick up any vision value.

### Instance caps (vault `_sglang_vision_instance`)

TP4 on spark1-4, sriov_key `sglang_vision`, `context_length 131072`, `max_total_tokens 524288`, `max_running_requests 8`, `max_mamba_cache_size 48`, `cuda_graph_max_bs 8`, `mem_fraction_static 0.90`, ClusterIP only.

### Boot facts (TP0 rank)

|                 | Omni (`sglang-vision-head-76f8f85cbf-9dllv`)                                                                                                                   | Super (`sglang-head-6497fb4f88-p5ccj`)                                                                                                      |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| launch flags    | `--context-length 131072 --mem-fraction-static 0.90 --max-mamba-cache-size 48 --max-running-requests 8 --max-total-tokens 524288 --cuda-graph-max-bs-decode 8` | `--mem-fraction-static 0.75 --max-mamba-cache-size 160 --max-running-requests 32 --max-total-tokens 16777216 --cuda-graph-max-bs-decode 32` |
| Load weight end | 51.6 s, mem usage 29.57 GB (inflated, see anomalies), avail 67.42                                                                                              | 122.8 s, mem usage 31.47 GB, avail 60.32; draft 2.15 GB, avail 59.02                                                                        |
| Mamba cache     | 48 slots, conv 0.01 + ssm 0.55 GB                                                                                                                              | 160 slots, conv 0.09 + ssm 6.29 + intermediate ssm 6.45 + intermediate conv 0.04 GB                                                         |
| KV cache (fp8)  | 524288 tokens, K 0.38 + V 0.38 GB                                                                                                                              | 10182073 tokens, K 9.71 + V 9.71 GB; draft K 1.21 + V 1.21 GB                                                                               |
| Memory pool end | avail 65.74 GB                                                                                                                                                 | 25.57 GB, then 23.14 GB (after draft)                                                                                                       |
| CUDA graphs     | bs [1,2,4,8], 5.08 s, mem usage -0.17 GB                                                                                                                       | verify 8.91 s (1.32 GB), draft decode 5.68 s (1.20 GB), draft extend 1.04 s (0.50 GB); bs 1..32 (20 sizes)                                  |
| final           | max_total_num_tokens=524288, available_gpu_mem=60.96 GB                                                                                                        | max_total_num_tokens=10182073, available_gpu_mem=14.61 GB                                                                                   |

No `Scheduler hit`, `Traceback`, OOM or `max_running_requests` clamp lines in either head log. 0 restarts, workers Running.

### Memory per rank

|       | expected | measured                                                     |
|-------|----------|--------------------------------------------------------------|
| Super | ~83 GB   | 61.6 GB GPU process memory (nvidia-smi, same on all 4 ranks) |
| Omni  | ~15 GB   | 10.7 GB GPU process memory (same on all 4 ranks)             |

`free -g` (total/used/free/shared/buff-cache/available): spark1 121/107/4/15/26/14, spark2 121/104/3/15/29/17, spark3 121/104/5/15/27/17, spark4 121/104/5/15/27/16. The free-after-capture figures in the table above are not additive, because both processes share the unified memory pool.

### Smoke tests

- Super chat: 17*23 gives 391, `reasoning_content` populated.
- Omni text: 391, `reasoning_content` populated.
- Omni image (256x256 PNG, red circle centre, blue rectangle top right): finish stop, "red circle ... blue square positioned in the upper-right". Blue shape right, but the circle was called "lower-left central" instead of centre (minor position error).
- LiteLLM from inside the litellm pod via `http://localhost:4000/v1`: `vision-default` and `hermes-default` both answer 391. No 404.

### Throughput (gsm8k_chat, n=8, temperature 0, max_tokens 8192)

| run            | questions | correct | peak tok/s | agg tok/s | p50 s | wall s |
|----------------|-----------|---------|------------|-----------|-------|--------|
| Omni alone     | 16        | 15      | 516.4      | 399.7     | 6.7   | 19.5   |
| Super alone    | 32        | 32      | 289.4      | 219.8     | 11.0  | 55.8   |
| Omni parallel  | 16        | 15      | 189.2      | 134.3     | 21.7  | 59.4   |
| Super parallel | 32        | 31      | 293.8      | 130.8     | 16.0  | 86.9   |

Standalone references: Omni n8 507.3, Super n8 277.4. Alone, both are at or above the references. In parallel, Omni peak drops to 37 % of standalone (time-sliced GPU, Super gets the bandwidth); Super keeps its peak but its agg falls to 60 % because the two phases only partly overlap (Omni finished at 59 s, Super at 87 s). The one wrong answer (id 12) is the same question in all three runs that contained it.

### Anomalies

- Super's KV pool is 10.18M tokens, not the 16.78M cap: with both started concurrently the 0.75 fraction auto-fit bound before the cap (the profile note expected the cap to bind). Capacity is still 19 x 524288-token requests.
- Weights "mem usage" (29.57 GB Omni, 31.47 GB Super) is inflated by the other pod loading at the same time on the shared pool; the nvidia-smi totals above are the reliable numbers.
- Transient Multus events during rollover: none checked beyond pods reaching Ready.

### Conclusion

Co-location works: both models serve on the same four GPUs with 0 restarts, ~72 GB GPU memory per rank in total and 14 to 17 GB available per node. Throughput under simultaneous load is clearly reduced for Omni. Decision on Super's cap (it did not bind) is left to the user.

Super cap is now 9437184 (was 16777216); it bound (KV exactly 9,437,184 tokens, Super GPU 59,959 MiB per rank, node available 16 to 19 GB).
The vision pods survived the Super-only rollout (same pod names, 0 restarts) and still answered a chat completion afterwards.
