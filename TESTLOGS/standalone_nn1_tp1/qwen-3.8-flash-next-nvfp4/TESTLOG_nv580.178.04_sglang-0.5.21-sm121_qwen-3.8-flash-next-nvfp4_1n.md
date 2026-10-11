# SGLang Test Log: Qwen3.8-Flash-Next-NVFP4 (nvidia), 1 Node standalone (DGX Spark), TP=1, v0.5.21-sm121 (MTP off / 3 / 2 / adaptive)

## Environment

| Component | Value                                                                   |
|-----------|-------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121), 128 GB unified, driver 580.178.04 (nvidia-smi)     |
| Node      | 1x DGX Spark (standalone podman host, not a cluster node)                         |
| Image     | `xomoxcc/dgx-spark-sglang:0.5.21-sm121`                                 |
| Model     | `nvidia/Qwen3.8-Flash-Next-NVFP4` (`--quantization modelopt_mixed`)     |
| Launcher  | `scripts/run_standalone.sh` `PRESET=qwen38_nvidia` (working-tree version of 2026-10-10, with the `MTP_STEPS` / `ADAPTIVE` knobs) |
| Parallel  | tp=1, pp=1, ep=1                                                        |
| HF cache  | local NVMe (`/root/.cache/huggingface`), no network filesystem           |
| PLE table | 51 GB PLE table offloaded to a file-backed mmap in `/var/tmp/sglang-ple-offload` (`PLE_KEEP=0`), resident set capped at 8 GiB |
| Date      | 2026-10-10                                                              |

Run in parallel with a separate 4-node cluster sweep of the same model; this log is single-GPU only.

Preset knobs used in all cases: `MAX_RUNNING=32` (sets `--max-running-requests 32` and `--cuda-graph-max-bs-decode 32`), `MEM_FRACTION` default `0.87` (`--mem-fraction-static 0.87`). Boot with these values succeeded in all four cases (no OOM, no fallback needed). Preset args: `--quantization modelopt_mixed --fp4-gemm-backend flashinfer_cutlass --mm-attention-backend triton_attn` plus the common qwen38 args; with MTP: `--speculative-algorithm EAGLE --speculative-draft-model-path nvidia/Qwen3.8-Flash-Next-NVFP4 --speculative-eagle-topk 1 --speculative-moe-runner-backend triton`, steps/draft tokens per case below.

| Case | Env                                   | speculative args (server_args)                                    |
|------|---------------------------------------|-------------------------------------------------------------------|
| S0   | `SPEC=0 MAX_RUNNING=32`               | none                                                              |
| S1   | `SPEC=1 MAX_RUNNING=32`               | num_steps 3, topk 1, num_draft_tokens 4                           |
| S2   | `SPEC=1 MTP_STEPS=2 MAX_RUNNING=32`   | num_steps 2, topk 1, num_draft_tokens 3                           |
| S3   | `SPEC=1 ADAPTIVE=1 MAX_RUNNING=32`    | `speculative_adaptive: True`, `speculative_adaptive_config: /launch/adaptive_config.json`, start num_steps 2 / num_draft_tokens 3 |

S3 adaptive config (`adaptive_config.json`): `{"1": {"candidate_steps": [1, 2, 3]}, "8": {"candidate_steps": [1, 2, 3]}, "32": {"candidate_steps": [1, 2]}}`. (An earlier revision of the preset used other tiers; this run used the corrected file above.)

Method: one container start per case, bench via `sglang-gsm8k` from the control node against `<host>:30000` (chat API, GSM8K test split, temperature 0, max_tokens 8192), levels n=1/4/8/16/32 with 4/8/16/32/64 questions (first N, identical across cases), 20 s pause between levels. Peak = harness `concurrent_peak_tok_s` (n=1: single-stream tok/s). Agg = `aggregate_tok_s`. Accept len = mean of `accept len` over all `Decode batch` lines of the case. Probe: 16 requests (4 concurrent), max_tokens 400, temperature 1.0, top_p 0.95, thinking on for even indices, 4 tool-call requests; collapse = `!{4,}` or any char repeated 10+ times.

## Matrix

| #  | Config                  | n=1 tok/s | n=4 peak | n=8 peak | n=16 peak | n=32 peak | GSM8K n=1/4/8/16/32 (correct/N) | accept len | max_running_requests (effective) | probe collapses | probe tool calls parsed |
|----|-------------------------|----------:|---------:|---------:|----------:|----------:|---------------------------------|-----------:|---------------------------------:|----------------:|-----------------------:|
| S0 | MTP off                 | 16.3      | 51.6     | 85.4     | 119.3     | 133.9     | 4/4, 8/8, 15/16, 30/32, 63/64   | -          | 25                               | 0/16            | 4/4                    |
| S1 | MTP 3 / 1 / 4           | 31.7      | 83.6     | 130.8    | 188.1     | 239.7     | 4/4, 8/8, 15/16, 31/32, 62/64   | 3.04       | 9                                | 0/16            | 4/4                    |
| S2 | MTP 2 / 1 / 3           | 32.0      | 79.2     | 119.4    | 175.1     | 225.8     | 4/4, 8/8, 15/16, 30/32, 63/64   | 2.55       | 10                               | 0/16            | 4/4                    |
| S3 | adaptive (start 2 steps)| 31.3      | 82.1     | 128.3    | 165.4     | 246.8     | 4/4, 8/8, 15/16, 31/32, 63/64   | 2.99       | 11                               | 0/16            | 4/4                    |

Aggregate tok/s (total tokens / wall time, n=1/4/8/16/32): S0 15.9 / 39.6 / 27.6 / 31.9 / 44.5; S1 30.2 / 55.4 / 75.2 / 69.3 / 71.1; S2 29.8 / 59.8 / 73.7 / 57.8 / 86.0; S3 29.4 / 61.1 / 60.4 / 69.7 / 112.9. Aggregate is dominated by the longest request at each level (the few lengthy generations) and is not comparable to peak.

GSM8K misses: id 12 (gt 13, answered 12, finish `stop`) in every case at every level n>=8, including S0 without speculation. Truncations (finish `length`, not wrong answers): S1 n=32 id 13; S0 n=16 id 13; S2 n=16 id 2. No harness char-runs and no odd non-ASCII in any case.

Probe finish reasons (stop / length / tool_calls): S0 11/1/4, S1 9/3/4, S2 10/2/4, S3 10/2/4. No request errors.

## Boot findings

- Effective `max_running_requests` is capped far below `MAX_RUNNING=32` by the mamba state cache (`max_mamba_cache_size`, 4 state slots per request): S0 25 (cache 102), S1 9 (38), S2 10 (42), S3 11 (44). The SGLang hint: raise `--mamba-full-memory-ratio` / `--max-mamba-cache-size` or use `--mamba-ssm-dtype bfloat16`. Consequence: n=16 and n=32 levels queue (requests beyond the cap wait), so those peaks are queue-limited batch-of-9..11 numbers. With MTP the speculative intermediate SSM state cache (S1: 4.22 GB) is what lowers the cap vs S0.
- `max_total_num_tokens`: S0 530368, S1 392576, S2 383168, S3 379840.
- Startup timings (scheduler_e2e): S0 687 s, S1 764 s, S2 945 s, S3 796 s; load_weight 623 / 697 / 881 / 711 s. CUDA graph capture: S1 target_verify 6.34 s, draft_decode 9.63 s, draft_extend 0.83 s; S3 target_verify 16.62 s, draft_decode 9.85 s, draft_extend 3.06 s; S0 decode 5.56 s.
- Eager decode batches (`cuda graph: False` on `Decode batch` lines): 0 in all four cases.
- Warnings/errors in the logs: only `tl.make_block_ptr is deprecated`, the `sglang.launch_server` entrypoint notice, a tilelang "Data race detected: Logits(bx, position)" warning during kernel compile (all cases), and `torchcodec` not installed (audio inputs). No Traceback, no OOM.
- QSA kernel: `[patch] QSA SM121 KDA kernel (#36845): shipped by upstream, not overwriting` and `register the SM121 QSA KDA kernel (#36845) already applied, skipping`; the TP4-widening and trtllm-veto patches report `gate not matched, skipping` (not needed at TP1).
- S3 adaptive checks: `--speculative-adaptive` present in server_args (`speculative_adaptive: True`, config path `/launch/adaptive_config.json`); not silently disabled (no adaptive-disabled warning; the only related line is `index_share_for_mtp_iteration is disabled under adaptive speculative decoding`, logged twice). 65 `Adaptive spec params updated` events: steps 2 <-> 3 (draft tokens 3 <-> 4), e.g. `steps 2 -> 3 (ema_accept_len=1.80)`, `steps 3 -> 2 (ema_accept_len=0.95..1.14)`. With the effective cap of 11 only tiers 1 and 8 are reachable (tier 32 never); no step 1 switch seen in the log. No capture crash.

## Results

MTP vs off (peak tok/s): n=1 x1.95 (S1), n=4 x1.62, n=8 x1.53, n=16 x1.58, n=32 x1.79. At n>=16 the cap differs per case (S0 25 vs 9-11 with MTP), so the like-for-like comparison is n<=8.

Among the MTP variants:
- n=1: all equal within noise (31.3-32.0).
- n=4/n=8: S1 (3 steps) >= S3 (adaptive) > S2 (2 steps): 83.6 / 82.1 / 79.2 and 130.8 / 128.3 / 119.4.
- n=16: S1 188.1 > S2 175.1 > S3 165.4.
- n=32: S3 246.8 > S1 239.7 > S2 225.8.
- Differences of a few percent are within what one run per case can resolve; S2 is consistently the lowest up to n=16 and its accept len (2.55) is lowest. Accept len for S3 (2.99) is near S1 (3.04) because the adaptive state mostly sits at 3 steps.
- Accuracy is the same in all cases (id 12 wrong everywhere, i.e. a model/question property); no "!" collapse in any case (0/16 probes, 0 harness char-runs) and tool-call parsing 4/4 in all cases.
- Caveat for the n=16/32 levels: effective concurrency was 9-11 (S1/S2/S3) vs 25 (S0).
