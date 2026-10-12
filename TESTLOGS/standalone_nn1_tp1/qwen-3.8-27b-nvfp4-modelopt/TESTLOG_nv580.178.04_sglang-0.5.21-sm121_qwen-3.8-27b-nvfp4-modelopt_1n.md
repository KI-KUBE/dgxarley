# SGLang Test Log: Qwen3.8-27B-NVFP4-ModelOpt, 1 Node standalone (DGX Spark), TP=1, v0.5.21-sm121 (speculative sweep: none / MTP / DSpark RadixArk / DSpark RedHat)

## Environment

| Component | Value                                                                                   |
|-----------|-----------------------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121), 128 GB unified, driver 580.178.04                                  |
| Node      | 1x DGX Spark (standalone podman host, not a cluster node)                                         |
| Image     | `xomoxcc/dgx-spark-sglang:0.5.21-sm121` (digest `sha256:4ab12e77ff98…`)                 |
| Model     | `vroomfondel/Qwen3.8-27B-NVFP4-ModelOpt` (snapshot `e76f894`, uniform W4A4 NVFP4, FP8 KV) |
| Drafts    | `RadixArk/Qwen3.8-27B-DSpark` (block 7, 5 target layers); `RedHatAI/Qwen3.8-27B-speculator.dspark` (block 8, 8 target layers, config converted, see below); built-in MTP head |
| Launcher  | `scripts/run_standalone.sh` `PRESET=qwen38_27b`, patches `vroomfondel/dgxarley@e9bbf67` |
| Parallel  | tp=1, pp=1, ep=1                                                                        |
| HF cache  | local NVMe (`/root/.cache/huggingface`), no network filesystem                           |
| Date      | 2026-10-10                                                                              |

Profile: `roles/k8s_dgx/model_profiles/vroomfondel-qwen3.8-27b-nvfp4-modelopt.yml` (cluster profile runs tp_size 4; this log is single-GPU only).
Predecessor: none in TESTLOGS (the profile header records a 2026-08-15 single-Spark run on 0.5.17 via the quantizer's serve path: full GSM8K 97.42 %, accept len 3.77, NEXTN crashed).

Preset args (all cases): `--trust-remote-code --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer --fp4-gemm-backend flashinfer_cutlass --cuda-graph-backend-prefill disabled --mem-fraction-static 0.80 --context-length 262144 --max-running-requests 48 --cuda-graph-max-bs-decode 16 --mamba-ssm-dtype float32 --mamba-radix-cache-strategy extra_buffer --reasoning-parser qwen3 --tool-call-parser qwen3_coder`, env `SGLANG_ENABLE_JIT_DEEPGEMM=false SGLANG_RAGGED_VERIFY_MODE=static PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. DSpark adds `--speculative-draft-model-quantization unquant --speculative-draft-attention-backend flashinfer --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-dspark-block-size <g> --speculative-num-draft-tokens <g+1>`; MTP adds `--speculative-algorithm EAGLE --speculative-draft-model-path <target> --speculative-num-steps <s> --speculative-eagle-topk 1 --speculative-num-draft-tokens <s+1>`. (`--disable-piecewise-cuda-graph` no longer exists in 0.5.21; replaced as sglang_launch.sh does.)

Method: one container start per case, bench via `sglang-gsm8k` from the control node against `<host>:30000` (chat API, GSM8K test split, temperature 0, max_tokens 8192), levels n=1/4/8/16 with 4/8/16/32 questions (first N, identical across cases), 20 s pause between levels. n=1 tok/s = single-stream rate; n>1 throughput of this sweep is not listed (`concurrent_peak_tok_s` overstates full load), see "Engine-log rerun" below. Accept len = mean of `accept len` over all `Decode batch` lines of the case.

## Matrix

| #  | Spec config                                  | n=1 tok/s | GSM8K   | accept len (ceiling) | max_running_requests |
|----|----------------------------------------------|----------:|---------|---------------------:|---------------------:|
| A  | none                                         | 12.6      | 58/60   | -                    | 44 |
| M3 | MTP (EAGLE) 3 / 1 / 4 (card config)          | 24.9      | 58/60   | 3.10 (4)             | 22 |
| M5 | MTP (EAGLE) 5 / 1 / 6                        | 24.7      | 57/60 ¹ | 3.62 (6)             | 17 |
| D3 | DSpark RadixArk gamma 3                      | 31.5      | 58/60   | 3.20 (4)             | 22 |
| D5 | DSpark RadixArk gamma 5                      | 41.3      | 57/60 ¹ | 3.81 (6)             | 18 |
| D7 | DSpark RadixArk gamma 7 (card, = profile)    | 48.3      | 58/60   | 4.31 (8)             | 15 |
| R1 | DSpark RedHat gamma 7, target ids aux-1      | **52.5**  | 58/60   | 4.43 (8)             | 15 |
| R0 | DSpark RedHat gamma 7, target ids = aux      | 49.3      | 58/60   | **4.77** (8)         | 15 |

Every miss is GSM8K id 12 (gt 13, answered 12, finish `stop`), present in ALL cases including A (no draft), i.e. a model/question property. ¹ plus one truncation (M5: id 16 at 8192 tokens; D5: one additional miss at n=16).

No eager decode batches in any case. No harness char-runs or odd non-ASCII.

**FP8 KV scales NOT picked up:** `Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0.` appears once in A (target only) and twice with a draft (target + draft). The profile's SM121 review item 7 requires it ZERO times for this checkpoint (baked KV scales, PR #31220 / p51). So on 0.5.21 the target runs FP8 KV with scale 1.0. GSM8K is unaffected at this sample size (58/60, misses identical to no-draft), but it is an open regression against the profile's expectation; check against `--kv-cache-dtype auto` and the p51 gate before the cluster deploy.

## RedHat draft conversion

`RedHatAI/Qwen3.8-27B-speculator.dspark` ships in the vLLM `speculators` format (nested `transformer_layer_config`, `aux_hidden_state_layer_ids`, `auto_map` to a custom config class), which SGLang 0.5.21 does not parse. Tensor names are identical to the RadixArk draft (62 tensors: `fc`, `hidden_norm`, 5 decoder layers, `markov_head.markov_w{1,2}`, `confidence_head.proj`, `norm`), so only `config.json` was rewritten into the RadixArk layout (`dspark_config`/`dflash_config`, top-level `target_layer_ids`, `num_target_layers 64`, `projector_type dspark`): 5 sliding-attention layers (window 2048), 20 q / 4 kv heads x 256, intermediate 17408, block_size 8, markov_rank 256, mask_token_id 248077, `sample_from_anchor true`. Its mrope (`mrope_section [44,44,40]`, full rotary) was replaced by plain rope theta 1e7, equivalent for text-only positions. `fc.weight` [5120, 40960] confirms 8 target layers. Converted dirs (weights hard-linked, 3.98 GB, sha256 `6cf6f33f…`): `<hf-cache>/converted/redhat-q38-27b-dspark-{tm1,t0}`.

Target-layer convention: SGLang's qwen3_5 captures `target_layer_ids[i]` at the INPUT of layer id+1; vLLM captures `aux_hidden_state_layer_ids` at the input of layer id. That predicts aux-1 (R1), but R0 (ids taken 1:1) measured the higher accept len (4.77 vs 4.43). Both are healthy, so the off-by-one is a second-order effect here; R0 is the better mapping by measurement. Small sample (45 vs 80 decode batches), re-check before relying on it.

## Results and decision

- Speculation is mandatory for this model on GB10: without it a single stream gets 12.6 tok/s (27B dense, bandwidth-bound). DSpark gamma 7 is x3.8 at n=1; in the engine-log rerun below (engine max, float32 SSM) x3.1 at n=4, x2.6 at n=8, x2.0 at n=16, x1.25 at n=32 (15 vs 44 slots), with bfloat16 SSM (preset) x3.8 / x3.3 / x2.6 / x2.2.
- MTP works on 0.5.21 (the 0.5.17 NEXTN `fast_prefill_plan` crash is gone; driven as EAGLE here). It doubles throughput, but DSpark gamma 7 beats the best MTP setting (M3) by x1.9 at n=1 (MTP not rerun). MTP 5 steps is no better than 3 (higher accept len eaten by sequential draft steps).
- DSpark gamma scales monotonically 3 -> 5 -> 7 at n=1 (31.5 -> 41.3 -> 48.3; gamma 3/5 not rerun); 7 is the RadixArk draft's block size.
- RedHat vs RadixArk at gamma 7: same class (n=1 49-52 vs 48, accept len 4.43-4.77 vs 4.31; engine max R0 vs D7 within 5 % at n=4..32 in the rerun). R0 is marginally ahead; not enough to justify a converted local draft in the cluster (would need the converted config hosted somewhere the hf-preload can fetch).
- Concurrency cap: with float32 SSM state and extra_buffer, gamma 7 caps `max_running_requests` at 15 (`max_mamba_cache_size=76, 5 state slots per request`), so the n=16 level of D7/R1/R0 queued one request. bfloat16 SSM state lifts the cap to 30-32 (rerun below) and the engine max at n=32 from 322 to 553-558 tok/s; it is the preset default since 2026-10-11.
- Open: FP8 KV scales defaulting to 1.0 on the target (see Matrix notes).
- **No profile change**: the profile already runs DSpark with `RadixArk/Qwen3.8-27B-DSpark`, num_draft_tokens 8 (gamma 7), the winner among practical options. The profile header's "NEXTN is not a working fallback" note is stale for 0.5.21 (MTP via EAGLE works, but is slower than DSpark).

## Engine-log rerun (2026-10-12)

All five configs re-run on spark5 on 2026-10-12 (`scripts/run_standalone.sh PRESET=qwen38_27b` plus the listed overrides, all with `--decode-log-interval 10`, server in the foreground of a tmux session). Levels n4/n8/n16/n32 with 8/16/32/64 questions, 20 s pause, max-tokens 8192, temperature 0, harness `sglang-gsm8k` (token-exact streaming via `continuous_usage_stats`). Throughput comes from the server log (`Decode batch` lines every 10 decode steps): each cell is the highest `gen throughput` (tok/s) in the level's time window. SGLang sums every generated token between two log lines and divides by the elapsed wall time, so each line is a measured rate over a few seconds. These numbers replace the client-side figures of 2026-10-11. MTP and gamma 3/5 were not rerun.

engine max tok/s:

| config                                                | slots | bench (UTC) | n4  | n8  | n16 | n32 | accept len |
|-------------------------------------------------------|------:|-------------|-----|-----|-----|-----|-----------:|
| none, float32 SSM, cache 222                          |    44 | 13:58-14:26 | 50  | 93  | 164 | 257 |          - |
| DSpark g7 RadixArk, float32 SSM, cache 77             |    15 | 13:31-13:39 | 153 | 244 | 329 | 322 |       4.57 |
| DSpark g7 RedHat R0, float32 SSM, cache 76            |    15 | 13:44-13:53 | 152 | 233 | 320 | 319 |       4.49 |
| DSpark g7 RadixArk, bfloat16 SSM, cache 152           |    30 | 13:18-13:26 | 178 | 293 | 439 | 558 |       4.37 |
| DSpark g7, bfloat16 SSM, cache 160, CG bs 32 (preset) |    32 | 12:14-12:27 | 188 | 305 | 424 | 553 |       4.36 |

- Overrides: none `SPEC=0 MAMBA_SSM_DTYPE=float32 MAMBA_CACHE=222 MAX_RUNNING=16`; float32 DSpark `MAMBA_SSM_DTYPE=float32 MAMBA_CACHE=77 MAX_RUNNING=16` (R0: `MAMBA_CACHE=76 DRAFT_MODEL=/root/.cache/huggingface/converted/redhat-q38-27b-dspark-t0`); cache 152 `MAMBA_CACHE=152 MAX_RUNNING=16`; preset with no overrides. SGLang raises `max_running_requests` to the mamba-cache fit (slots column).
- Eager decode batches (`cuda graph: False`): none 61 of 1990 logged (44 slots, CUDA graphs up to 16), cache 152 12 of 301 (graphs up to 16, 30 slots); the rest 0.
- GSM8K (n4/n8/n16/n32): 8/8 everywhere; n8 15/16; n16 31/32; n32 61-63/64 (preset and R0 +1 truncation each).
- bfloat16 SSM (preset) vs float32 RadixArk, engine max: +23 % n4, +25 % n8, +29 % n16, +72 % n32. RadixArk vs RedHat R0 draft (both float32): within 5 % at every level.
- Comparison with the cluster and the other models: 4n log and `SINGLE_CLUSTER_COMPARE.md`.
