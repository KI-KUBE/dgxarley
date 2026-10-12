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

Method: one container start per case, bench via `sglang-gsm8k` from the control node against `<host>:30000` (chat API, GSM8K test split, temperature 0, max_tokens 8192), levels n=1/4/8/16 with 4/8/16/32 questions (first N, identical across cases), 20 s pause between levels. n=1 tok/s = single-stream rate; n>1 throughput of this sweep is not listed (`concurrent_peak_tok_s` overstates full load), see "Full-load rerun" below. Accept len = mean of `accept len` over all `Decode batch` lines of the case.

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

- Speculation is mandatory for this model on GB10: without it a single stream gets 12.6 tok/s (27B dense, bandwidth-bound). DSpark gamma 7 is x3.8 at n=1; at full load (rerun below, float32 SSM) x2.6 at n=4, x2.2 at n=8, x1.7 at n=16 (15 vs 44 slots), with bfloat16 SSM x2.9 / x2.5 / x2.4 / x2.1 at n=4/8/16/32.
- MTP works on 0.5.21 (the 0.5.17 NEXTN `fast_prefill_plan` crash is gone; driven as EAGLE here). It doubles throughput, but DSpark gamma 7 beats the best MTP setting (M3) by x1.9 at n=1 (MTP not rerun at full load). MTP 5 steps is no better than 3 (higher accept len eaten by sequential draft steps).
- DSpark gamma scales monotonically 3 -> 5 -> 7 at n=1 (31.5 -> 41.3 -> 48.3; gamma 3/5 not rerun at full load); 7 is the RadixArk draft's block size.
- RedHat vs RadixArk at gamma 7: same class (n=1 49-52 vs 48, accept len 4.43-4.77 vs 4.31; full load R0 vs D7 within 7 % at n=4..32 in the rerun). R0 is marginally ahead; not enough to justify a converted local draft in the cluster (would need the converted config hosted somewhere the hf-preload can fetch).
- Concurrency cap: with float32 SSM state and extra_buffer, gamma 7 caps `max_running_requests` at 15 (`max_mamba_cache_size=76, 5 state slots per request`), so the n=16 level of D7/R1/R0 queued one request. bfloat16 SSM state lifts the cap to 30-32 (rerun below) and full load at n=32 from 279 to 499-512 tok/s; it is the preset default since 2026-10-11.
- Open: FP8 KV scales defaulting to 1.0 on the target (see Matrix notes).
- **No profile change**: the profile already runs DSpark with `RadixArk/Qwen3.8-27B-DSpark`, num_draft_tokens 8 (gamma 7), the winner among practical options. The profile header's "NEXTN is not a working fallback" note is stale for 0.5.21 (MTP via EAGLE works, but is slower than DSpark).

## Full-load rerun with the streaming harness (2026-10-11)

All n>1 levels re-measured on spark5 (`scripts/run_standalone.sh PRESET=qwen38_27b` plus the listed overrides) with `sglang-gsm8k run` (streams by default) and `analyze` -> `full_load_tok_s` (token rate while >= 90 % of the client slots are busy). The old `concurrent_peak_tok_s` was 1.0-1.2x full_load with 30-44 slots and 1.6-1.8x at n32 with 15 slots, so no n>1 peak values are listed in this log. Levels n4/n8/n16/n32 with 8/16/32/64 questions, 20 s pause, max-tokens 8192, temperature 0; n1 not rerun (exact at n=1). MTP was not rerun.

full_load tok/s:

| config                                                | slots | bench (UTC) | n4    | n8    | n16   | n32   | engine n32 (samples) | accept len | queue max |
|-------------------------------------------------------|------:|-------------|-------|-------|-------|-------|---------------------:|-----------:|----------:|
| none, float32 SSM, cache 222                          | 44    | 15:10-15:30 | 49.3  | 94.4  | 157.2 | 245.7 | 222 (12)             | -          | 0         |
| DSpark g7 RadixArk, float32 SSM, cache 77             | 15    | 14:37-14:49 | 128.4 | 205.4 | 261.9 | 279.1 | 210 (4)              | 4.32       | 19        |
| DSpark g7 RedHat R0, float32 SSM, cache 76            | 15    | 14:55-15:05 | 120.3 | 191.6 | 267.3 | 280.6 | 237 (6)              | 4.25       | 18        |
| DSpark g7 RadixArk, bfloat16 SSM, cache 152           | 30    | 14:22-14:33 | 143.1 | 239.8 | 372.9 | 499.0 | 343 (3)              | 4.38       | 5         |
| DSpark g7, bfloat16 SSM, cache 160, CG bs 32 (preset) | 32    | 13:24-13:34 | 145.1 | 240.4 | 378.2 | 512.1 | 317 (2)              | 4.28       | 2         |

Best 10 s window / engine max, tok/s. Best 10 s window = highest 10 s streamed token rate of the level (character-weighted timeline; the token-exact timeline via `continuous_usage_stats` came after these runs). Engine max = highest head-log `gen throughput` in the level's time window, logged decode batches in parentheses.

| config                                                | n4            | n8             | n16             | n32             |
|-------------------------------------------------------|---------------|----------------|-----------------|-----------------|
| none, float32 SSM, cache 222                          | 57 / 50 (27)  | 106 / 93 (22)  | 194 / 162 (209) | 314 / 257 (103) |
| DSpark g7 RadixArk, float32 SSM, cache 77             | 137 / 127 (5) | 219 / 209 (21) | 286 / 278 (45)  | 308 / 289 (51)  |
| DSpark g7 RedHat R0, float32 SSM, cache 76            | 132 / 114 (4) | 208 / 197 (31) | 310 / 277 (15)  | 320 / 279 (45)  |
| DSpark g7 RadixArk, bfloat16 SSM, cache 152           | 150 / 142 (4) | 265 / 253 (10) | 408 / 390 (48)  | 575 / 515 (35)  |
| DSpark g7, bfloat16 SSM, cache 160, CG bs 32 (preset) | 160 / 157 (4) | 254 / 239 (14) | 405 / 386 (34)  | 587 / 501 (40)  |

- Overrides: none `SPEC=0 MAMBA_SSM_DTYPE=float32 MAMBA_CACHE=222 MAX_RUNNING=16`; float32 DSpark `MAMBA_SSM_DTYPE=float32 MAMBA_CACHE=77 MAX_RUNNING=16` (R0: `MAMBA_CACHE=76 DRAFT_MODEL=/root/.cache/huggingface/converted/redhat-q38-27b-dspark-t0`); cache 152 `MAMBA_CACHE=152 MAX_RUNNING=16`; preset with no overrides. SGLang raises `max_running_requests` to the mamba-cache fit (slots column).
- Eager decode batches: none 16 (44 slots, CUDA graphs up to 16, so its n32 is a lower bound), cache 152 3, the rest 0.
- GSM8K (n4/n8/n16/n32): 8/8 everywhere; n8 14-16/16; n16 30-31/32 (none and cache 152 +1 truncation); n32 61-62/64 (float32 RadixArk +2 truncations).
- Engine n32 = mean of the decode batches at >= 90 % of min(32, slots) running; the server logs a decode batch only every 40 steps, so the DSpark rows have 2-6 samples and the engine column is a rough check only (none, 12 samples: full_load 11 % above).
- bfloat16 SSM (preset) vs float32 RadixArk: +13 % n4, +17 % n8, +44 % n16, +83 % n32. RadixArk vs RedHat R0 draft (both float32): within 7 % at every level. DSpark g7 preset vs none: x2.9 n4, x2.5 n8, x2.4 n16, x2.1 n32.
- Comparison with the cluster and the other models: `SINGLE_CLUSTER_COMPARE.md`.
