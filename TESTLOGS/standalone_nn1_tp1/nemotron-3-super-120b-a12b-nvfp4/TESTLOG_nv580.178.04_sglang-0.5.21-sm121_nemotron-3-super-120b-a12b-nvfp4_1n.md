# SGLang Test Log: NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 + MTPv2 draft head, 1 Node standalone (DGX Spark), TP=1, v0.5.21-sm121

## Environment

| Component | Value                                                                                                                       |
|-----------|-----------------------------------------------------------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121), 128 GB unified, driver 580.178.04, kernel 7.0.0-1019-nvidia                                            |
| Node      | spark5, 1x DGX Spark (standalone podman host, not a cluster node)                                                           |
| Image     | `xomoxcc/dgx-spark-sglang:0.5.21-sm121`                                                                                     |
| Model     | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (`NemotronHForCausalLM`, Mamba2 + MoE + attention hybrid, reasoning model) |
| Draft     | `nvidia/Nemotron-3-Super-120B-A12B-BF16-MTPv2` (EAGLE, `--speculative-draft-model-quantization unquant`)                    |
| Launcher  | `scripts/run_standalone.sh` `PRESET=nemotron3_super`, server in the foreground of a tmux session                            |
| Parallel  | tp=1, pp=1, ep=1                                                                                                            |
| HF cache  | local NVMe (`/root/.cache/huggingface`)                                                                                     |

Cluster counterpart (TP4/EP4, same image and draft): `TESTLOGS/sglang_nn4_tp4_ep1/nemotron-3-super-120b-a12b-nvfp4/TESTLOG_sglang-0.5.21-sm121_nemotron-3-super-120b-a12b-nvfp4-mtpv2_4n.md`.

Preset args: `--trust-remote-code --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer --moe-runner-backend flashinfer_cutlass --fp4-gemm-backend flashinfer_cutlass --cuda-graph-backend-prefill disabled --mem-fraction-static 0.90 --context-length 262144 --max-running-requests 32 --cuda-graph-max-bs-decode 32 --max-mamba-cache-size 128 --mamba-ssm-dtype bfloat16 --reasoning-parser nemotron_3 --tool-call-parser qwen3_coder --speculative-algorithm EAGLE --speculative-draft-model-path nvidia/Nemotron-3-Super-120B-A12B-BF16-MTPv2 --speculative-draft-model-quantization unquant --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5`, env `SGLANG_ENABLE_JIT_DEEPGEMM=false`.

## Mamba cache and concurrency

With MTP 4/1/5 each running request holds 5 mamba state slots, and the state cache comes out of the same `mem_fraction_static` pool as the KV cache.
- Auto-sized cache (float32 SSM, 24 slots): `max_running_requests` capped at 4.
- float32 SSM, `--max-mamba-cache-size 64`: capped at 12; KV fp8 728k tokens.
- bfloat16 SSM, `--max-mamba-cache-size 128` (same cache memory as float32 x 64, now the preset default): capped at 25; KV fp8 668k-693k tokens. Mamba state: ssm 10.08 GB + intermediate ssm 10.16 GB + conv 0.44 GB; 8.0 GB free after CUDA graph capture.

## Run 1: single-stream and accuracy (2026-10-11)

Levels n=1/4/8/16/32 with 4/8/16/32/64 questions (later reruns: 4/16/32/64/128), max_tokens 8192, temperature 0, harness `sglang-gsm8k`.

| config                             | n=1 tok/s | GSM8K (n1/n4/n8/n16/n32)                          | accept len |
|------------------------------------|----------:|---------------------------------------------------|-----------:|
| float32 SSM, cache 64 (12 slots)   |      32.1 | 4/4, 16/16, 32/32, 64/64, 126/128                 |    4.0-4.2 |
| bfloat16 SSM, cache 128 (25 slots) |      32.2 | 4/4, 15/16, 31/32, 64/64, 125/128 (+1 truncation) |    4.0-4.3 |

- bfloat16 SSM does not hurt MTP: accept len is the same at every batch size (cluster: 4.0-4.1). The n=4/n=8 miss is id 12, as in every other model's run.
- 0 eager decode batches, no char runs.
- The n>1 throughput of this run came from the client-side `concurrent_peak_tok_s`, which overstates full load; it is not listed. Throughput at n>1: Run 2.

## Run 2: engine-log rerun (2026-10-12)

Re-run on spark5 with `--decode-log-interval 10` added. Levels n4/n8/n16/n32 with 8/16/32/64 questions, 20 s pause, max-tokens 8192, temperature 0, harness `sglang-gsm8k` (token-exact streaming via `continuous_usage_stats`). Throughput comes from the server log (`Decode batch` lines every 10 decode steps): each cell is the highest `gen throughput` (tok/s) in the level's time window. SGLang sums every generated token between two log lines and divides by the elapsed wall time, so each line is a measured rate over a few seconds.

engine max tok/s:

| config                                          | slots | bench (UTC) | n4 | n8  | n16 | n32 | accept len |
|-------------------------------------------------|------:|-------------|----|-----|-----|-----|-----------:|
| MTPv2 4/1/5, bfloat16 SSM, cache 128 (= preset) |    25 | 13:04-13:12 | 81 | 119 | 178 | 236 |       4.03 |
| MTPv2 4/1/5, float32 SSM, cache 64              |    12 | 16:43-16:53 | 78 | 107 | 127 | 133 |       4.08 |

- Overrides: bfloat16 = preset as is; float32 `MAMBA_SSM_DTYPE=float32 MAMBA_CACHE=64`.
- bfloat16: GSM8K 8/8, 16/16, 32/32, 62/64; accept len mean 4.03 over 131 logged decode batches; `max_total_num_tokens` 668012.
- float32: GSM8K 8/8, 16/16, 31/32, 64/64; accept len mean 4.08 over 169 logged decode batches; `max_total_num_tokens` 735558.
- 0 decode batches with `cuda graph: False` in both.
- bfloat16 over float32 (25 vs 12 slots): +3 % n4, +10 % n8, +41 % n16, +78 % n32. bfloat16 does not change the accept length.
- Comparison with the cluster and the other models: `SINGLE_CLUSTER_COMPARE.md`.
- Cluster rerun 2026-10-19 (dual-rail, 4 NCCL channels, 4n MTPv2 log Run 9): the TP=4 profile is now x2.5-3.0 of this node at every level (n32 704 vs 236; before x2.0-2.1 at n16/n32).
