# SGLang Test Log: Nemotron-3.5-Lightning-30B-A3B-NVFP4, 1 Node standalone (DGX Spark), TP=1, v0.5.21-sm121 (first contact + speculative sweep)

## Environment

| Component | Value                                                                                   |
|-----------|-----------------------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121), 128 GB unified, driver 580.178.04                                  |
| Node      | 1x DGX Spark (standalone podman host, not a cluster node)                                         |
| Image     | `xomoxcc/dgx-spark-sglang:0.5.21-sm121` (digest `sha256:4ab12e77ff98…`)                 |
| Model     | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (snapshot `bee7596`)               |
| Draft     | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` (Qwen3DSparkModel, 967M)    |
| Launcher  | `scripts/run_standalone.sh` `PRESET=nemotron35_lightning`, patches `vroomfondel/dgxarley@e9bbf67` (47) |
| Parallel  | tp=1, pp=1, ep=1                                                                        |
| HF cache  | local NVMe (`/root/.cache/huggingface`), no network filesystem                           |
| Date      | 2026-10-10                                                                              |

Profile: `roles/k8s_dgx/model_profiles/nvidia-nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4.yml` (the cluster profile runs tp_size 4; this log is single-GPU only).
Predecessor: none (first contact).

Preset args (all cases): `--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.78 --context-length 262144 --mamba-ssm-dtype float16 --mamba-radix-cache-strategy extra_buffer --reasoning-parser nemotron_3 --tool-call-parser qwen3_coder`, env `SGLANG_ENABLE_JIT_DEEPGEMM=false`. No `--quantization` (auto -> modelopt mixed), no `--attention-backend` (Nemotron-H resolver), no `--moe-runner-backend`.

Method: one container start per case, bench via `sglang-gsm8k` from the control node against `<host>:30000` (chat API, GSM8K test split, temperature 0, max_tokens 8192), levels n=1/4/8/16 with 4/8/16/32 questions (the first N questions, identical across cases), 20 s pause between levels. Peak = harness `concurrent_peak_tok_s`. Accept len = mean of the `accept len` field over all `Decode batch` log lines of the case.

## Boot

- Boot to "fired up" ~6 min (load_weight 115-133 s from local NVMe page cache, the rest is scheduler init + autotune + graph capture). First boot additionally pulled the image and ~21 GB of weights.
- Checkpoint detected as mixed: "Detected ModelOpt FP8 checkpoint" + "Detected nvfp4 checkpoint" (W4A16_NVFP4 experts/lm_head, FP8 mamba in/out_proj).
- DSpark: `Initialized DSpark draft runner. attention_backend=trtllm_mha, model=Qwen3DSparkModel, gamma=<g>, verify_num_draft_tokens=<g+1>, markov_head=Nemotron35VanillaMarkov`. The `trtllm_mha` draft backend (required by the draft's attention sinks) works on SM121: `trtllm_mha_backend.py` takes its XQA impl under `is_sm120_supported()` (major 12, includes GB10).
- Expected warning for gamma != 8: `DSpark gamma mismatch: using gamma=<g> ... but draft config block_size=8`.
- Speculation resets `max_running_requests` to 48 (no draft: 256). `max_total_num_tokens` 7.1M (spec) / 11.6M (no draft), ~23 GB GPU memory left after boot.
- `Disable prefill CUDA graph because some layers do not apply Standard GQA` (hybrid arch, informational).

## Matrix (5 cases + first-contact run)

| #  | Spec config                         | cuda-graph bs | n=1 tok/s | n=4 peak | n=8 peak | n=16 peak | GSM8K   | accept len |
|----|-------------------------------------|--------------:|----------:|---------:|---------:|----------:|---------|-----------:|
| 00 | DSpark gamma 3 (cookbook)           | 4             | 142.0     | 335.9    | 461.2    | 650.6     | 60/60   | ~3.3       |
| A  | none                                | 16            | 71.3      | 194.9    | 273.2    | 358.8     | 60/60   | -          |
| B  | DSpark gamma 3                      | 16            | 137.2     | 322.0    | 445.7    | 687.6     | 59/60 ¹ | 3.27       |
| C  | DSpark gamma 5                      | 16            | 154.6     | 345.7    | 493.9    | 710.5     | 60/60   | 4.32       |
| D  | **DSpark gamma 7**                  | 16            | **179.7** | **355.1**| **495.4**| **755.5** | 59/60 ² | **4.73**   |
| E  | EAGLE, built-in MTP head (5/1/6)    | 16            | 114.7     | 258.0    | 374.2    | 512.8     | 60/60   | 4.34       |

¹ id 12 truncated (finish `length` at 8192 tokens, empty content).
² id 12 wrong (gt 13, answered 12, finish `stop`): the same off-by-one reading of the question as the Nano-Omni sibling (see its TESTLOG).

Aggregate tok/s (total_tokens / wall), n=1/4/8/16: A 71/107/139/225, B 134/192/259/262, C 147/230/247/448, D 145/235/289/293, E 112/171/203/339. Aggregate swings with the per-level tail (one long answer dominates the wall), so peak is the comparison metric.

## Quality checks

| Check                           | All cases                                         |
|---------------------------------|---------------------------------------------------|
| GSM8K correct                   | 298/300 (both misses are id 12, see above)        |
| Same-char runs (harness)        | none                                              |
| Odd non-ASCII (harness)         | none                                              |
| Truncations                     | 1 (B, id 12)                                      |
| `reasoning_content` populated   | yes (nemotron_3 parser splits cleanly)            |
| Eager decode batches            | 0 in A-E (bs 16); 00 ran n=8/16 partly eager (bs 4) |

Manual smoke test on 00: 4 parallel free-form prompts (code, story, German history, calculus) coherent; the German prompt was answered in English (model behaviour, not setup). float16 SSM state with speculation showed no corruption at this sample size (the qwen38 preset forces float32 for its own spec path; not needed here).

## Results and decision

- DSpark roughly doubles throughput at every concurrency vs no draft (gamma 3: x1.9 at n=1, x1.9 at n=16).
- Larger gamma pays off on GB10 despite the cookbook's 3: gamma 7 vs 3 is +31 % at n=1, +10 % at n=4/8, +10 % at n=16; accept len rises 3.27 -> 4.32 -> 4.73. gamma 5 -> 7 is flat at n=4/8, so the gain past 5 is mostly single-stream.
- DSpark gamma 7 beats EAGLE on the built-in MTP head by +57 % at n=1 and +47 % at n=16 at similar accept len (EAGLE pays 5 sequential draft steps).
- cuda-graph-max-bs 4 -> 16 is neutral within noise (00 vs B), +6 % at n=16 only.
- **Profile change:** `speculative_dspark_block_size: 3 -> 7`, `speculative_num_draft_tokens: 4 -> 8`. `cuda_graph_max_bs` left at 4 (not shown to matter here; re-check on the TP4 cluster). Standalone preset default `GAMMA` set to 7.
- Untested: gamma 8 (checkpoint's native block_size, num_draft_tokens 9), DFlash draft, TP4 on the cluster, long-context behaviour, >60-question accuracy.
