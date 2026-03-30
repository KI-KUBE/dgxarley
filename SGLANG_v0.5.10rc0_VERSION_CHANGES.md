# SGLang v0.5.10rc0 Changes (vs. 0.5.9-dev2-acab24a7)

Changes between our current image `0.5.9-dev2-acab24a7` (2026-03-11) and release
`v0.5.10rc0` (tag `1115dbf2`, 2026-03-28). 462 commits.

Previous delta doc: `SGLANG_DEV2_VERSION_CHANGES.md` (v0.5.9 → acab24a7, 656 commits).

## Breaking Changes

**Piecewise CUDA Graph now enabled by default** (PR #16331)
Previously opt-in (`--enable-piecewise-cuda-graph`). Now the default execution mode for all
models. Reduces memory overhead and improves throughput for models with complex control flow
(MoE, hybrid Mamba/GDN). Multiple follow-up fixes landed in this window:
- #20441 — crash with `--enable-mixed-chunk`
- #21452 — `qo_indptr` correctness bug
- #21565 — guard crash when model has no layers
- #20747 — Kimi-K2.5 compatibility

If instability is observed, --disable-cuda-graph disables both standard and piecewise mode.

**`sgl-kernel` renamed to `sglang-kernel`** (PR #20440)
Package renamed from `sgl-kernel` (0.3.21) to `sglang-kernel` (0.4.0). Any pip dependency or
CI script referencing `sgl-kernel` by name will break on upgrade. Affects us only when
`scitrera` rebuilds the image.

**transformers 4.57.1 → 5.3.0** (PR #17784)
Major version jump. GLM-5 now runs on main. May affect tokenizer behavior or model loading for
models that test against transformers 4.x APIs. The Qwen3/3.5 models we run are compatible.

**FlashInfer 0.6.3 → 0.6.6**
New mxfp8 GEMM and MoE kernels. Required by the new `sglang-kernel` package.

**Flash Attention 3 → 4**
Official Flash Attention 4 integration.

**xgrammar 0.1.25 → 0.1.32**
Structured output / JSON grammar engine update.

## Critical Fixes for Our Deployment

### Subprocess Liveness Monitor (PR #18582, merged 2026-03-29)

A `SubprocessWatchdog` daemon thread polls scheduler subprocesses every 1 second. When a
scheduler crashes at the C++ level (e.g., NCCL timeout triggering `std::terminate()`), the
main process previously became a "zombie service" — accepting TCP connections but unable to
process them. The watchdog now sends `SIGQUIT` to trigger proper cleanup.

**Directly addresses the stale worker scenario described in CLAUDE.md**: when the head's
NCCL connection breaks, the worker stays in `Running 1/1` indefinitely. With the watchdog,
the C++-level crash on the head triggers self-termination, kubelet restarts the head, and
the worker's livenessProbe eventually detects the broken NCCL pipe.

Design:
- Multi-node: non-zero rank nodes get `None` watchdog (only rank 0 monitors)
- Ray backend: only monitors detokenizer, not actor schedulers
- Normal exit (exitcode=0) does NOT trigger SIGQUIT

### Scheduler Launch Hang on Rank Death (PR #20287, merged 2026-03-29)

Fixes the exact head/worker deadlock pattern from CLAUDE.md: when a non-current rank
(worker) dies during `Init torch distributed`, the surviving rank (head) previously hung
indefinitely at the `barrier()` call. Now detected and handled.

### Revert Early HTTP Port Reservation (PR #20468, merged 2026-03-12)

Reverts both the prebinding feature (#17754) and our snapshot's fix (#19805). The early
`listen()` during initialization — which caused K8s startup/liveness probes to hang instead
of getting 503 — is gone. Back to uvicorn's native bind behavior.

**Note**: This feature is present in our current `acab24a7` image (the fix #19805 is in our
snapshot). In v0.5.10rc0, both the feature and its fix are reverted.

**EADDRINUSE sidecar workaround still needed**: The prebinding removal alone does NOT fix
the EADDRINUSE bug. The Scheduler subprocess still binds `<pod-ip>:<port>` for internal
communication, so uvicorn's `0.0.0.0:<same-port>` still conflicts. Confirmed on v0.5.10rc0:
`[Errno 98] error while attempting to bind on address ('0.0.0.0', 8000): address already in use`.
The HAProxy sidecar (`haproxy:lts-alpine`, `0.0.0.0:8000` → `127.0.0.1:30080`) remains in
the deployment.

### MoE Triton Tuning text_config Fix (PR #20232, merged 2026-03-27)

The `get_model_config()` crash documented in `SGLANG_MOE_TUNE_UPSTREAM_BUG.md` is fixed
upstream. Our monkey-patch in `sglang_tune_moe.sh` will gracefully skip (target string no
longer found) when running against an image that includes this fix.

## NCCL / Multi-Node Improvements

### NCCL Pre-Warming (PR #20477, merged 2026-03-17)

A daemon warms up NCCL communicators during startup with a single `all_reduce`. Eliminates
cold-start P99 TTFT spike on the first few requests after pod startup. Measured: 74.9%
improvement on AMD (1413ms → 357ms).

**Default: disabled on NVIDIA** (explicit `--pre-warm-nccl` flag required). Enabled by
default on AMD/ROCm. Triggers when `tp_size > 1 OR pp_size > 1 OR moe_ep_size > 1`.

For our TP=2 over QSFP setup, adding `--pre-warm-nccl` to `sglang_launch.sh` is a no-cost
improvement.

### dist_init_method IPv6/Loopback Fixes (PRs #20491, #20657, #20306, #20643)

A series of fixes for dual-stack IPv6 socket handling, loopback fallbacks, and a new
`NetworkAddress` abstraction. PR #20657 specifically fixes `dist_init_method` for multi-node
setups. Not directly relevant today (we use IPv4 on QSFP), but prevents future surprises.

### RayEngine Multi-Node Co-location (PR #20722, merged 2026-03-19)

Co-locates rank0 scheduler with Engine + fixes CUDA device selection in multi-node Ray setups.

## Qwen3 / Qwen3.5 Fixes (after acab24a7)

**GDN attention state layout** (#20283, merged 2026-03-12)
`[N,HV,K,V]` → `[N,HV,V,K]`. Merged the day after our snapshot — not in acab24a7.

**Fused GDN projection Triton kernel** (#21019, merged 2026-03-23)
New Triton kernel for GDN projection. Performance improvement for Qwen3.5.

**Qwen3 RoPE parameter compatibility** (#20931, merged 2026-03-20)
Fixes `rope_scaling` parameter handling for Qwen3 models.

**Context Parallelism for Qwen3 MoE prefill** (#18233, merged 2026-03-22)
Distributes long sequences across GPUs during prefill. Reduces per-GPU memory pressure for
long-context MoE inference.

**GDN packed decode** (#20627)
Packed decode for GDN attention layers.

**CuTeDSL KDA decode kernel** (#21203)
CUTLASS DSL-based kernel for key-value delta attention decode.

**Fix broken PP layer splitting** (#21070, merged 2026-03-21)
`PPMissingLayer` placeholder was broken for Qwen3.5.

**Mamba slice fix for Prefill TP ≠ Decode TP** (#20655, merged 2026-03-17)
Fixes a crash when prefill and decode use different TP sizes with Mamba/GDN hybrid models.

## Speculative Decoding Changes

**Reference-based speculative decoding refactor** (#20393, merged 2026-03-22)
Major refactoring of the speculative decoding code path.

**Fix spec v1 `token_ids_logprobs`** (#20718)
Correctness fix for log probabilities in speculative v1.

**Fix synchronization issues in `Ngram.cpp`** (#21186, merged 2026-03-23)
Thread safety fix in the ngram speculation engine.

**Fix FA3 SWA spec `pg_size > 1`** (#20369, merged 2026-03-12)
Flash Attention 3 + Sliding Window Attention + speculative decoding fix.

**Fix Kimi K2.5 DP attention + spec decoding launch crash** (#21391)
DP attention combined with speculative decoding crashed on startup for certain models.

## Performance Improvements

**Context Parallelism for prefill** (#18233, merged 2026-03-22)
Distributes long-context prefill across GPUs. Relevant for our 2-GPU setup with long prompts.

**LoRA for MoE expert layers** (#19710, #14105, #21439)
LoRA can now target expert layers in DeepSeek-style MoE models, with fused Triton kernels
and TP support. Not directly used by us, but broadens the fine-tuning options.

**FlashInfer mxfp8 GEMM/MoE kernels** (via FlashInfer 0.6.6)
Microscaling FP8 precision for GEMM and MoE operations.

## New Features

**Elastic EP for partial failure tolerance** (#19248, #17374, #12068)
When a GPU fails, expert weights are redistributed and serving continues without full restart.
Not applicable to our 2-node setup, but signals MoE infrastructure maturity.

**Reasoning effort `none` option** (#20556, merged 2026-03-16)
Adds `reasoning_effort: "none"` to `ChatCompletionRequest`. Useful for disabling thinking
on reasoning models without changing the model configuration.

## Reasoning Parser Changes

- #20284 — Nemotron reasoning parser fix (merged 2026-03-16)
- #20556 — `reasoning_effort: "none"` option (merged 2026-03-16)
- #19552 — Kimi-K2/K2.5 function call and reasoning detection (merged 2026-03-19)

## Security Fixes

**CVE-2026-3989** (PR #20904)
`pickle.loads` → `SafeUnpickler` in `replay_request_dump.py`. Prevents arbitrary code
execution via crafted pickle payloads in request replay.

**CVE-2026-3059 / CVE-2026-3060** (PR #21435)
ZMQ sockets for multimodal generation broker and encoder parallel disaggregation were bound
to `0.0.0.0`, allowing unauthenticated remote access. Now bound to `127.0.0.1`.

## Other Notable Fixes

- **EPLB + Kimi K2.5** (#21004) — rebalance support
- **VRAM leak in overlap scheduling with structured output** (#20697)
- **Chunked prefill + KV cache leaks for streaming** (#20476, merged 2026-03-13)
- **Streaming session with paged KV cache (SWA/MLA)** (#20070)
- **UnboundLocalError when DetokenizerManager constructor fails** (#21471, merged 2026-03-26)
- **PD disagg decode infinite loop when prefill server offline** (#20371) — already in acab24a7
- **Decode throughput metric fix** (#19984) — already in acab24a7

## Dependency Summary

| Package | acab24a7 | v0.5.10rc0 |
|---------|----------|------------|
| sgl-kernel | 0.3.21 | sglang-kernel 0.4.0 (renamed) |
| FlashInfer | 0.6.3 | 0.6.6 |
| transformers | 4.57.1 | 5.3.0 |
| xgrammar | 0.1.25 | 0.1.32 |
| Flash Attention | 3.x | 4.x |
| diffusers | 0.36.0 | 0.37.0 |
| mooncake-transfer-engine | 0.3.9 | 0.3.10 |

## Status of Our Known Bugs

| Bug | Doc | Status in v0.5.10rc0 |
|-----|-----|---------------------|
| reasoning_tokens always 0 | `SGLANG_REASONING_TOKENS_UPSTREAM_BUG.md` | **NOT FIXED** — PR #15562 still open |
| moe_wna16 qzeros + EP | `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md` | **NOT FIXED** — monkey-patch still needed |
| EPLB + Qwen3 | `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md` | **NOT FIXED** — PR #21461 still open |
| NVFP4 input_scale + EP | `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md` | **NOT FIXED** — PRs #20869/#21630 still open |
| ModelOptModelLoader + sharded_state | `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md` | **NOT FIXED** — no fix PR filed |
| sharded_state + speculative | `SGLANG_SHARDED_SPECULATIVE_UPSTREAM_BUG.md` | **NOT FIXED** — not reported upstream |
| MoE Triton tuning text_config | `SGLANG_MOE_TUNE_UPSTREAM_BUG.md` | **FIXED** — PR #20232 merged 2026-03-27 |

## Upgrade Action Items

1. ~~**Update `sglang_image`**~~ — done (`scitrera/dgx-spark-sglang:0.5.10rc0`)
2. ~~**Update `SGLANG_EXPECTED_IMAGE`**~~ — done in `sglang_launch.sh` and `sglang_shard_launch.sh`
3. **Add `--pre-warm-nccl`** to launch args — eliminates cold-start TTFT spike
4. ~~**CUDA graph OOM**~~ — MiniMax-M2.5-NVFP4 profile: `disable_cuda_graph: true` (weights
   now ~72 GB, only ~21 GB free after KV — standard capture OOMs). Other models keep default.
5. **HAProxy sidecar still needed** — EADDRINUSE NOT fixed by PR #20468 alone (Scheduler
   still binds `<pod-ip>:<port>`). Sidecar remains.
6. ~~**Startup probe removed**~~ — replaced with `livenessProbe.initialDelaySeconds: 1800`
   (same pattern as worker). No more fixed time budget for startup.
7. ~~**tqdm progress visibility**~~ — `TQDM_POSITION: "-1"` in ConfigMap forces newlines.
8. **MoE tuning patch** (`sglang_tune_moe.sh`) — will auto-skip, can be removed after
   confirming the fix is in the image
9. **Remaining monkey-patches** (moe_wna16, modelopt_quant, sharded_state loader) — still
   needed, verify grep guards still match the new code
10. **transformers 5.3.0** — verify tokenizer behavior for our model profiles hasn't changed
