# SGLang `0.5.10-20260429-dev1` Changes (vs. `v0.5.10` / `v0.5.10.post1`)

Image: `scitrera/dgx-spark-sglang:0.5.10-20260429-dev1` (built 2026-04-29).
Wraps SGLang `main` cut at commit `2bbd30a` (2026-04-29 23:53 UTC).

Delta basis: `v0.5.10.post1` (tag `2026-04-09`) → `2bbd30a` (`main`, 2026-04-29).
**832 commits, 893 merged PRs in the window**.

Previous delta doc: `SGLANG_v0.5.10_VERSION_CHANGES.md` (rc0 → v0.5.10, 250 commits).

> The "dev1" suffix means **post-release main snapshot**, not a stable point release.
> No `v0.5.11` exists yet on GitHub — only `v0.5.10` and `v0.5.10.post1` are tagged.

---

## Dependency Bumps (the big ones)

| Package              | v0.5.10.post1 | dev1 (`2bbd30a`)                          | Note                                                                      |
|----------------------|---------------|-------------------------------------------|---------------------------------------------------------------------------|
| `cuda-python`        | `==12.9`      | `>=13.0`                                  | **CUDA 13 toolchain** — base image switched to cu130                      |
| `sglang-kernel`      | `0.4.1`       | `0.4.1.post1` (→ `0.4.2` later in window) | wheels now built for both cu129 & cu130                                   |
| `flashinfer_python`  | `0.6.7.post3` | `0.6.8.post1`                             | new TRTLLM-Gen FP4/FP8 routed-MoE; SM103 router-gemm                      |
| `flashinfer_cubin`   | `0.6.7.post3` | `0.6.8.post1`                             | matches                                                                   |
| `transformers`       | `5.3.0`       | **`5.6.0`**                               | two bumps in series (#21569: 5.3→5.5.3, #23525: 5.5.4→5.6.0)              |
| `torchao`            | `0.9.0`       | **`0.17.0`**                              | huge skip; new MX/FP4 paths                                               |
| `nvidia-cutlass-dsl` | `>=4.4.1`     | `==4.4.2`                                 | pinned exact                                                              |
| `flash-attn-4`       | `>=4.0.0b4`   | `>=4.0.0b9`                               | five betas later                                                          |
| `mistral_common`     | `>=1.9.0`     | `>=1.11.0`                                | required by Mistral-Small-4 native config path                            |
| `easydict`           | —             | **added**                                 | `trust_remote_code` models (DeepSeek-OCR) need it under transformers 5.4+ |

**Action item**: any host monkey-patching that imports `transformers` internals will likely
need a re-audit — tokenizer / config layout changed across 5.3 → 5.6.

---

## New Features (User-Visible)

### DFLASH Speculative Decoding (PR #22077, #23553 docs, #22836 graph fix)

New speculative-decoding algorithm from z-lab. Adds:
- `python/sglang/srt/models/dflash.py` — non-causal draft model with mask-token
  block prediction, separate draft KV pool sharing the target's `req_to_token_pool`.
- `python/sglang/srt/speculative/dflash_worker.py` — spec-v1 only worker.
- TP-safe greedy sampling, CUDA-graph-compatible draft buffers.
- AMD ROCm enablement (#22342).

Fix in #22836: aux-hidden-state capture moved BEFORE CUDA graph capture, so EAGLE3
**and** DFLASH no longer collapse to acceptance-length=0 with PCG enabled. Was a
silent regression — would explain "speculative ON but no speedup" reports.

### Gemma 4 (PR #21952)

Full upstream support for the Gemma 4 family (Dense E2B/E4B/31B, MoE 26B-A4B,
multimodal). **Replaces our `xomoxcc/main-gemma4-sm121` ad-hoc build** for BF16
variants going forward — the dev1 image is the first scitrera tag containing
`#21952`.

Status against `SGLANG_GEMMA4_UPSTREAM_BUG.md`:
- BF16 dense + MoE: `#21952` is now in the image ⇒ no longer needs main build.
- NVFP4 SM121: still blocked. The four sm120/121 PRs (#22929, #22928, #22927,
  #22615) had **NOT merged by 2026-04-29**; verify before retiring the bug doc.

### NVFP4 KV Cache (PR #21954, part 1/4)

`FP4KVCacheQuantMethod` strategy ABC + quant/dequant utilities for SM100/SM120.
Pure additions — does not yet wire into runtime. PR2-4 (memory pool, attention
backends, MTP/config) land in subsequent PRs that may or may not be in this cut;
**not yet a runnable feature on our SM121 cluster**.

### DeepSeek V4 cookbook (PRs #23605, #23617, #23622, #23715, #23725, #23737,
#23817, #24104, …)

Reference recipes added for DeepSeek-V4 across H200, B200, GB200, GB300, H100,
MI355x. Many are recipe-only (no runtime change), but the PRs collectively wire
in:
- `swiglu_limit` clamp on `DeepseekV2MLP` to fix "meaningless numbers" in chat
  output (#23776) — applies to DSV4-quantized variants.
- `topk512transform` kernel (#24143).

Not relevant for our spark cluster (insufficient VRAM for DSV4) but signals where
upstream attention is going.

### Other models

- **Xiaomi MiMo-V2.5 / MiMo-V2.5-Pro** day-0 support (#23808, #23811) +
  reasoning parser (#21414 was rc0; this cycle adds full model).
- **Hy3 preview** (#23533).
- **Moss-VL** Python runtime (#23454).
- **Parakeet Nemotron encoder** (#23568).
- **LFM2 MoE tuning configs** for H100/B200/MI325X (#22791).

---

## Critical Bug Fixes

### NCCL AllGather Hang in EAGLE/EAGLE3 + TP>1 + non-greedy (PR #22458)

**Fixes a hang we may have hit silently.** Root cause chain:
identical `next_token_logits` → softmax + top_k/top_p → tiny FP non-determinism
across ranks → different sampled tokens → different prefix matches in radix cache
→ different `extend_seq_lens` → AllGather size mismatch → hang.

Fix: broadcast predicted token IDs from rank 0 after sampling. Applies to **both
EAGLE V1 and V2**, hits any TP>1 spec-decode setup with temperature>0. Filed as
issue #22276.

If we ever see deadlocks on Qwen3-235B / Qwen3.5 with `temperature>0` + EAGLE3 +
TP=4, this is the cause and is now fixed.

### Qwen3.5 FP8 Per-Tensor Weight Loading Crash (PR #23062)

Regression introduced in v0.5.10 by the GatedDeltaNet fused-projection refactor:

```
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 1, but got 3
```

Affects FP8 per-tensor quantized Qwen3.5 dense **and** MoE. v0.5.9 worked,
v0.5.10.post1 crashed. Fixed by broadcasting per-tensor scales in
`_make_packed_weight_loader`.

**Relevant**: if any Qwen3.5 FP8 profile in `roles/k8s_dgx/model_profiles/*.yml`
silently failed on v0.5.10, retest after upgrading.

### Qwen3.5-27B Accuracy Regression (PR #22312)

Qwen3.5-27B on the GDN BA-fallback path (introduced in `5bdc07d9`) returned
non-contiguous split views; existing GDN Triton kernels assumed contiguous
layouts and hardcoded strides → wrong memory reads → **49/50 → 3/50 GSM8K**
collapse.

Fix: GDN kernels now handle non-continuous B/A tensors.

### Mistral Small 4 Startup Crash (already in v0.5.10, unchanged here)

Carried forward.

### LFM2 ShortConv Mamba State Indexing (PR #23975)

Off-by-one in mamba state indexing for LFM2 ShortConv. Affects LFM2 variants only.

### NSA / Mamba State Transfer in Disagg (#23773, #22240)

`nixl` transport now correctly handles SWA / NSA / Mamba state buffers; Mamba
state slice transfer for heterogeneous TP (Qwen3.5 disagg step 2/2).

### CUDA 13.0 `cudaMemcpyBatchAsync` Segfault (#23136)

Specific to CUDA 13.0 (which we now run by default). Worth knowing if any odd
crashes appear in multimodal pipelines — fixed before the dev1 cut.

### FP4 DeepGEMM Path on Blackwell (#23948, #23703)

`#23686` broke FP4 DeepGEMM on Blackwell; `#23948` restored it. `#23703` switches
FP4 packed expert weights to `uint8` to match `kPackedFP4`. Both relevant for
NVFP4 MoE on SM120/SM121.

### Qwen3 MoE Double-Reduce in DP+EP+reduce_scatterv (#23731, #23734, #23732)

Three-PR chain. After `#22642` introduced `reduce_scatterv` for DP attention,
Qwen3 MoE plus several other models still emitted a redundant all-reduce inside
EP, producing `2x` reduced outputs (numerical garbage). Guard added.

**Relevant for our DP attention experiments**: if we have ever enabled
`enable_dp_attention=true` together with `enable_ep_moe=true` on Qwen3 since
#22642 landed, results were silently doubled. Now fixed.

### Health-Probe Subprocess Visibility (PR #23320)

Engine now exposes child-process PIDs so K8s health probes can verify the
scheduler subprocesses are still alive. Combined with the v0.5.10 SubprocessWatchdog
this closes the "zombie service" window further. Still no probe-side change
needed in our manifests, but the data is now available.

---

## Multi-Node / NCCL / Scheduler Improvements

### Reduce GPU Memory for MoE Parallel Groups (PR #22515)

`initialize_model_parallel` previously created MOE_EP and MOE_TP groups with full
pynccl + custom_allreduce stacks (~700 MB each). With `--tp 8 --ep 4` torch
distributed init was using **2.7 GB vs 1.3 GB** for pure TP — a **1.4 GB**
overhead.

Fix: MOE groups now disable pynccl + custom_allreduce (they only ever do
`all_reduce`, which the standard NCCL path handles fine). Direct memory win
on our `--tp=4 --ep=4` configurations.

### `reduce_scatterv` Replaces all-reduce + dp_scatter for DP Attention (PR #22642)

For DP attention + EP, the old path was:
1. `tensor_model_parallel_all_reduce` across DP workers
2. `dp_scatter` to extract local slice

Now fused into a single `reduce_scatterv` NCCL collective — half the comm rounds.
Combined with `#22515` this is the most impactful comm path change in the window.

### Ray scheduler NUMA binding (PR #22989)

`SGLANG_NUMA_BIND_V2=True` (default) skipped Ray-spawned actors entirely because
they're not `multiprocessing.spawn`'d. Scheduler actors were running unbound on
the wrong socket.

Fix: in-process libnuma bind inside the Scheduler before model load.

**Not directly relevant** to us (no Ray backend), but indicates V2 path is
production-tested now.

### Multi-replica Ray serving with unique actor names (PR #22917)

Same caveat — Ray-only.

### Scheduler launch hang on rank death (already in v0.5.10, unchanged)

Carried forward.

### `pp_max_micro_batch_size=0` silent deadlock fix (PR #23799)

Now rejected with clear error instead of hanging in `generate()`.

---

## Piecewise CUDA Graph (PCG)

PCG remains the default. Several PCG-specific fixes & extensions:

### Speculative Decoding + PCG (PR #22128)

**Removes the rule that disabled PCG when speculative decoding was on.**
Reasoning: PCG captures `ForwardMode.EXTEND` (prefill) with `spec_info=None`,
while spec-decode uses `TARGET_VERIFY` decode graphs — independent paths. The
old guard (#16331) was a CYA fix when PCG became default.

Now: `--enable-piecewise-cuda-graph` works alongside EAGLE/EAGLE3/NEXTN/STANDALONE/NGRAM.

**Action item**: revisit the speculative profiles (`speculative_*` vars in
defaults). Some models had PCG silently disabled. Combined with `#22836`
(EAGLE3/DFLASH aux capture before graph init) this is a real first-token-latency
win on speculative paths.

### Inductor Path FP8 Optimization (PR #21734 / redo #23227)

`apply_qk_norm` skips `fused_inplace_qknorm` when
`piecewise_cuda_graph_compiler=inductor`; reshape pattern changed from
`reshape(-1, head_dim)` to a stride-preserving `view(...)` so inductor fuses
cleanly. Also FP8 `apply_fp8_linear` tweaks. Reduces kernel-launch overhead on
FP8 models under PCG-inductor.

### Breakable PCG (PR #22218, experimental)

`--enable-breakable-cuda-graph` — alternative PCG implementation that does NOT
require `torch.compile`. Marked experimental. Potentially useful if our
`torch.compile` interactions misbehave.

### `--moe-dense-tp-size 1` + PCG compatibility (PR #23972)

Was incompatible until now.

### EAGLE3 hidden-state copy in PCG (PR #23613)

`mm_input` is now copied for EAGLE3 in piecewise cuda graph — previously could
alias.

---

## Performance

### Token Group Quant: faster v2 kernel default (PR #22467)

`sgl_per_token_group_quant_8bit_v2` now the default; `per_token_group_quant_fp8`
redirected to it. Direct latency win on FP8 paths.

### `act_and_mul_triton` deprecated, folded into JIT silu/gelu_and_mul (PR #23707)

Cleanup; minor speedup from removing extra dispatch hop.

### NSA indexer: fewer kernels and copies (PR #22232)

DSA / GLM-5 pathway.

### `MemoryPoolConfigurator` class hierarchy (PR #22389)

Internal refactor — more extensible KV/Mamba pool config; cleaner shape for
NVFP4 KV cache PR series to land into.

### MoE Gating Optimizations

- DSA / GLM-5: all-reduce fusion enabled (#22390).
- DeepEP LL Dispatch FP8 communication for DeepSeek-R1-0528-w4a8 (#22316).
- DeepGemm warmup envvar for DeepSeek-V4 (#23756).

### Skip `torch.cuda.empty_cache()` in weight-update flush path (#22998)

Removes a slow synchronous cuda call from RL weight-update hot path.

### Whisper Encoder Batching for Concurrent Prefill (#22361)

Whisper now batches the encoder forward across concurrent transcription requests.

---

## API / Tool Calling

### `defer_loading` field at function level for ChatCompletions (PR #22702)

Per-function streaming defer hint.

### DSV4 / `latest_reminder` / content-parts in OpenAI chat API (PR #23692)

API surface for DSV4 / content-parts (mixed text+image inline).

### GLM tool-call value whitespace preserved (PR #20543)

Don't strip whitespace from GLM tool-call values — was breaking JSON values that
intentionally start/end with spaces.

### Tool-call message normalization for GLM5.1 (PR #22595)

Chat-template conformance fix.

### `parallel_tool_calls` (already in v0.5.10) — unchanged.

### Streaming Validation HTTP 400 (already in v0.5.10) — unchanged.

### Streaming Session Spec-v2 Bonus Accounting + Test Matrix (PR #22651)

Capstone PR for the streaming-session correctness work. Spec-v2 bonus-slot
accounting was the last open correctness item; the PR also lands a comprehensive
test matrix. Look at the PR description for the index of all streaming-session
fixes (#22862 tail-free, #22897 overshoot trim, …).

### Chunk-Based Streaming ASR for Qwen3-ASR (#22089)

ASR pipeline gains chunked streaming.

---

## Observability / Logging

- **OpenTelemetry tracing for Pipeline Parallelism** (PR #23169) — opt-in OTel
  tracer covering PP forward.
- **`engine_type` label on tokenizer manager metrics** (#23978) —
  Prometheus dashboards can now distinguish `srt` vs other engines.
- **Quieter startup**: noisy third-party warnings filtered (#23669); `req_time_stats`
  cleanup (#22186); UX log cleanup (#22174).

---

## EADDRINUSE Sidecar Status

**Still needed in `0.5.10-20260429-dev1`.** No fix in this 832-commit window.
Scheduler subprocess still binds `<pod-ip>:<port>` for internal communication;
uvicorn `0.0.0.0:<same-port>` still conflicts. **HAProxy sidecar stays.**

---

## Status of Our Known Bugs

| Bug                                               | Doc                                          | v0.5.10             | dev1                                                                         |
|---------------------------------------------------|----------------------------------------------|---------------------|------------------------------------------------------------------------------|
| reasoning_tokens always 0                         | `SGLANG_REASONING_TOKENS_UPSTREAM_BUG.md`    | FIXED (PR #15562)   | FIXED                                                                        |
| moe_wna16 qzeros + EP                             | `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md`           | NOT FIXED           | NOT FIXED — vLLM PR #35598 still open                                        |
| EPLB + Qwen3 MoE                                  | same                                         | NOT FIXED           | NOT FIXED — PR #21822 still open                                             |
| NVFP4 input_scale + EP                            | same                                         | NOT FIXED           | NOT FIXED — PRs #20869/#21630/#20963/#23531 still open. Cited by issue #24502 (Blackwell EP+NVFP4) as stacking prerequisite |
| ModelOptModelLoader + sharded_state               | same                                         | NOT FIXED           | NOT FIXED — PR #21612 still open                                             |
| CutlassMoEParams global num_experts               | same                                         | NOT FIXED           | NOT FIXED — unreported                                                       |
| sharded_state + speculative                       | `SGLANG_SHARDED_SPECULATIVE_UPSTREAM_BUG.md` | NOT FIXED           | NOT FIXED                                                                    |
| MoE Triton tuning text_config                     | `SGLANG_MOE_TUNE_UPSTREAM_BUG.md`            | FIXED               | FIXED                                                                        |
| Gemma-4 NVFP4 SM121 (4 PRs)                       | `SGLANG_GEMMA4_UPSTREAM_BUG.md`              | BLOCKED             | **STILL BLOCKED** — none of #22929/#22928/#22927/#22615 merged by 2026-04-29 |
| Gemma-4 BF16 needs main build                     | same                                         | needed main build   | **NOT NEEDED** — `#21952` is in dev1                                         |
| Qwen3.5 FP8 per-tensor crash                      | (regression in v0.5.10)                      | BROKEN              | **FIXED** (PR #23062)                                                        |
| Qwen3.5-27B GDN accuracy                          | (regression in v0.5.10)                      | BROKEN (49/50→3/50) | **FIXED** (PR #22312)                                                        |
| EAGLE/EAGLE3 TP>1 non-greedy hang                 | (silent)                                     | BROKEN              | **FIXED** (PR #22458)                                                        |
| EAGLE3/DFLASH aux capture lost under PCG          | (silent, accept-len=0)                       | BROKEN              | **FIXED** (PR #22836)                                                        |
| DP attention + EP + reduce_scatterv double-reduce | (regression)                                 | BROKEN              | **FIXED** (PRs #23731/#23732/#23734)                                         |

---

## Upgrade Action Items

1. **Image bump done** — `default_sglang_image` at
   `roles/k8s_dgx/defaults/main/sglang.yml:10` (default_sglang_image) is already on dev1. Keep `0.5.10` line
   commented for fast revert.
2. **Update `SGLANG_EXPECTED_IMAGE`** in `sglang_launch.sh` and
   `sglang_shard_launch.sh` to the new tag if not already.
3. **`xomoxcc/main-gemma4-sm121` retirement (BF16 only)** — PR `#21952` is in
   dev1, so BF16 Gemma-4 dense and MoE no longer need the custom build. Test
   `google/gemma-4-31B-it` and `google/gemma-4-26B-A4B-it` on dev1 before
   removing the custom image alias from any model profile.
4. **Re-enable PCG on speculative profiles** — `#22128` removes the guard.
   Inspect each model profile in `roles/k8s_dgx/model_profiles/*.yml`: any with
   `enable_piecewise_cuda_graph: false` set *because of* speculative decoding
   should be re-tested with PCG ON. Combined with `#22836` aux-capture fix,
   acceptance length should now be non-zero.
5. **Re-test Qwen3.5 FP8 per-tensor** profiles — they were broken on v0.5.10
   (PR #23062). Confirm load + serve.
6. **Re-test Qwen3.5-27B accuracy** — `#22312` reverses the GSM8K collapse.
7. **DP attention + EP** — if any profile combined `enable_dp_attention=true`
   with EP, recheck outputs (PR #23731 chain).
8. **HAProxy sidecar still needed** — EADDRINUSE NOT fixed.
9. **Remaining monkey-patches** — same set as v0.5.10:
   - moe_wna16 qzeros EP patch
   - CutlassMoEParams num_experts patch
   - modelopt_quant NVFP4 input_scale patch (NVFP4 + EP)
   - sharded_state + speculative workaround (CLI flags)
10. **transformers 5.3 → 5.6 audit** — anything in `sglang_launch.sh` /
    monkey-patches that touches `transformers` internals must be re-verified.
    Tokenizer / config layout differs.
11. **CUDA 13.0** — base image moved cu129 → cu130. Any host-side CUDA toolkit
    version pin (e.g. for in-cluster torch builds) needs to match. The DGX
    Spark drivers must support cu130; verify with `nvidia-smi` driver version
    on each spark.
12. **NVFP4 SM121 still gated** — `SGLANG_GEMMA4_UPSTREAM_BUG.md` blockers
    haven't moved. Don't switch any NVFP4 Gemma-4 profile back to the official
    image yet.
13. **DFLASH speculative** — new option for our model profiles. Might be worth
    benchmarking for Qwen3.5 (block-based draft, no separate draft model
    needed) once a compatible target+config is published upstream.
14. **`moe_wna16` patch site re-check** — the GDN refactor and several MoE
    refactors in this window may have moved the patch target string. Run
    `sglang_launch.sh` once with `-x` and look for "patch target not found"
    warnings before declaring the deploy stable.
