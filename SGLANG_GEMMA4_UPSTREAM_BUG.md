# SGLang Upstream Bug: Gemma-4 not supported on v0.5.10

## Status

**Blocked — requires SGLang version newer than v0.5.10.** Native Gemma-4 support
was merged upstream (PR #21952, 2026-04-07) after v0.5.10 was cut. All Gemma-4
variants (BF16 and NVFP4, dense and MoE) crash on v0.5.10 because the
Transformers fallback backend cannot handle Gemma-4's architecture. NVFP4
variants need additional PRs (#22079, #22929, #22928) that are partially still
open. Not runtime-patchable.

## Affected models

| Model | Type | Quantization | Status on v0.5.10 |
|-------|------|-------------|-------------------|
| `google/gemma-4-26B-A4B-it` | MoE (128 experts, 26B/3.8B active) | BF16 | **crash** (dual head_dim RMSNorm) |
| `google/gemma-4-31B-it` | Dense (30.7B) | BF16 | **crash** (expected same as above) |
| `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` | MoE (128 experts, 26B/3.8B active) | NVFP4 | **crash** (top_k + weight mapping + GEGLU) |
| `nvidia/Gemma-4-31B-IT-NVFP4` | Dense (30.7B) | NVFP4 | **crash** (expected same as BF16 dense) |

All use `Gemma4ForConditionalGeneration` as architecture. None has a native
SGLang model class in v0.5.10 — all fall through to the Transformers backend.

## Root cause

Gemma-4 has several architectural features that the Transformers fallback
backend in v0.5.10 does not support:

1. **Dual head dimensions** — sliding-window layers use `head_dim=256`, global
   attention layers use `global_head_dim=512`. The fallback backend creates
   RMSNorm weights uniformly with one dimension, causing shape mismatches when
   the model alternates between layer types.

2. **MoE config naming** — Gemma-4 uses `top_k_experts` instead of the standard
   `num_experts_per_tok` / `top_k` that the fallback's `_getattr_first` lookup
   expects.

3. **NVFP4 per-expert weight format** — NVFP4 checkpoints store MoE expert
   weights in unfused per-expert format, which the fallback's weight mapper
   doesn't support.

4. **GEGLU activation** — Gemma-4 MoE uses GEGLU (`gelu_tanh`), but
   `cutlass_moe_fp4()` hardcodes `silu_and_mul()`.

5. **FP4 block scale NaN** — SM120/121 specific: uint8=127 in E4M3 block scales
   triggers NaN in the CUTLASS FP4 group GEMM kernel.

Issue (1) affects **all** Gemma-4 variants (BF16 and NVFP4, dense and MoE).
Issues (2–5) affect NVFP4 variants specifically.

## Failure details

### BF16 MoE (`google/gemma-4-26B-A4B-it`) — confirmed

Crash during warmup forward at the first global-attention layer's `v_norm`:

```
gemma4/modeling_gemma4.py:1220  value_states = self.v_norm(value_states)
  → layernorm.py:207  rmsnorm(x, self.weight.data, self.variance_epsilon)
  → flashinfer/norm/rmsnorm.py:1310  kernel(...)
ValueError: Mismatched mW.shape[0] on argument #1 when calling:
  `__call__(mX: Tensor([n0, 256], bfloat16), mW: Tensor([256], bfloat16),
            mY: Tensor([n0, 256], bfloat16), M: int32, eps: float32)`,
  expected to be 256
```

The `v_norm` layer has weight `[256]` (sliding-window `head_dim`), but on a
global-attention layer the value states have dimension 512 (`global_head_dim`).
The Transformers fallback creates all attention norms with the same dimension,
not distinguishing between sliding and global layers. The native implementation
(PR #21952) has separate norm configs per layer type.

### NVFP4 MoE (`bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`) — confirmed

Three sequential failures, each uncovered after patching the previous:

1. **`Cannot determine top_k from config`** — `_getattr_first` lookup tuple
   in `transformers.py:1197` doesn't include `top_k_experts`.
   - Runtime-patched in `sglang_launch.sh` (`PATCH_TRANSFORMERS_TOPK_EOF`).
   - First patch revision had a syntax bug: inline marker comment broke the
     closing paren of `_getattr_first(...)` → `'(' was never closed` on line
     1197. Fixed by placing marker on a separate line above.

2. **`No module or parameter named 'model.language_model.layers.0.moe'`** —
   NVFP4 checkpoints store MoE expert weights in unfused per-expert format.
   The Transformers backend's weight mapper only knows the fused format.
   - **Not runtime-patchable.**
   - **Upstream fix: PR #22929** (open, 2026-04-16).

3. **(Latent) GEGLU activation mismatch** — `cutlass_moe_fp4()` hardcodes
   `silu_and_mul()`. Gemma-4 MoE uses GEGLU → garbage output even if weights
   loaded.
   - **Upstream fix: PR #22928** (open, 2026-04-16).

### Dense variants (BF16 + NVFP4)

Not separately tested. `_is_moe_model()` returns `False` → dispatches to
`TransformersMultiModalForCausalLM` (no MoEMixin) → avoids issues 1–3 above
but still hits the dual head_dim RMSNorm crash (issue 1 in the root cause list),
which is shared across all variants.

## Upstream PRs

| PR | Title | Status | Merged | Relevance |
|----|-------|--------|--------|-----------|
| [#21952](https://github.com/sgl-project/sglang/pull/21952) | [New Model] Gemma 4 | **merged** | 2026-04-07 | Native `gemma4_causal.py`, `gemma4_mm.py`, `gemma4_vision.py`, `gemma4_audio.py`. Foundation for all Gemma-4 support. Fixes the dual head_dim issue. |
| [#22079](https://github.com/sgl-project/sglang/pull/22079) | [nvidia] Gemma4 nvfp4 fix | **merged** | 2026-04-10 | Triton attention PTX register exhaustion fix for NVFP4 on GB200/sm100a. fp8 kv cache dtype autodetection. |
| [#22929](https://github.com/sgl-project/sglang/pull/22929) | Add NVFP4 per-expert weight loading for Gemma 4 MoE | **open** | — | Per-expert → fused weight mapping for NVFP4 MoE checkpoints. |
| [#22928](https://github.com/sgl-project/sglang/pull/22928) | fix(sm120): MoE GEGLU activation + FP4 block scale NaN clamp | **open** | — | GEGLU activation for `cutlass_moe_fp4()` + E4M3 NaN clamp. SM120/121 critical. |
| [#22615](https://github.com/sgl-project/sglang/pull/22615) | Fix fp8 KV cache crash with KV-shared layers in triton backend | **open** | — | fp8 kv cache + `num_kv_shared_layers > 0` (Gemma-4 has KV-shared layers). |

## What's needed to run Gemma-4 on our cluster

### BF16 variants (google/gemma-4-*)

Minimum: PR #21952 (native Gemma-4 implementation). Already merged into main.
A new image build from SGLang main would be sufficient.

### NVFP4 variants (nvidia/*, bg-digitalservices/*)

All of the following must be present:

1. PR #21952 — native Gemma-4 model implementation (foundation)
2. PR #22079 — NVFP4 quantization + fp8 kv cache fixes
3. PR #22929 — per-expert NVFP4 weight loading for MoE
4. PR #22928 — GEGLU activation + FP4 block scale NaN clamp (SM121 critical)
5. PR #22615 — fp8 kv cache with KV-shared layers (may apply)

PRs 1–2 are merged into main. PRs 3–5 are open. Once all merge, a new image
build from SGLang main (or a future v0.5.11+) will include everything.

**Estimated timeline:** PR #22929 and #22928 are fresh (2026-04-16), tested on
RTX 5090 (SM120). SM121/GB10 testing is still needed.

## Our runtime patches (v0.5.10)

The `top_k_experts` patch in `sglang_launch.sh` (`PATCH_TRANSFORMERS_TOPK_EOF`)
remains useful — it fixes the `_getattr_first` lookup for any future model that
uses `top_k_experts` instead of `num_experts_per_tok`. However, it's insufficient
to make any Gemma-4 variant work on v0.5.10 because the dual head_dim, weight
loading, and activation function issues are not patchable at runtime.

## Relationship to other bugs

- **Independent of** the FlashInfer FP4 dynamo tracing bug
  (`FLASHINFER_CUDA_VERSION_SUBPROCESS_UPSTREAM_BUG.md`) — that affects
  piecewise CUDA graphs on all NVFP4 models, not Gemma-4 specifically.
- **Independent of** the SM121 JIT arch mismatch (`kvcache.cuh:196` illegal
  instruction) — that's in sglang's own jit_kernel, not the model loader.
- **Related to** issue #22277 (Gemma4 E4B fp8 KV cache crash) — same model
  family, overlapping root cause (KV-shared layers + fp8).

## Files

- `roles/k8s_dgx/files/sglang_launch.sh` — `top_k_experts` runtime patch
  (fixes NVFP4 MoE failure #1 only).
- `roles/k8s_dgx/model_profiles/google-gemma-4-26b-a4b-it.yml` — BF16 MoE
  profile (won't start on v0.5.10).
- `roles/k8s_dgx/model_profiles/google-gemma-4-31b-it.yml` — BF16 dense
  profile (won't start on v0.5.10).
- `roles/k8s_dgx/model_profiles/bg-digitalservices-gemma-4-26b-a4b-it-nvfp4.yml`
  — NVFP4 MoE profile (won't start on v0.5.10).
- `roles/k8s_dgx/model_profiles/nvidia-gemma-4-31b-it-nvfp4.yml` — NVFP4 dense
  profile (won't start on v0.5.10).
