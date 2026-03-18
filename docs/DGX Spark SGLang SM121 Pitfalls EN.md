# SGLang on DGX Spark (GB10 / SM121) — Known Pitfalls & Workarounds

_As of: March 17, 2026 — findings from the production K3s/Ansible implementation (Repo: dgxarley)_

The NVIDIA GB10 Grace Blackwell Superchip in DGX Spark / ASUS Ascent GX10 reports as **SM121** (`sm_121a`, compute capability 12.1). This is the consumer/workstation Blackwell — distinct from the datacenter B200/B100 which is **SM100** (`sm_100a`). Several SGLang/vLLM kernel paths assume SM100 when they detect "Blackwell", causing crashes on SM121.

This document covers all pitfalls discovered during deployment and their fixes.

## Hardware Context

| | DGX Spark (GB10) | B200 (Datacenter) |
|---|---|---|
| SM version | 121 (`sm_121a`) | 100 (`sm_100a`) |
| GPU memory | 128 GB LPDDR5x (unified with CPU) | 192 GB HBM3e |
| `nvidia-smi` memory reporting | **"Not Supported"** | Normal |
| `torch.cuda.mem_get_info()` | Works (~121.6 GB visible) | Works |

**Key difference**: `nvidia-smi` cannot report memory usage on GB10. SGLang falls back to `torch.cuda.mem_get_info()` with this warning:
```
WARNING common.py: Failed to get GPU memory capacity from nvidia-smi, falling back to torch.cuda.mem_get_info().
```
This is harmless — the fallback works correctly.

---

## Pitfall 1: DeepGemm JIT — "Unknown recipe"

**Symptom**: Crash during CUDA graph capture:
```
RuntimeError: Assertion error (.../deepgemm/.../layout.hpp:56): Unknown recipe
```

**Cause**: DeepGemm auto-enables on Blackwell (SM ≥ 90) but its JIT kernels lack "recipes" for certain GEMM configurations on SM121. The `scale_fmt` of FP8 checkpoints is not `ue8m0` (the format DeepGemm expects on Blackwell).

**Fix**: Disable DeepGemm via environment variable:
```yaml
SGLANG_ENABLE_JIT_DEEPGEMM: "false"
```

In the Ansible model profile:
```yaml
disable_deep_gemm: true
```

---

## Pitfall 2: FlashInfer FP8 GEMM — "does not support backend 'trtllm' with capability 121"

**Symptom**: Crash during CUDA graph capture after fixing DeepGemm:
```
flashinfer.utils.BackendSupportedError: gemm_fp8_nt_groupwise does not support backend 'trtllm' with capability 121
```

**Cause**: With DeepGemm disabled, SGLang's auto-dispatch selects FlashInfer's `trtllm` backend for FP8 block GEMM. FlashInfer 0.6.3 (bundled with SGLang 0.5.9) has a `@supported_compute_capability([100, 103])` guard — SM121 is not in that list. The fix (flashinfer PR #2631) landed in FlashInfer 0.6.5, one version after SGLang 0.5.9 shipped.

**Fix**: Force CUTLASS FP8 backend (CLI flag name is `--fp8-gemm-backend`, not `--fp8-gemm-runner-backend`):
```yaml
fp8_gemm_runner_backend: "cutlass"
```

The CUTLASS kernel in sgl-kernel has explicit `if (sm_version >= 120)` support for SM121.

**Note**: This only affects **FP8** models. AWQ/GPTQ/BF16 models use different kernel paths and are unaffected.

---

## Pitfall 3: AWQ MoE models — `--quantization awq` causes FP16 fallback

**Symptom**: OOM during model initialization — allocating FP16 tensors for MoE experts instead of 4-bit AWQ:
```
File "sglang/srt/layers/quantization/unquant.py", line 190, in create_weights
    torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
```

**Cause**: Explicitly passing `--quantization awq` creates `AWQConfig`, whose `get_quant_method()` returns `None` for `FusedMoE` layers. This causes MoE layers to fall through to `UnquantizedFusedMoEMethod`, which allocates full FP16 weight tensors.

Without `--quantization`, SGLang auto-detects `quant_method: "awq"` from the model's `config.json` and creates `AWQMarlinConfig` instead. `AWQMarlinConfig` correctly handles FusedMoE via `AWQMoEMethod` with 4-bit packed weights.

**Fix**: Do NOT pass `--quantization awq` for MoE models. Remove `quantization:` from the model profile and let auto-detection handle it:
```yaml
# WRONG — causes FP16 fallback for MoE layers:
quantization: "awq"

# CORRECT — auto-detects AWQMarlinConfig from config.json:
# (no quantization field)
```

**Exception**: Dense (non-MoE) AWQ models like `Qwen2.5-Coder-32B-Instruct-AWQ` work fine with explicit `--quantization awq` because they don't use FusedMoE.

---

## Pitfall 4: AWQ Marlin repack — CUDA allocator fragmentation OOM

**Symptom**: OOM during `process_weights_after_loading()` even though final model size fits in GPU memory. Peak memory is ~1.7× the expected model size:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB.
GPU has 121.63 GiB total, process has 105.6 GiB in use.
```

**Cause**: After loading AWQ weights, SGLang converts them from AutoAWQ layout to Marlin layout (`awq_marlin_moe_repack`). For each of the 94 layers:
1. The repack allocates a new tensor for the Marlin-formatted weights
2. `replace_parameter()` drops the old tensor reference (type mismatch: `PackedvLLMParameter` vs `Tensor` causes fallback path)
3. The freed GPU memory goes to PyTorch's CUDA caching allocator but isn't efficiently reused due to size/alignment mismatches
4. Over 94 layers, ~37 GB of fragmented cached blocks accumulate

Memory math: 62 GB (weights) + 37 GB (fragmentation) + overhead = ~105 GB peak.

**Fix**: Use `--quantization moe_wna16` which routes MoE layers through Triton-based INT4 kernels (no Marlin repack at all). Peak memory = final model size (~62 GB). Slightly slower inference than Marlin but the only option on memory-constrained GPUs:
```yaml
quantization: "moe_wna16"
```

**Insufficient workaround**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation during loading (60 GB stable) but does NOT prevent the repack OOM — the repack legitimately needs old + new tensors to coexist temporarily, pushing peak to ~109 GB regardless of allocator strategy.

**References**:
- SGLang issue [#11471](https://github.com/sgl-project/sglang/issues/11471) — Qwen3-VL-235B-AWQ OOM
- vLLM issue [#21864](https://github.com/vllm-project/vllm/issues/21864) — GPU OOM when processing AWQ Marlin weights

---

## Pitfall 5: K8s Service name collision — `SGLANG_PORT` env injection

**Symptom**: Crash during model initialization:
```
ValueError: invalid literal for int() with base 10: 'tcp://10.69.141.194:8000'
```

**Cause**: Kubernetes automatically injects environment variables for every Service in the same namespace. A Service named `sglang` creates `SGLANG_PORT=tcp://10.69.x.x:8000`. SGLang's `get_open_port()` reads this env var expecting an integer port number and crashes.

**Fix**: Disable Kubernetes service link injection in the Pod spec:
```yaml
spec:
  enableServiceLinks: false
```

This must be set on all SGLang Pods (head, worker, shard jobs).

---

## Pitfall 6: NVFP4 quantization — unstable on SM121

**Symptom**: NaN outputs, kernel crashes, or `cudnnGraphNotSupportedError` when using NVIDIA ModelOpt NVFP4 models.

**Cause**: The native FP4 GEMM kernels (CUTLASS `BlockScaledMmaOp`, FlashInfer TRTLLM MoE FP4) are compiled only for SM100. On SM121, they fall back to Marlin kernels which have known NaN issues under concurrent load.

**Status**: As of March 2026, NVFP4 on SM121 is a work-in-progress:
- SGLang issue [#11658](https://github.com/sgl-project/sglang/issues/11658) — DGX Spark SM121 support tracking
- SGLang issues [#20043](https://github.com/sgl-project/sglang/issues/20043), [#18954](https://github.com/sgl-project/sglang/issues/18954) — NaN outputs with NVFP4

**Recommendation**: Avoid NVFP4 on DGX Spark. Use FP8 (with CUTLASS backend) or AWQ 4-bit instead.

---

## Summary: Recommended Model Profile Settings for DGX Spark

### FP8 models (e.g. Qwen3.5-35B-A3B-FP8)
```yaml
disable_deep_gemm: true
fp8_gemm_runner_backend: "cutlass"
# No explicit quantization field (auto-detect)
```

### AWQ MoE models (e.g. Qwen3-235B-A22B-AWQ)
```yaml
quantization: "moe_wna16"  # avoids Marlin repack OOM, uses Triton INT4 kernels
# Do NOT use quantization: "awq" (FP16 fallback for MoE)
# Do NOT omit quantization (auto-detect uses Marlin repack → OOM on 121 GB GPUs)
```

### AWQ dense models (e.g. Qwen2.5-Coder-32B-AWQ)
```yaml
quantization: "awq"  # OK for dense models, not for MoE
```

### BF16 models (e.g. Qwen3-Coder-30B-A3B-Instruct)
```yaml
# No special settings needed
```

### All models on DGX Spark
```yaml
# In Pod spec:
enableServiceLinks: false

# In ConfigMap:
PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
SGLANG_ENABLE_JIT_DEEPGEMM: "false"  # unless model needs DeepGemm and has ue8m0 scales
```
