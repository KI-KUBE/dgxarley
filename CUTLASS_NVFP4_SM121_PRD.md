# PRD: CUTLASS NVFP4 SM121 Shared Memory Fix

## Problem

All NVFP4 MoE models (GLM-4.7-NVFP4, MiniMax-M2.5-NVFP4, Qwen3.5-397B-NVFP4, Qwen3-235B-NVFP4) fail on DGX Spark (GB10 / SM121) with two distinct crash modes:

**Crash A — `triton` MoE runner (device-side assert):**
```
RuntimeError: Runtime check failed at nvfp4_blockwise_moe.cuh:78: CUDA error: device-side assert triggered
```
Occurs during both CUDA graph capture AND eager inference. Root cause: shared memory overflow in CUTLASS grouped GEMM kernel.

**Crash B — `flashinfer_cutlass` MoE runner (Xid 13 Illegal Instruction):**
```
NVRM: Xid (PCI:000f:01:00): 13, Graphics SM Warp Exception: Illegal Instruction Parameter
ESR 0x1c81fb60:0x1174  (consistent across all crashes)
Fatal Python error: Aborted
  File "flashinfer/fused_moe/core.py", line 490 in cutlass_fused_moe
```
Occurs during inference on v0.5.10 only. The same config worked on v0.5.10rc0. Root cause: FlashInfer CUTLASS MoE kernel regression between rc0 and 0.5.10.

**Combined impact:** No NVFP4 MoE runner works on v0.5.10 for SM121. The only stable configuration known is v0.5.10**rc0** with `flashinfer_cutlass` MoE + `triton` attn + `flashinfer_cudnn` fp4 + eager (GLM-4.7 rc0 Test 23: 8.06/21.94/30.01 tok/s).

## Root Cause Analysis

### Crash A: shared memory overflow in `nvfp4_blockwise_moe.cuh`

The SM120 kernel path in `nvfp4_blockwise_moe.cuh` uses:

```cpp
// Line 290-293
using ArchTag = cutlass::arch::Sm120;
using ThreadBlockShape = Shape<_128, _128, _128>;
using StageCountType = cutlass::gemm::collective::StageCountAuto;
// KernelSchedule:
cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong  // ← double-buffered
```

`StageCountAuto` + `Pingpong` schedule double-buffers the tiles, requiring ~147 KB shared memory.

| GPU | SM Version | Shared Memory per Block |
|-----|-----------|------------------------|
| B200/B100 (datacenter) | SM100 | 228 KB |
| RTX 5090 (consumer) | SM120 | 228 KB |
| **DGX Spark GB10** | **SM121** | **101 KB** |

SM121 dispatches to the SM120 kernel (`sm_version >= 120`, line 809) but has 46 KB less shared memory than SM120 expects. Kernel launch triggers a device-side assert. All subsequent CUDA calls (including `cudaMallocAsync` at line 78) return `cudaErrorAssert` (sticky error).

### Crash B: FlashInfer regression between rc0 and 0.5.10

Comparison of `scitrera/cuda-containers` recipes:

| Aspect | rc0 recipe | 0.5.10 recipe |
|--------|-----------|---------------|
| Base image | `nvcr.io/nvidia/pytorch:26.02-py3` | `scitrera/dgx-spark-pytorch-dev:2.11.0-v1-cu132` |
| FlashInfer version | unset (bundled with SGLang) | `0.6.7.post3` (explicit pin) |
| sgl-kernel ref | v0.5.10rc0 | v0.5.10 |
| Transformers | unset | `5.5.0` |

The `cutlass_fused_moe` kernel in `flashinfer/fused_moe/core.py:490` differs between the FlashInfer bundled with rc0 and 0.6.7.post3 used in 0.5.10. The 0.6.7.post3 version triggers Xid 13 Illegal Instruction on SM121 — a sgl-kernel / FlashInfer binary regression, NOT a Python-level change.

### Previous investigations (ruled out)

1. **Python DSL `admissible_archs`** ([NVIDIA/cutlass#2800](https://github.com/NVIDIA/cutlass/issues/2800)): We patched `BlockScaledMmaOp.admissible_archs` to include `sm_120a` + `sm_121a`. **Ineffective** — the `.cuh` kernel is JIT-compiled via TVM/C++, not through the Python DSL.

2. **Runtime `.cuh` source patch**: We attempted to inject an SM121-specific kernel function with `Shape<_64, _128, _128>` tiles via python heredoc string replacement at startup. **Failed** — CUTLASS template validation rejects the smaller tile shape with the `KernelPtrArrayTmaWarpSpecializedPingpong` schedule:
   ```
   error: static assertion failed with "TMA requires CTA_Tile and SLayout top-level size equivalence."
   error: static assertion failed with "Shape Divisibility Condition"
   error: static assertion failed with "Could not find a common tile-gmem vectorization."
   ```
   The Pingpong schedule has strict TMA descriptor requirements that don't allow reducing M or N dimensions independently.

### What BTankut did differently (important correction)

The [BTankut/dgx-spark-sglang-moe-configs](https://github.com/BTankut/dgx-spark-sglang-moe-configs) repo targets **GLM-4.7-FP8**, not NVFP4. His base image `lmsysorg/sglang:spark` (SGLang v0.5.4.post2, FlashInfer 0.5.0) was inspected:

- The file is `nvfp4_blockwise_moe.cu` (compile-time C++), not `.cuh` (JIT-compiled via TVM in v0.5.10)
- **Identical SM120 function** to v0.5.10: `Shape<_128, _128, _128>`, `KernelPtrArrayTmaWarpSpecializedPingpong`, `StageCountAuto`
- **No SM121 code path at all** — SM121 falls into `TORCH_CHECK_NOT_IMPLEMENTED(false, "Unsupported SM version: " + std::to_string(sm_version))`
- Dispatch: `} else if (sm_version == 120) {` (strict `==`, not `>= 120`)

This means `lmsysorg/sglang:spark` would **crash on SM121** if you tried to run NVFP4 MoE on it. BTankut never ran NVFP4 MoE — his 20–27 tok/s result is for GLM-4.7-**FP8** with Triton MoE (tuned via the MoE kernel config JSONs for the 101 KB shared memory budget).

**The 356 TFLOPS NVFP4 result from the forum post** is a dense GEMM micro-benchmark, not a running MoE inference. It demonstrates that SM121 FP4 Tensor Cores work in principle, but does not show a working sgl-kernel + NVFP4 MoE pipeline.

**Conclusion: no known working NVFP4 MoE configuration exists on SM121 in any public sglang build.**

The v0.5.10 code change from `sm_version == 120` to `sm_version >= 120` was the SGLang team's attempt to enable SM121 — it "activates" the code path but fails at runtime because of the 46 KB shared memory shortfall. Copying the v0.5.4 `.cu` would regress us to `Unsupported SM version` crashes — worse than what we have now.

## Known-good baseline

**v0.5.10rc0** with `flashinfer_cutlass` MoE + `triton` attn + `flashinfer_cudnn` fp4 + eager (no CUDA graphs):
- GLM-4.7-NVFP4 TP=4 EP=4: **8.06 / 21.94 / 30.01 tok/s** at n=1/n=4/n=8 (rc0 test #23)
- Stable across all concurrency levels

Any change away from this baseline (newer SGLang, newer FlashInfer, different MoE runner, different fp4_gemm backend) breaks NVFP4 MoE on SM121.

**Why does rc0 work?** The rc0 `flashinfer_cutlass` MoE runner routes through a FlashInfer kernel path (`flashinfer/fused_moe/core.py:cutlass_fused_moe`) that either:
1. Does not invoke the problematic `nvfp4_blockwise_moe.cu` grouped GEMM (uses a different CUTLASS kernel with smaller tiles), OR
2. Uses a FlashInfer version (likely 0.6.5 or 0.6.6) that handles SM121 shared memory differently

The rc0 → 0.5.10 FlashInfer upgrade (to 0.6.7.post3) introduced either a new kernel dispatch or a regression that triggers the Xid 13 Illegal Instruction on SM121.

## Implementation Options

### Option 1: Pin FlashInfer to rc0 version (fastest, RECOMMENDED)

Determine which FlashInfer version rc0 bundled (likely 0.6.5 or 0.6.6) and pin it explicitly in the 0.5.10 recipe. This restores the proven-working `flashinfer_cutlass` MoE path on SM121.

**Steps:**
1. Run `docker run --rm scitrera/dgx-spark-sglang:0.5.10rc0 pip show flashinfer-python 2>/dev/null | grep Version`
2. Fork `scitrera/cuda-containers`, edit `container-recipes/sglang-0.5.10.recipe`:
   ```
   FLASHINFER_VERSION=<rc0_version>
   ```
3. Build: `container-build/build.sh sglang 0.5.10-fi-pin`
4. Update `sglang_image` in `roles/k8s_dgx/defaults/main.yml` to the new tag

**Pros:** Minimal change. Restores rc0's stable baseline. No source code edits.
**Cons:** Still only fixes `flashinfer_cutlass` MoE runner (Crash B). `triton` MoE runner (Crash A) remains broken. Locks out FlashInfer bug fixes.

### Option 2: Modify sgl-kernel source before build (speculative)

Patch `sgl-kernel/csrc/moe/nvfp4_blockwise_moe.cu` (note: `.cu` in 0.5.10 as well, JIT happens via sgl_kernel's jit_kernel module) in `scitrera/cuda-containers` build to add a compile-time SM121-specific kernel path with reduced shared memory footprint. Speculative because:

1. Runtime `.cuh` patching failed with CUTLASS TMA static_asserts (see below)
2. Compile-time patching has the same template constraints — same static_asserts will fire
3. The valid tile/schedule combinations for CUTLASS Sm120 ArchTag with NVFP4 block-scaled ops within 101 KB shared memory are unknown
4. Would require empirical iteration on a slow (30-60 min) sgl-kernel build loop

**Recommendation: try Option 1 first.** If Option 1 works, Option 2 is unnecessary for the `flashinfer_cutlass` MoE codepath. Option 2 only makes sense if we need the `triton` MoE runner specifically, which has never worked on SM121 with any SGLang version.

## Implementation (Option 2)

### Target file

`sgl-kernel/src/sgl-kernel/csrc/moe/nvfp4_blockwise_moe.cuh` in the SGLang source tree cloned during Docker build.

### Patch location in build pipeline

In `scitrera/cuda-containers/container-build/Dockerfile.sglang-nightly`, insert a `sed`/`python3` patch step **after** the sgl-kernel source is cloned and **before** `uv build --wheel` runs:

```dockerfile
# Patch nvfp4_blockwise_moe.cuh for SM121 (GB10, 101 KB shared memory)
# The default SM120 kernel path uses Pingpong schedule + Shape<_128,_128,_128>
# which requires ~147 KB shared memory. SM121 only has 101 KB.
# Fix: add SM121-specific kernel function with a schedule + tile combination
# that fits in 101 KB.
COPY patches/sgl-kernel-sm121.patch /tmp/
RUN cd /data/sglang/sgl-kernel && \
    patch -p1 < /tmp/sgl-kernel-sm121.patch && \
    grep -q 'run_fp4_blockwise_scaled_group_mm_sm121' src/sgl-kernel/csrc/moe/nvfp4_blockwise_moe.cuh
```

### Patch content: several candidate approaches

#### Approach 2a: Reduce pipeline stages (keep tile shape, reduce double-buffering)

Keep `Shape<_128, _128, _128>` but force `StageCount<1>` instead of `StageCountAuto`:

```cpp
// Current (SM120):
using StageCountType = cutlass::gemm::collective::StageCountAuto;
// ... in CollectiveMainloop builder:
cutlass::gemm::collective::StageCountAutoCarveout<...>,

// SM121 variant:
using StageCountType = cutlass::gemm::collective::StageCount<1>;
// ... in CollectiveMainloop builder:
cutlass::gemm::collective::StageCount<1>,
```

Single-stage = no double buffering. Shared memory halves from ~147 KB to ~74 KB. Fits in 101 KB.

**Risk:** Lower throughput (no pipeline overlap). But: still faster than no inference at all.

#### Approach 2b: Switch to single-SM schedule (like SM100 path)

The SM100 path uses `KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100`. Create an SM121 variant using the generic 1Sm schedule:

```cpp
// Replace:
cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong
// With:
cutlass::gemm::KernelPtrArrayTmaWarpSpecialized  // Non-pingpong variant
```

**Risk:** The generic `KernelPtrArrayTmaWarpSpecialized` (non-pingpong) may not exist as a named schedule for FP4 block-scaled kernels, or may have different API requirements.

#### Approach 2c: Combined — smaller tile + Sm100 schedule hybrid

Use `Shape<_128, _128, _64>` (reduce K from 128 to 64, not M/N). K is the accumulation dimension — reducing it doesn't affect TMA descriptor validation the same way as M/N reductions do.

```cpp
using ThreadBlockShape = Shape<_128, _128, _64>;
// Keep Pingpong schedule unchanged
```

**Risk:** K=64 may violate CUTLASS FP4 granularity requirements (FP4 uses group_size=16 scales, K must be multiple of scale block).

### Validation plan

Build sgl-kernel with the patch, then test each approach on a DGX Spark with GLM-4.7-NVFP4 (TP=4 EP=4, `triton` MoE runner, `disable_cuda_graph: true`):

1. Patch applies cleanly during Docker build
2. sgl-kernel wheel builds without CUTLASS static_assert errors
3. SGLang starts without crash
4. First inference request completes without device-side assert
5. Throughput at n=1, n=4, n=8 — compare to rc0 baseline (8.06/21.94/30.01)

If one approach fails to compile, try the next. Approach 2a (StageCount<1>) has the highest probability of success since it's the smallest change and doesn't touch TMA descriptor logic.

### Docker build integration

The patch file `sgl-kernel-sm121.patch` should be stored in `scitrera/cuda-containers/container-build/patches/` and referenced from the Dockerfile as shown above. Build with:

```bash
cd scitrera/cuda-containers
container-build/build.sh sglang 0.5.10-sm121
```

Resulting image: `scitrera/dgx-spark-sglang:0.5.10-sm121`.

### Fix for Crash B (FlashInfer regression) — separate change

Independently of the sgl-kernel patch, `FLASHINFER_VERSION` in the 0.5.10 recipe should be downgraded from `0.6.7.post3` to whatever version rc0 bundled. This fixes `flashinfer_cutlass` MoE runner stability.

Check rc0 bundled FlashInfer version:
```bash
docker run --rm scitrera/dgx-spark-sglang:0.5.10rc0 \
  pip show flashinfer-python 2>/dev/null | grep Version
```

Then pin in `container-recipes/sglang-0.5.10.recipe`:
```
FLASHINFER_VERSION=<rc0_version>
```

## Validation — Success Criteria

- Crash A fixed: `triton` MoE runner (→ `cutlass_moe_fp4`) works on SM121 without device-side assert
- Crash B fixed: `flashinfer_cutlass` MoE runner works on SM121 without Xid 13 Illegal Instruction
- GLM-4.7-NVFP4 TP=4 EP=4 throughput ≥ rc0 baseline (8.06 / 21.94 / 30.01 tok/s)
- All other NVFP4 MoE models (MiniMax-M2.5, Qwen3-235B, Qwen3.5-397B) also stable
- CUDA graphs can be enabled (previously forced off with `disable_cuda_graph: true`)

## Risks

1. **CUTLASS template constraints are strict**: Smaller tiles may fail compilation with static_asserts. Multiple approaches need trying (2a, 2b, 2c).
2. **StageCount<1> reduces throughput**: Single-stage pipeline is slower than pingpong. Expected ~20-40% slower per-kernel.
3. **SGL-kernel build is long**: Full sgl-kernel compilation takes 30-60 minutes. Iteration cycle is slow.
4. **Custom image maintenance**: The patched image must be rebuilt for every SGLang upgrade. Grep/sed anchors may break with upstream changes.
5. **Upstream fix may land first**: SGLang [#11658](https://github.com/sgl-project/sglang/issues/11658) tracks SM121 support. An official fix may come before our custom build is ready.
6. **Crash B (FlashInfer) may need separate fix**: Approach 2 only addresses Crash A. If `flashinfer_cutlass` MoE is also broken, both fixes are needed.

## References

- [NVIDIA/cutlass#2800](https://github.com/NVIDIA/cutlass/issues/2800) — Python DSL arch restriction (not the root cause)
- [SGLang SM121 tracking #11658](https://github.com/sgl-project/sglang/issues/11658) — upstream support status
- [BTankut/dgx-spark-sglang-moe-configs](https://github.com/BTankut/dgx-spark-sglang-moe-configs) — GLM-4.7-**FP8** (not NVFP4) workaround for v0.5.4
- [scitrera/cuda-containers](https://github.com/scitrera/cuda-containers) — our base image build recipes
- [Forum: SM121 CUTLASS optimization](https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960) — 356 TFLOPS NVFP4 **dense GEMM** benchmark only (not a working MoE inference pipeline)
- [Forum: GLM-4.7-FP8 on 4x DGX Spark](https://forums.developer.nvidia.com/t/running-glm-4-7-fp8-355b-moe-on-4x-dgx-spark-with-sglang-eagle-speculative-decoding/359256) — MoE tuning + EAGLE (FP8 only)
- `nvfp4_blockwise_moe.cu` source analysis — compared v0.5.4 (`lmsysorg/sglang:spark`) and v0.5.10 (`scitrera/dgx-spark-sglang:0.5.10`). **Both versions have the identical SM120 kernel** (same Pingpong schedule, same `Shape<_128,_128,_128>`), neither has an SM121-specific kernel path. Only difference: v0.5.4 dispatches `sm_version == 120` (SM121 → Unsupported), v0.5.10 dispatches `sm_version >= 120` (SM121 → crashes).
- rc0 vs 0.5.10 recipe comparison from `scitrera/cuda-containers`
