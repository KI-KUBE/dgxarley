# PRD / Runbook: Native DeepGEMM `fp8_paged_mqa_logits` on GB10/SM121

**Status:** ⚠️ SUPERSEDED (2026-06-21) — NOT pursued. The ~18 tok/s floor was lifted instead
via SGLang's **TileLang DSA indexer** (a 3-line compat patch + `opt_use_tilelang_indexer`),
no DeepGEMM port needed. End-to-end: n=1 ~18.7 tok/s, 12-concurrent ~42 tok/s (~2.3x). See
`scripts/patches/sglang-tilelang-018-indexer-compat.patch` and the DSV4-Flash profile. This
runbook is kept only as the fallback plan if even-higher throughput is later required.
**Author:** (investigation 2026-06-20)
**Audience:** single expert engineer (exhaustive runbook, not a product doc)
**Related memory:** `reference_dsv4_sm121_dsa_kernels`, `reference_dsv4_spec_v2_eagle`

---

## 1. Introduction / Overview

DeepSeek-V4-Flash-NVFP4 decode on our 4× GB10 (SM121) cluster is **structurally throttled to ~18 tok/s aggregate** regardless of concurrency. Root cause: the DSA "Lightning-Indexer" hot kernel **`fp8_paged_mqa_logits`** has no native SM121 implementation in DeepGEMM, so SGLang runs it in a **pure-PyTorch fallback** (`SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1`) — per decode step, across all 20 DSA layers, **outside the CUDA graph**. That per-step torch dispatch is the floor.

Upstream **DeepGEMM PR #324** ("feat: add sm120 support", `leavelet:sm120` → `deepseek-ai:nv_dev`) adds native `sm120_fp8_paged_mqa_logits.cuh` + `sm120_fp4_paged_mqa_logits.cuh` and the `arch_major==12` dispatch, explicitly covering **sm120 AND sm121 (GB10)**, and respects GB10's ~99 KB SMEM limit. This document is the runbook to **port PR #324 into our `sgl-deep-gemm` build** and validate whether it lifts the floor.

This is a **project**, not a drop-in patch (see §7). Pursue only if the floor is blocking and we don't want to wait for #324 to land in the `sgl-deep-gemm` wheel.

---

## 2. Goals

- **G1:** `deep_gemm.fp8_paged_mqa_logits` + `get_paged_mqa_logits_metadata` JIT-compile and **run correctly on GB10/SM121** (no `Unsupported architecture`, no SMEM `cudaErrorInvalidValue`, no tcgen05/wgmma errors).
- **G2:** Numerical correctness — native output **bit-close** to `fp8_paged_mqa_logits_torch_sm120` on representative DSV4 shapes (64 index heads, index_topk=512, ≤512 KV pages, bs 1/4/12).
- **G3:** With `fp8_paged_mqa_logits_torch: false`, DSV4-Flash **aggregate decode throughput exceeds the ~18 tok/s floor** (target: measurable improvement at concurrency ≥4; ideally 1.5–3×).
- **G4:** No regression to the validated baseline (MTP n=1 ~13.5 tok/s / accept len ~1.8; 12-concurrent cuda-graph path).
- **G5:** Reversible — a one-line profile + image-tag rollback restores today's known-good state.

---

## 3. Background / Current State (must-read before starting)

### 3.1 The two+one walls (live-validated 2026-06-20, GPU debug pod on GB10)
1. **Arch allowlist (precompiled host launcher):** `deep_gemm._C.so` → `attention.hpp:219` → `RuntimeError: Assertion error ... Unsupported architecture`. Allowlist is SM100/SM120 only; SM121 rejected.
2. **Compute kernel uses SM100-only ISA:** `sm100_fp8_paged_mqa_logits.cuh` uses `cute::SM100_MMA_F8F6F4_SS`, `cutlass::arch::umma_arrive`, TMEM loads (`SM100_TMEM_LOAD_*`) — none exist on GB10. (This is exactly what PR #324's `sm120_*` kernels replace.)
3. **SMEM budget (JIT metadata path):** `paged_mqa_metadata.cuh` `setup_kernel_smem_once` requested `(kMaxBatchSizeInSmem=32768 +1)*4 = 131076 B (128 KB)` via `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize)`; GB10 opt-in max = **101376 B (99 KB)** → `cudaErrorInvalidValue`. **Already mitigated** in our image: `kMaxBatchSizeInSmem` lowered to `24576` (96 KB). The metadata kernel itself is pure warp/SMEM (no tensor-core) and runs on SM121 once the SMEM fits. **Note:** PR #324 may handle this differently — reconcile (don't double-patch).

### 3.2 SGLang ALSO force-routes to torch on SM121
Even with a working DeepGEMM kernel, SGLang at startup detects SM121 via `is_sm120_supported()` (matches `device_capability_majors=[12]`) and calls `envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.set(True)`, bypassing DeepGEMM entirely. **A SGLang-side patch is required** so the native path is used when DeepGEMM advertises SM121 support (see US-006). Setting the profile flag `false` is necessary but may not be sufficient if SGLang re-forces it.

### 3.3 How DeepGEMM ships in our image
- `deep_gemm` import = **`sgl-deep-gemm` wheel** = `sgl-project/DeepGEMM` **fork** of `deepseek-ai/DeepGEMM`. Branches: main/dev/release-v0.1.x — **no sm120 branch**.
- Pulled **transitively** via the sglang install in `scitrera/cuda-containers` `container-build/Dockerfile.sglang-nightly` (the Dockerfile builds sglang+sgl-kernel from source but **does not build deep_gemm**).
- DeepGEMM is **JIT**: kernels compile at runtime from bundled source (.cuh/.hpp). The wheel ships a **lightweight `_C.so` host launcher** (this is where the `attention.hpp:219` allowlist lives). → Kernel source = patchable + JIT-rebuilt; the `_C.so` launcher = needs a **from-source rebuild**.

### 3.4 PR #324 shape
- **38 files, +6944 / −161.** Base **`deepseek-ai:nv_dev`** (NOT main, NOT our fork). From `leavelet:sm120` (55 commits). Author: "verified on sm90, sm100, sm120, ready for merge" (2026-06-07); June-12 fixes. Maintainer (jasl PR #318 thread) flagged no-hardware-to-merge; NVIDIA DevTech pushing the `nv_dev` route.
- Adds: `csrc/jit_kernels/impls/sm120_{fp8,fp4}_paged_mqa_logits.cuh`, `sm120_{fp8_fp4,fp4}_mqa_logits.cuh`, `sm120_*_gemm.{hpp,cuh}`, `sm120_tf32_hc_prenorm_gemm.*`, `sm120_utils.cuh`, `deep_gemm/include/.../sm120.cuh`; dispatch edits in `csrc/apis/attention.hpp`, `einsum.hpp`, `gemm.hpp`, `runtime.hpp`, `config.hpp`, heuristics. ~70% new .cuh (JIT), ~20% dispatch/host .hpp, ~10% tests.

### 3.5 Build & validation infra (already in place)
- **Builder:** `scripts/build_sm121_image.sh --remote-host root@spark5.local` (podman on arm64 spark5, `--no-push` available). Recipe `scripts/patches/sglang-0.5.13-sm121.recipe`. Pattern for unmerged PRs = a patch file + `dockerfile-*.patch` COPY/RUN step + recipe `APPLY_*` gate (see the existing `sglang-dsv4-nvfp4-pr25820.patch` + `dockerfile-dsv4-nvfp4.patch`).
- **Validation:** GPU debug pod (`tail -f /dev/null`, `nvidia.com/gpu:1` time-sliced, arch arm64) — see `reference_debug_pod_for_inspection`. Live endpoint: `sglang.sglang.svc.cluster.local:8000` (OpenAI API); read `Decode batch` log lines for `accept len` / `cuda graph` / `gen throughput`.
- **Baseline to beat:** ~18 tok/s aggregate @ concurrency; ~13.5 tok/s n=1 (MTP on).

---

## 4. Strategy

Port PR #324's source onto the **exact `sgl-project/DeepGEMM` commit our wheel is built from**, switch the image to **build `sgl-deep-gemm` from that patched source** (instead of the wheel), patch SGLang to honor the native path on SM121, build a **throwaway image tag**, validate the kernel standalone, then end-to-end. Keep `0.5.13-sm121` untouched until validated.

**Why port onto the fork base (not just apply #324 raw):** #324 targets `deepseek-ai:nv_dev`; applying its 6.9k-line diff onto the sgl fork (different base) will conflict. We rebase the *content* of #324 onto the fork's tree.

---

## 5. User Stories (each = one focused session)

### US-001: Pin the exact DeepGEMM source version in our image
**Description:** Determine precisely which `sgl-project/DeepGEMM` commit/tag `sgl-deep-gemm` (in image `0.5.13-sm121`) is built from, so the port targets the right base.
**Acceptance Criteria:**
- [ ] Recreate a GPU debug pod from `0.5.13-sm121`; capture `python3 -c "import deep_gemm; print(deep_gemm.__version__, deep_gemm.__file__)"`.
- [ ] `pip show sgl-deep-gemm` → version + homepage; map the wheel version → `sgl-project/DeepGEMM` tag/commit (check its release workflow / `release/v0.1.x` tags, dates align with our image build).
- [ ] Record: fork commit SHA, whether `_C.so` is in the wheel (prebuilt) vs built on install, and the bundled-source layout (`csrc/`, `deep_gemm/include/`).
- [ ] Document findings at the top of this file.

### US-002: Acquire PR #324 as a portable patch set
**Description:** Get the full #324 diff and split it into (a) additive kernel sources and (b) dispatch/host edits.
**Acceptance Criteria:**
- [ ] Fetch `https://github.com/deepseek-ai/DeepGEMM/pull/324.diff` (and the `leavelet:sm120` branch tip) locally.
- [ ] Classify each of the 38 files: NEW file (additive, low-risk) vs MODIFIED existing file (conflict-prone).
- [ ] Confirm the paged-MQA files are present: `sm120_fp8_paged_mqa_logits.cuh`, `sm120_fp4_paged_mqa_logits.cuh`, `sm120_fp8_fp4_mqa_logits.cuh`, metadata launcher.
- [ ] List the host/dispatch files that gate arch (`csrc/apis/attention.hpp`, `runtime.hpp`, `config.hpp`, heuristics) — these are the merge-risk set.

### US-003: Rebase PR #324 content onto the sgl-fork base
**Description:** Produce a single applies-cleanly patch (`deepgemm-sm120-pr324-port.patch`) against the US-001 fork commit.
**Acceptance Criteria:**
- [ ] Check out `sgl-project/DeepGEMM` @ the US-001 commit in a scratch clone.
- [ ] Drop in all NEW `sm120_*` files verbatim (additive).
- [ ] Hand-port the MODIFIED dispatch/host hunks (`attention.hpp` arch routing for paged_mqa, `runtime.hpp`/`config.hpp` heuristics, `arch_major==12` checks); resolve nv_dev-vs-fork divergences.
- [ ] Reconcile the SMEM handling with our existing `kMaxBatchSizeInSmem=24576` patch — adopt #324's approach if it caps SMEM for GB10; otherwise keep the 96 KB cap. No double-patch.
- [ ] `git diff` → `scripts/patches/deepgemm-sm120-pr324-port.patch`. `git apply --check` passes against a fresh fork checkout.

### US-004: Switch the image to build `sgl-deep-gemm` from patched source
**Description:** Add a Dockerfile stage that clones the fork @ pinned commit, applies the US-003 patch, builds the `_C.so` launcher + installs over the wheel.
**Acceptance Criteria:**
- [ ] New `scripts/patches/dockerfile-deepgemm-sm120.patch` adds: `COPY` the port patch + a `RUN` that (clone fork@SHA → `git apply` port → `pip install .` / build) AFTER the sglang install so it overrides the transitive wheel. Mirror the `dockerfile-dsv4-nvfp4.patch` pattern.
- [ ] Recipe gate `APPLY_DEEPGEMM_SM120_PR324=1` added to `sglang-0.5.13-sm121.recipe` (so it's toggleable / drops out when the wheel ships sm120).
- [ ] `build_sm121_image.sh::apply_patches` wires the gate (grep-verify the patch landed; in-container `patch --dry-run` gate).
- [ ] Verify `python -c "import deep_gemm; print(deep_gemm.__file__)"` resolves to the from-source install, and the `_C.so` arch check now includes `arch_major==12`.

### US-005: Build a throwaway validation image (do NOT overwrite 0.5.13-sm121)
**Description:** Build + (optionally) push under a distinct tag.
**Acceptance Criteria:**
- [ ] Set `IMAGE_TAG=xomoxcc/dgx-spark-sglang:0.5.13-dsv4dg-sm121` (test tag) in `build_sm121_image.sh`.
- [ ] `./scripts/build_sm121_image.sh --remote-host root@spark5.local --no-push` succeeds; build log shows the deepgemm-sm120 patch applied (dry-run + apply, no Hunk FAILED).
- [ ] Image lands in x86 local store; only push to Docker Hub after standalone kernel validation (US-007) passes.

### US-006: Patch SGLang to honor the native path on SM121
**Description:** Stop SGLang auto-forcing `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=True` on SM121 when DeepGEMM advertises support.
**Acceptance Criteria:**
- [ ] Locate the startup gate (`is_sm120_supported()` → `envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.set(True)`); the file/line is in the SGLang dsv4 indexer init.
- [ ] New `scripts/patches/sglang-dsv4-sm121-native-indexer.patch`: only force torch if DeepGEMM lacks the sm121 kernel (probe), or gate behind an env (`SGLANG_DSV4_NATIVE_INDEXER=1`) so we control it from the profile.
- [ ] Wire it into the build (Dockerfile RUN patch + recipe gate) alongside the pr25820 patch.
- [ ] Confirm at runtime: with the flag, `fp8_paged_mqa_logits` calls reach DeepGEMM (not the torch fn). (Add a temporary log/assert in the debug pod.)

### US-007: Standalone kernel validation on GB10 (gate before any redeploy)
**Description:** In the GPU debug pod on the new image, prove the native kernel runs + is correct.
**Acceptance Criteria:**
- [ ] Recreate `sglang-gpu-debug` from `0.5.13-dsv4dg-sm121`.
- [ ] Reproduce SGLang's indexer call shapes (64 heads, index_topk=512, ≤512 pages, bs 1/4/12); invoke `get_paged_mqa_logits_metadata` + `fp8_paged_mqa_logits` with `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0`.
- [ ] No `Unsupported architecture`, no `cudaErrorInvalidValue`, no tcgen05/wgmma/illegal-instruction errors; JIT compiles for `sm_121`.
- [ ] Output `allclose` vs `fp8_paged_mqa_logits_torch_sm120` (define tol; FP8/FP4 so use a generous rtol/atol + check top-k index agreement, which is what the indexer consumes).
- [ ] Record peak kernel time vs torch path for one decode step.

### US-008: End-to-end throughput validation
**Description:** Deploy the new image with the native indexer on; measure the floor.
**Acceptance Criteria:**
- [ ] Profile: `fp8_paged_mqa_logits_torch: false` + US-006 native flag; `sglang_image` → test tag; redeploy `k8s_dgx.yml --tags sglang`.
- [ ] Head reaches `2/2 Ready`, 0 crashes; warmup + first forward clean.
- [ ] Fire n=1 and 12-concurrent requests (debug pod → `sglang.sglang.svc:8000`); read `Decode batch`: `accept len`, `cuda graph: True`, `gen throughput`.
- [ ] **Compare to baseline:** aggregate @ 12 concurrent > 18 tok/s (the floor) → SUCCESS; output coherent (pattern + tail check, not just finish_reason).
- [ ] Record before/after table in this file + memory.

### US-009: Promote or roll back
**Acceptance Criteria:**
- [ ] If US-007/008 pass: re-tag/push as `0.5.13-sm121` (deliberate, confirm external push), keep profile native-on; update memory (`reference_dsv4_sm121_dsa_kernels`: floor lifted, numbers).
- [ ] If fail: leave `0.5.13-sm121` as-is, profile stays `fp8_paged_mqa_logits_torch: true`, delete the test tag; record the failure mode + next blocker in memory.
- [ ] Delete the debug pods (`reference_debug_pod_for_inspection`).

---

## 6. Functional Requirements

- **FR-1:** The build MUST install a from-source `sgl-deep-gemm` carrying PR #324's sm120/121 kernels, overriding the transitive wheel, gated by `APPLY_DEEPGEMM_SM120_PR324`.
- **FR-2:** The `_C.so` host launcher MUST accept `arch_major==12` (no `Unsupported architecture` on GB10).
- **FR-3:** `fp8_paged_mqa_logits` / `get_paged_mqa_logits_metadata` MUST JIT-compile for `sm_121` and execute without CUDA errors within GB10's 99 KB SMEM opt-in.
- **FR-4:** SGLang MUST route the indexer to DeepGEMM (not the torch fallback) when the native flag is set, controllable from the model profile.
- **FR-5:** Native output MUST be numerically equivalent to the torch fallback for the indexer's consumed result (top-k selection).
- **FR-6:** All changes MUST be toggleable (recipe gate + profile flag) and revertible to today's known-good state.
- **FR-7:** Validation MUST occur on a throwaway image tag; `0.5.13-sm121` is only overwritten after E2E success + explicit push approval.

## 7. Non-Goals

- NOT upstreaming/merging PR #324 ourselves (track it; this is a local fork-build).
- NOT porting the non-indexer sm120 kernels for their own sake (GEMM/MoE/einsum) — only what the DSV4 decode path needs; carry the rest only if required for a clean build.
- NOT touching the `hc_prenorm` (MHC) TileLang fallback unless it blocks the build (separate kernel).
- NOT changing tp/ep/context/MTP settings — those are validated and orthogonal.
- NOT a perf-tuning pass on the new kernel (heuristics/block sizes) beyond "beats the torch floor".

## 8. Technical Considerations / Risks

- **R1 (high):** PR #324 is unmerged and based on `nv_dev`; rebasing 6.9k lines onto the sgl fork may surface non-trivial conflicts in the dispatch/heuristics files. Budget real time for US-003.
- **R2 (high):** GB10 (sm_121) ≠ RTX (sm_120). #324 claims GB10 coverage but the author's verification was on sm120; GB10-specific SMEM/occupancy may still bite — US-007 is the real gate.
- **R3 (med):** SGLang's auto-force-torch (US-006) is a separate, easy-to-miss blocker — without it US-008 silently still uses torch and shows "no improvement".
- **R4 (med):** `sgl-deep-gemm` may diverge from `sgl-project/DeepGEMM` HEAD (wheel could be an older release tag); pin precisely (US-001) or the patch won't apply.
- **R5 (med):** Building deep_gemm from source lengthens the image build and may need extra CUDA/CMake deps not in the base; the `_C.so` build step is new to our pipeline.
- **R6 (low):** Even if the kernel runs, the win may be modest if the indexer isn't the *sole* bottleneck at concurrency (MoE all-reduce, etc.) — US-008 quantifies it.
- **Effort estimate:** ~1–3 focused days (US-003 + US-006 are the unknowns), plus 2–3 build/validate cycles (~30–45 min each on spark5).

## 9. Success Metrics

- DSV4-Flash aggregate decode @ 12 concurrent **> 18 tok/s** (floor lifted); stretch: ≥ 1.5–3× → 27–50+ tok/s.
- n=1 latency unchanged-or-better vs MTP baseline (~13.5 tok/s).
- Zero correctness regression (coherent output; indexer top-k matches torch within tol).
- Cleanly toggleable + revertible.

## 10. Open Questions

- Does `sgl-deep-gemm`'s `_C.so` build from source in a plain `pip install .`, or does it need a separate CMake/nvcc invocation (and which CUDA arch list)?
- Does PR #324's dispatch already emit `sm_121f` JIT targets (PR #318 explicitly "normalizes to sm_120f JIT target + sm120 include suffix") — or only `sm_120`? If sm120-only, does the JIT target need a 121→120 normalization for GB10?
- Is the metadata SMEM cap (`kMaxBatchSizeInSmem`) superseded by #324, or do we keep our 96 KB patch?
- Cheaper alternative to re-check first: has #324 (or a sibling) **already landed in `sgl-project/DeepGEMM` or the `sgl-deep-gemm` wheel** since 2026-06-20? If yes, this whole runbook collapses to a wheel/version bump — **check that before US-001.**

---

## Appendix: fast rollback

```yaml
# profile nvidia/DeepSeek-V4-Flash-NVFP4 — known-good (torch floor, ~18 tok/s)
fp8_paged_mqa_logits_torch: true
# sglang_image: pin back to the pre-experiment 0.5.13-sm121 digest
```
Redeploy `k8s_dgx.yml --tags sglang`. The torch fallback path is unconditionally correct on GB10.
