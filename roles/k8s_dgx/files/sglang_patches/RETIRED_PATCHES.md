# Retired runtime patches

Breadcrumbs for patches that once lived in this directory and were deleted after
upstream fixed the underlying bug. The ConfigMap is built from a
`fileglob p[0-9][0-9]_*.py` (`roles/k8s_dgx/tasks/sglang_instance.yml`), so this
file is documentation only and is never shipped into the pod.

Reusing a retired number is fine (patches are independent, ordering only matters
inside an `NN_` prefix group), but check the entry below first so an old testlog
referencing that number stays interpretable.

## p61_flashinfer_fp4_allow.py

Retired 2026-07-28. Registered `flashinfer.quantization.fp4_quantization.fp4_quantize`
as an opaque leaf op via `torch.compiler.allow_in_graph`, so dynamo would stop
tracing into the FP4 path (which hit `posix.stat` during the JIT module lookup
and `torch.autograd.Function.__call__` inside `fp4_quantize_sm100`).

Superseded by flashinfer PR #3081, which registers a real
`@torch.library.custom_op("flashinfer::fp4_quantize")` plus a `register_fake`
implementation. Both are present in 0.6.14 and in the 0.6.15.post1 we pin
(verified against the upstream tags on 2026-07-28).

Removal was NOT a no-op cleanup. The patch's append anchor (`def fp4_quantize(`)
still matched on 0.6.15.post1, so it kept wrapping the now-custom-op-backed
function with `allow_in_graph`, which makes dynamo treat the call as opaque and
bypasses the fake registration. That is the "strictly worse than a no-op" state
described in `FLASHINFER_0.6.12_TODO.local.md`, i.e. it papers over the old
failures 2a/2b and introduces 2d (`RuntimeError when making fake tensor call`).

Full rationale and the piecewise-CUDA-graph history:
`FLASHINFER_0.6.12_TODO.local.md`, `FLASHINFER_CUDA_VERSION_SUBPROCESS_UPSTREAM_BUG.md`.

Marker strings for cleaning a container that still carries an old in-place
edit: `_fi_fp4_allow_in_graph_`, `_fi_fp4_cache_and_prewarm_`,
`_fi_fp4_prewarm_const_`, `_fi_fp4_compiler_disable_`. Since the patches only
ever ran against a fresh image layer, a pod restart on the pinned image is
enough to get an unpatched `fp4_quantization.py` back.

## p67_deepgemm_sm121_gate.py

Retired 2026-10-11. Backport of sgl-project/sglang#39482 ("[Bugfix] Include
SM121 in DeepGEMM packed-scale selection"): widened the two `sm_version == 120`
special cases in `sglang/srt/layers/deep_gemm_wrapper/configurer.py` to
`in (120, 121)`, so GB10 got the SM120 DeepGEMM capability probe in
`_compute_enable_deep_gemm()` and the UE8M0 packed-scale selection
(`DEEPGEMM_SCALE_UE8M0`). Opt-in via env `SGLANG_OPT_DEEPGEMM_SM121_PACKED_SCALE=1`
(profile key `opt_deepgemm_sm121_packed_scale`); no profile ever enabled it.

Absorbed in v0.5.21: the probe was reworked into the SM121-aware
`_sm120_deep_gemm_apis_available()` helper (#38792) and the UE8M0 line reads
`get_device_sm() in (120, 121)`. On the 0.5.21-sm121 image the patch already
self-disabled (`when=` keyed on that helper), so removal is a no-op there. The
env var, the `sglang_opt_deepgemm_sm121_packed_scale` default and its wiring in
`sglang_instance.yml` / `sglang_shard.yml` were removed with it.

An instance pinned to an image older than v0.5.21 (0.5.20-sm121) loses the
backport; restore the file from git history (`git log --diff-filter=D --
roles/k8s_dgx/files/sglang_patches/p67_deepgemm_sm121_gate.py`) if that is ever
needed. Kernel side: `sgl-deep-gemm==0.2.0` in the 0.5.21 image ships the SM120
kernel family with `arch_major == 12` dispatch (verified 2026-10-08). History:
`UPSTREAM_DSV4_BUGS.md` (09-22 to 10-11 entries),
`scripts/patches/sglang-0.5.21-sm121.recipe`.
