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
