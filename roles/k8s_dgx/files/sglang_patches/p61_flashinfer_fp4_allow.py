"""[dgxarley] flashinfer/quantization/fp4_quantization.py: register `fp4_quantize` as
an opaque leaf op via `torch.compiler.allow_in_graph`, so dynamo emits a
single FX graph node for the call without tracing its body.

Symptom chain (GLM-4.7-NVFP4 EP=1, piecewise CUDA graphs):
  sglang/srt/compilation/compile.py:183 _ensure_compiled
  -> torch._dynamo compile_wrapper
  -> sglang/.../modelopt_quant.py:1482 fp4_quantize(x, layer.input_scale_inv)
  -> flashinfer/quantization/fp4_quantization.py:700 fp4_quantize

Dynamo hits a whole family of un-traceable things as it recurses into the
FP4 path:
  (a) get_fp4_quantization_module -> JitSpec.is_aot -> Path.exists -> os.stat
      -> "Attempted to call function marked as skipped: posix.stat"
  (b) fp4_quantize_sm100 (line 222) -> `module.fp4_quantize(...)` is a
      torch.autograd.Function -> "Unsupported method call: Function.__call__"
  (c) almost certainly more further in.

Whack-a-mole fixes (functools.cache wrapping, pre-resolving the module
lookup into a module-level constant, etc.) each unblock one layer and
then hit the next. `@torch.compiler.disable` also doesn't work: sglang's
piecewise compile path treats skipped calls as a hard error (gb0098:
"Skip calling torch.compiler.disable()d function") rather than a normal
graph break.

Correct approach: `torch.compiler.allow_in_graph`. This tells dynamo to
emit a single opaque call node for `fp4_quantize` in the FX graph without
tracing through its body at all -- the JIT lookup, os.stat, and
autograd.Function.__call__ all execute at real runtime (outside any
trace), which is completely fine for all of them.

Contract for allow_in_graph: the function must take/return tensors
(or pytrees of tensors) and must be deterministic in output dtype/shape
given input dtype/shape. `fp4_quantize(input, global_scale)` returns
(x_q, sf) -- both tensors -- and its output shapes are a deterministic
function of input shape + sf_vec_size. Contract satisfied.

This patches flashinfer, NOT sglang -- target is
`flashinfer/quantization/fp4_quantization.py` under DIST_PACKAGES.

No model gate, no env gate: unconditional, same as the original heredoc.

Not a plain single-anchor swap, so this bypasses `Patch.replace()` for two of
its steps and writes through the public `p.code` buffer (see `apply()`):

* Undo remnants from earlier patch revisions in the same container (e.g. a
  crash loop re-execs sglang_launch.sh without a fresh image): rewrite a stale
  `_SGLANG_FP4_MOD.fp4_quantize_sm100(` call back to the live lookup, and strip
  any leftover appended blocks from now-retired marker generations
  (`_fi_fp4_cache_and_prewarm_`, `_fi_fp4_prewarm_const_`,
  `_fi_fp4_compiler_disable_`). Both are best-effort: absent is the common
  case, and the original script's plain `str.replace`/`str.find` never treated
  "not found" as an error here, so `Patch.replace()` (which raises on a
  missing anchor) would be the wrong tool -- it would misreport a clean,
  already-migrated file as anchor-drifted.
* Remove the `@torch.compiler.disable` decorator above `def fp4_quantize(`,
  again via a tolerant replace (no-op if already removed by a prior run).
* Only the final step -- appending the `allow_in_graph` registration block --
  uses the MARKER already-applied guard and can raise AnchorDrift (if
  `def fp4_quantize(` itself has gone missing, i.e. real SGLang/flashinfer
  version drift).
"""

from _patchlib import AnchorDrift, Patch

patch = Patch(
    name="fp4_quantize -> torch.compiler.allow_in_graph (opaque leaf op for dynamo)",
    target="flashinfer/quantization/fp4_quantization.py",
)

MARKER = "# [patch] _fi_fp4_allow_in_graph_"

# Undo any remnants from earlier patch revisions in the same container
# (e.g. after a crash loop re-execs sglang_launch.sh without a fresh
# image) so the source stays clean.
STALE_CONST_CALL = "_SGLANG_FP4_MOD.fp4_quantize_sm100("
STALE_CONST_REPLACEMENT = 'get_fp4_quantization_module(f"{major}{minor}").fp4_quantize_sm100('

OLD_DECORATOR = "@torch.compiler.disable\n@flashinfer_api\ndef fp4_quantize(\n"
NEW_DECORATOR = "@flashinfer_api\ndef fp4_quantize(\n"

STALE_MARKERS = (
    "# [patch] _fi_fp4_cache_and_prewarm_",
    "# [patch] _fi_fp4_prewarm_const_",
    "# [patch] _fi_fp4_compiler_disable_",
)

APPEND_BLOCK = (
    "\n\n"
    "# " + MARKER + "\n"
    "# Appended by sglang_launch.sh runtime patch. Registers fp4_quantize\n"
    "# as an opaque leaf op so dynamo emits a single FX graph node for it\n"
    "# instead of tracing into the body (which hits os.stat during the JIT\n"
    "# lookup and torch.autograd.Function.__call__ inside fp4_quantize_sm100).\n"
    "# See sglang_launch.sh header for full rationale.\n"
    "try:\n"
    "    import torch as _sglang_t\n"
    "    fp4_quantize = _sglang_t.compiler.allow_in_graph(fp4_quantize)\n"
    "    import sys as _sglang_sys\n"
    "    print('[fp4_quantization] fp4_quantize registered via allow_in_graph', file=_sglang_sys.stderr)\n"
    "except Exception as _sglang_e:\n"
    "    import sys as _sglang_sys\n"
    "    print(f'[fp4_quantization] allow_in_graph registration failed: {_sglang_e}', file=_sglang_sys.stderr)\n"
)


@patch.run
def apply(p: Patch) -> None:
    # These steps are deliberately tolerant (absence is the steady state, not
    # drift), and they replace EVERY occurrence rather than just the first, so
    # they go through the `code` buffer rather than p.replace()/replace_optional().
    # The `code` setter marks the patch changed only if the text actually differs.
    src = p.code

    if STALE_CONST_CALL in src:
        src = src.replace(STALE_CONST_CALL, STALE_CONST_REPLACEMENT)

    src = src.replace(OLD_DECORATOR, NEW_DECORATOR)

    for old_marker in STALE_MARKERS:
        idx = src.find("\n\n# " + old_marker)
        if idx != -1:
            src = src[:idx].rstrip() + "\n"

    if MARKER in src:
        p.code = src
        return

    if "def fp4_quantize(" not in src:
        raise AnchorDrift("fp4_quantize definition missing")

    p.code = src + APPEND_BLOCK
