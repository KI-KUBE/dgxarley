"""[dgxarley] models/hunyuan_v3.py: remap shared-expert weight names (.shared_experts. -> .shared_mlp.).

HYV3 checkpoints (vroomfondel/Hy3-NVFP4-W4A4, tencent/Hy3) name the shared
expert `model.layers.N.mlp.shared_experts.*`, but SGLang's HYV3 model module
is `shared_mlp` (self.shared_mlp = HYV3FeedForward). load_weights has NO remap
for it (only router.gate -> gate), so the shared-expert weights -- which ARE
present and FP4-quantized -- are silently skipped (`if name not in params_dict:
continue`) -> shared_mlp stays zero-init -> gate_up_proj outputs 0 -> down_proj
FP4-quantizes a zero input -> scale 448*6/0 degenerates -> NaN at layer 1, first
forward. Localised via --debug-tensor-dump layer tracing (see QUANT_HY3_GOTCHAS).
Fix: remap .shared_experts. -> .shared_mlp. at the TOP of the load loop, so the
existing gate_proj/up_proj -> gate_up_proj stacking then applies correctly.

This is an UNRELATED bug from the token-suffix backport in p62/p63
(function_call/hunyuan_detector.py, parser/reasoning_parser.py) -- it just
happens to be colocated under the same Hy3/Hunyuan model gate in the old
script for convenience, since both only matter for Hy3/HYV3 checkpoints.

Verified against the 0.5.15-sm121 pristine source: the `.shared_experts.` ->
`.shared_mlp.` remap is NOT present natively (unlike p62/p63's
resolve_hunyuan_tokens, which upstream already ships) -- this patch still
applies on the current image.

[moved 2026-07-16] Was the inline `python3 - <<'PATCH_HUNYUAN_SHARED_EOF'`
heredoc inside a bash `if [[ $SGLANG_MODEL == *Hy3* || $SGLANG_MODEL == *Hunyuan*
|| $SGLANG_TOOL_CALL_PARSER == "hunyuan" || $SGLANG_REASONING_PARSER == "hunyuan"
]]` gate (shared with p62/p63). That gate is now
`when=gate_model("Hy3", "Hunyuan") or gate_env(...)` below.
"""

from _patchlib import Patch, gate_env, gate_model

patch = Patch(
    name="HYV3 shared-expert weight-name remap (.shared_experts. -> .shared_mlp.)",
    target="sglang/srt/models/hunyuan_v3.py",
    when=gate_model("Hy3", "Hunyuan")
    or gate_env("SGLANG_TOOL_CALL_PARSER", "hunyuan")
    or gate_env("SGLANG_REASONING_PARSER", "hunyuan"),
)

MARKER = 'replace(".shared_experts.", ".shared_mlp.")'

ANCHOR = "        for name, loaded_weight in weights:\n"

INJECT = (
    "            # [dgxarley] HYV3 checkpoints name the shared expert\n"
    "            # `mlp.shared_experts.*`; the SGLang model module is `shared_mlp`.\n"
    "            # Remap so the (real, FP4) shared-expert weights actually load —\n"
    "            # else silently skipped -> shared_mlp zero-init -> NaN at down_proj.\n"
    '            name = name.replace(".shared_experts.", ".shared_mlp.")\n'
)


@patch.run
def apply(p: Patch) -> None:
    p.insert_after(ANCHOR, INJECT, marker=MARKER, what="HYV3 shared-expert weight-name remap")
