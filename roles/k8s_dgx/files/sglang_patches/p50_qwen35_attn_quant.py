"""[dgxarley] qwen3_5.py: allow QUANTIZED attention for uniform-NVFP4 modelopt_fp4
checkpoints (e.g. 's vroomfondel/Ornith-1.0-35B-NVFP4-ModelOpt).

qwen3_5.py hardcodes attention (both the full self_attn qkv/o_proj AND the
GatedDeltaNet linear-attn in_proj_qkvz/in_proj_ba) to UNQUANTIZED (BF16) the
moment quant_config.get_name() == "modelopt_fp4", at two byte-identical sites:
    linear_attn_quant_config = (
        None
        if quant_config and quant_config.get_name() == "modelopt_fp4"
        else quant_config)
Every NVIDIA-published Qwen3.5/3.6 NVFP4 excludes attention (MoE-only NVFP4), so
the shortcut matches THEIR modelopt_fp4 checkpoints. Distinct from the
MIXED_PRECISION patches above: nvidia/Qwen3.6 (W4A16_NVFP4 MoE + FP8 attn) is
quant_method modelopt_mixed, so this override never fires for it. But a UNIFORM
W4A4 checkpoint that ALSO quantizes attention ('s deliberately-risky nvfp4
recipe, vs the safe nvfp4_mlp_only) is modelopt_fp4 -> hits this override -> gets
a plain BF16 param for qkv_proj/in_proj_qkvz, and the merged loader then copies a
NVFP4-packed uint8 chunk (half input-dim width) into the full-width BF16 slot ->
"assert param_data.shape == loaded_weight.shape" (linear.py Merged/QKV loader).
Fix: replace the forced-None condition with False so the real quant_config passes
through and ModelOptFp4Config.is_layer_excluded() decides per-prefix (the SAME
dispatch that already keeps mlp.gate/conv1d/in_proj_a/in_proj_b BF16 for this
checkpoint). create_weights() already supports merged quantized linears; the code
was reachable, just unreached. INERT for NVIDIA MoE-only NVFP4 (excluded
attention -> UnquantizedLinearMethod), for MIXED_PRECISION (modelopt_mixed, not
modelopt_fp4), and for FP8/etc (the ternary already took the else branch). No
model-name gate: qwen3_5.py is imported only for the Qwen3.5 arch.
"""

from _patchlib import Patch

patch = Patch(name="allow quantized attention for uniform-NVFP4 modelopt_fp4", target="sglang/srt/models/qwen3_5.py")

OLD = '            if quant_config and quant_config.get_name() == "modelopt_fp4"'
NEW = "            if False  # [patch]: let is_layer_excluded() decide (allow quantized attention)"


@patch.run
def apply(p: Patch) -> None:
    # replace_all: the original used s.replace(old, new) with no count and the
    # real 0.5.15 file has 2 occurrences (verified: patching only the first left
    # the second guard live while still logging success).
    p.replace_all(OLD, NEW, what="attn-quant override")
