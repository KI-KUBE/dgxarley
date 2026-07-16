"""[dgxarley] modelopt_quant.py: treat "W4A16_NVFP4" (and any *NVFP4* variant) as
NVFP4 in the ModelOptMixedPrecisionConfig per-layer dispatch.

modelopt 0.44 labels weight-only NVFP4 layers as quant_algo "W4A16_NVFP4"
(4-bit NVFP4 weights, higher-precision activations), NOT plain "NVFP4". On the
OLD image this patch targeted, SGLang's ModelOptMixedPrecisionConfig.get_quant_method()
did exact `quant_algo == "NVFP4"` comparisons, so W4A16_NVFP4 experts/linears
matched NEITHER the FP8 nor the NVFP4 branch -> got NO quant method -> were built
as unquantized -> load_weights crashed on the missing w*_weight_scale_2 params
(KeyError model.layers.0.mlp.experts.w2_weight_scale_2). Normalizing the variant
string to "NVFP4" routed them to the right loader (ModelOptNvFp4FusedMoEMethod /
ModelOptFp4LinearMethod). Pairs with the _sgl_mixed_nvfp4_dispatch_ patch
(p21_mixed_nvfp4_dispatch.py). Target: nvidia/Qwen3.6-35B-A3B-NVFP4 (W4A16_NVFP4
MoE + shared-expert linears).

SELF-DISABLING as of 2026-07-15/16: newer images resolve W4A16_NVFP4 natively --
a DEDICATED dispatch branch (`quant_algo == "W4A16_NVFP4": return
ModelOptNvFp4A16LinearMethod(self.nvfp4a16_config)` for both the Linear and
FusedMoE arms, plus a separate `nvfp4a16_config` instance) that correctly keeps
W4A16 (weight-only) distinct from full NVFP4 (weight+activation). On such an
image, THIS patch's variant-collapse (`"NVFP4" in quant_algo -> quant_algo =
"NVFP4"`) fires BEFORE that dedicated branch is ever reached and silently
reroutes W4A16 layers into the wrong (full-quant) method/config -- a REGRESSION,
not a fix, on the current image. The `when=` gate below detects the native branch
(`ModelOptNvFp4A16LinearMethod` / `nvfp4a16_config` in source) and skips the
whole patch (both edits below) as an explicit no-op when present, so it only ever
applies on a genuine pre-native-W4A16 image.

NOTE on all-or-nothing vs. the original bash heredoc: the original wrote patch A
(get_quant_method dispatch) to disk even when patch B (group_size probe) drifted,
i.e. it allowed a partial A-only apply. _patchlib's contract is deliberately
all-or-nothing per file (see _patchlib.py docstring), so under this rewrite a
drift in B now rolls back A too instead of leaving a half-applied file. This is a
narrow, intentional behaviour change; see the conversion report for detail.
"""

import os

from _patchlib import DIST_PACKAGES, Patch

_TARGET = "sglang/srt/layers/quantization/modelopt_quant.py"
_TARGET_PATH = os.path.join(DIST_PACKAGES, _TARGET)


def _upstream_handles_w4a16_nvfp4_natively() -> bool:
    """True when modelopt_quant.py already has a dedicated W4A16_NVFP4 dispatch
    branch (ModelOptNvFp4A16LinearMethod / nvfp4a16_config), making this patch's
    variant-collapse a regression rather than a fix.
    """
    try:
        with open(_TARGET_PATH) as fh:
            src = fh.read()
    except OSError:
        return False
    return "ModelOptNvFp4A16LinearMethod" in src or "nvfp4a16_config" in src


patch = Patch(
    name="NVFP4-variant dispatch + group_size probe",
    target=_TARGET,
    when=not _upstream_handles_w4a16_nvfp4_natively(),
)

MARKER = "# [patch] _sgl_mixed_nvfp4_variant_"

# Patch A: normalize the resolved quant_algo in get_quant_method.
OLD_A = """        quant_algo = self._resolve_quant_algo(prefix)

        if isinstance(layer, (LinearBase, ParallelLMHead)):
"""

NEW_A = (
    """        quant_algo = self._resolve_quant_algo(prefix)
        """
    + MARKER
    + """
        # modelopt 0.44 labels weight-only NVFP4 as "W4A16_NVFP4"; the exact
        # == "NVFP4" checks below would otherwise leave these layers unquantized.
        # The checkpoint ships the full NVFP4 tensor set, so route any *NVFP4*
        # variant through ModelOptNvFp4* / ModelOptFp4LinearMethod.
        if quant_algo and "NVFP4" in quant_algo:
            quant_algo = "NVFP4"

        if isinstance(layer, (LinearBase, ParallelLMHead)):
"""
)

# Patch B: group_size probe in from_config (exact-match -> substring).
OLD_B = '            if layer_info.get("quant_algo", "").upper() == "NVFP4":\n'
NEW_B = '            if "NVFP4" in layer_info.get("quant_algo", "").upper():\n'


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD_A, NEW_A, marker=MARKER, what="get_quant_method dispatch")
    p.replace(OLD_B, NEW_B, what="group_size probe")
