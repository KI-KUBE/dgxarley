"""[dgxarley] model_config.py: route NVFP4-bearing MIXED_PRECISION modelopt
checkpoints to the modelopt_mixed loader regardless of architecture.

SGLang 0.5.12/0.5.13's _parse_modelopt_quant_config() (configs/model_config.py,
this function is byte-identical across both releases) maps a
`quant_algo: MIXED_PRECISION` checkpoint to a quant method by ARCHITECTURE:
NemotronH -> modelopt_mixed, everything else -> w4afp8. For
Qwen3_5MoeForConditionalGeneration (nvidia/Qwen3.6-35B-A3B-NVFP4: W4A16_NVFP4 MoE
experts + FP8 linear_attn, group_size 16) that fall-through is WRONG:
  * quantization=modelopt_fp4/modelopt_mixed -> _verify_quantization rejects the
    mismatch (detected w4afp8 not in compatible[modelopt_*]=["modelopt"]).
  * quantization=w4afp8 passes verify but the W4AFp8Config loader is the W4A8
    (int4 + per-group fp8-scale, default group_size 128) path -- the wrong
    encoding for NVFP4 (e2m1) experts.
The ModelOptMixedPrecisionConfig loader DOES compose per-layer NVFP4+FP8 correctly
but is unreachable for this arch (its override_quantization_method only fires once
detection already returned modelopt_mixed). Fix: also route to modelopt_mixed when
any quantized layer is NVFP4 -- discriminate on the real checkpoint format, not the
arch name. NemotronH path preserved; pure-W4A8 MIXED_PRECISION still -> w4afp8.
Pairs with quantization: modelopt_mixed in model_profiles/nvidia-qwen3.6-35b-a3b-nvfp4.yml.

RE-CHECKED 2026-07-16: upstream's _parse_modelopt_quant_config() now discriminates
MIXED_PRECISION checkpoints by scanning quantized_layers for NVFP4/W4A16_NVFP4
entries (has_modelopt_nvfp4_layers) and routes to modelopt_mixed generically -- the
NemotronH-only arch-name gate this patch targeted is GONE, replaced by exactly the
real-format discrimination this patch wanted (verified live, same semantics as our
own _has_nvfp4 check below, just upstream-native now). OBSOLETE: self-disable
instead of forcing a stale re-anchor onto code upstream already solved better.

The self-disable is expressed via the `when=` gate below: if `has_modelopt_nvfp4_layers`
is already present in the shipped model_config.py, the patch is skipped up front
(reported by _patchlib as a normal gate-skip line, not as an ANCHOR-DRIFT -- there is
nothing to re-check an anchor against once upstream has solved this natively).
"""

import os

from _patchlib import DIST_PACKAGES, Patch

_TARGET = "sglang/srt/configs/model_config.py"
_TARGET_PATH = os.path.join(DIST_PACKAGES, _TARGET)


def _upstream_dispatches_nvfp4_natively() -> bool:
    """True when model_config.py already discriminates MIXED_PRECISION NVFP4 vs
    w4afp8 natively (has_modelopt_nvfp4_layers), making this patch obsolete.
    """
    try:
        with open(_TARGET_PATH) as fh:
            return "has_modelopt_nvfp4_layers" in fh.read()
    except OSError:
        return False


patch = Patch(
    name="MIXED_PRECISION NVFP4 -> modelopt_mixed dispatch",
    target=_TARGET,
    when=not _upstream_dispatches_nvfp4_natively(),
)

MARKER = "# [patch] _sgl_mixed_nvfp4_dispatch_"

OLD = """        if quant_algo == "MIXED_PRECISION":
            architectures = getattr(self.hf_config, "architectures", []) or []
            if getattr(self.hf_config, "model_type", None) == "nemotron_h" or any(
                arch.startswith("NemotronH") for arch in architectures
            ):
                return {"quant_method": "modelopt_mixed", "quant_algo": quant_algo}
            return {"quant_method": "w4afp8", "quant_algo": quant_algo}
"""

NEW = (
    """        if quant_algo == "MIXED_PRECISION":
            architectures = getattr(self.hf_config, "architectures", []) or []
            """
    + MARKER
    + """
            # Route NVFP4-bearing MIXED_PRECISION checkpoints (e.g.
            # nvidia/Qwen3.6-35B-A3B-NVFP4: W4A16_NVFP4 MoE + FP8 attn) to the
            # modelopt_mixed loader. Upstream gates modelopt_mixed to NemotronH
            # only; other archs fall through to w4afp8 (W4A8 int4), the wrong
            # weight format for NVFP4 experts. Discriminate on real format.
            _qlayers = json_quant_configs.get("quantized_layers", {}) or {}
            _has_nvfp4 = any(
                "NVFP4" in str(_li.get("quant_algo", "")).upper()
                for _li in _qlayers.values()
                if isinstance(_li, dict)
            )
            if (
                getattr(self.hf_config, "model_type", None) == "nemotron_h"
                or any(arch.startswith("NemotronH") for arch in architectures)
                or _has_nvfp4
            ):
                return {"quant_method": "modelopt_mixed", "quant_algo": quant_algo}
            return {"quant_method": "w4afp8", "quant_algo": quant_algo}
"""
)


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD, NEW, marker=MARKER)
