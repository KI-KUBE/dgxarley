"""[dgxarley] models/hunyuan_v3_nextn.py load_weights: remap the draft head's output norm.

HYV3 NEXTN/MTP head fixes -- ONLY when EAGLE/MTP speculative decode is on.
The built-in NEXTN/MTP head (whole model.layers.80*) is BF16-EXCLUDED from NVFP4
in hf_quant_config.json (verified: layer 80 = 0 scale tensors, all BF16), but
SGLang's hunyuan_v3_nextn.py does NOT honour that exclude -- it inherits the
target's modelopt_fp4 quant_config -> FusedMoE builds a packed NVFP4 buffer
(hidden=2048) -> boot crash vs the BF16 weight (hidden=4096) in _load_w13.
--speculative-draft-model-quantization unquant does NOT help (SGLang normalizes
"unquant"->None -> re-auto-detects modelopt_fp4 from the shared checkpoint).
Two source patches (neither upstream-merged as of 0.5.14). Drop on an image that
ships PR #30331 + a NEXTN-side quant guard for hunyuan_v3_nextn.py.

This is patch 2 of 2 (see p40_hy3_nextn_bf16.py for patch 1 -- same file, this one
must run after that one, hence the p40 < p41 filename ordering):

2) hunyuan_v3_nextn.py load_weights: remap the draft head's output norm.
   Checkpoint stores it as model.layers.80.final_layernorm.weight; the module is
   model.shared_head.norm. Without this it falls into the generic else ->
   model.decoder.final_layernorm.weight (no such param) -> silently dropped ->
   shared_head.norm stays default-init -> accept-rate collapses. Upstream PR #30331.

[moved 2026-07-16] Was an inline `python3 - <<'PATCH_HY3_NEXTN_FINALNORM_EOF'` heredoc
inside the same bash `if { [[ $SGLANG_MODEL == *Hy3* ]] || [[ $SGLANG_MODEL == *Hunyuan* ]]; }
&& [ "$SGLANG_SPECULATIVE_ENABLED" = "true" ]` gate as p40. That gate is now
`when=gate_model("Hy3", "Hunyuan") and gate_env("SGLANG_SPECULATIVE_ENABLED", "true")`.
"""

from _patchlib import Patch, gate_env, gate_model

patch = Patch(
    name="HY3 NEXTN/MTP final_layernorm -> shared_head.norm remap",
    target="sglang/srt/models/hunyuan_v3_nextn.py",
    when=gate_model("Hy3", "Hunyuan") and gate_env("SGLANG_SPECULATIVE_ENABLED", "true"),
)

MARKER = "# [patch] _sgl_hy3_nextn_final_layernorm_"

ANCHOR = (
    "                if any(subname.startswith(s) for s in spec_weight_names):\n"
    '                    name = f"model.{subname}"\n'
    "                else:\n"
    '                    name = f"model.decoder.{subname}"\n'
)

INJECT = (
    "                if any(subname.startswith(s) for s in spec_weight_names):\n"
    '                    name = f"model.{subname}"\n'
    '                elif subname.startswith("final_layernorm"):\n'
    "                    " + MARKER + "  # upstream PR #30331\n"
    '                    name = "model.shared_head.norm.weight"\n'
    "                else:\n"
    '                    name = f"model.decoder.{subname}"\n'
)


@patch.run
def apply(p: Patch) -> None:
    p.replace(ANCHOR, INJECT, marker=MARKER, what="HY3 NEXTN/MTP final_layernorm -> shared_head.norm remap")
