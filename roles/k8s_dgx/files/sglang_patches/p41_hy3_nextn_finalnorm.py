"""[dgxarley] models/hunyuan_v3_nextn.py load_weights: remap the draft head's output norm.

HYV3 NEXTN/MTP head fixes -- ONLY when EAGLE/MTP speculative decode is on.
The built-in NEXTN/MTP head (whole model.layers.80*) is BF16-EXCLUDED from NVFP4
in hf_quant_config.json (verified: layer 80 = 0 scale tensors, all BF16), but
SGLang's hunyuan_v3_nextn.py does NOT honour that exclude -- it inherits the
target's modelopt_fp4 quant_config -> FusedMoE builds a packed NVFP4 buffer
(hidden=2048) -> boot crash vs the BF16 weight (hidden=4096) in _load_w13.
--speculative-draft-model-quantization unquant does NOT help (SGLang normalizes
"unquant"->None -> re-auto-detects modelopt_fp4 from the shared checkpoint).
Two source patches. PR #30331 (this one) is MERGED since v0.5.18 and the patch
self-gates off there, see the gate below; drop the file once no pinned image
predates v0.5.18 and a NEXTN-side quant guard exists for p40 as well.

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

from _patchlib import Patch, gate_env, gate_model, target_contains

TARGET = "sglang/srt/models/hunyuan_v3_nextn.py"

# GATE (2026-09-11): upstream PR #30331 has LANDED. v0.5.18 and v0.5.19 both ship
# the `elif subname.startswith("final_layernorm")` branch this patch injects, so
# the anchor (the two-branch if/else) legitimately no longer exists and the patch
# reported ANCHOR-DRIFT where the honest answer is "nothing to do". Probe for the
# upstream branch and skip when it is there; images pinned at <= v0.5.17 still
# need the injection, which is why the edit stays rather than being deleted.
# Found by the v0.5.19 offline replay, where it drifted on the v0.5.18 CONTROL
# run too: it had gone unnoticed because that cycle's Hy3 scenario did not also
# set SGLANG_SPECULATIVE_ENABLED=true, so this patch was never evaluated.
patch = Patch(
    name="HY3 NEXTN/MTP final_layernorm -> shared_head.norm remap",
    target=TARGET,
    when=gate_model("Hy3", "Hunyuan")
    and gate_env("SGLANG_SPECULATIVE_ENABLED", "true")
    and not target_contains(TARGET, 'elif subname.startswith("final_layernorm")'),
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
