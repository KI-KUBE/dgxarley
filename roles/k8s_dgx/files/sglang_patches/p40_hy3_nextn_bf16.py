"""[dgxarley] models/hunyuan_v3_nextn.py: force the NEXTN/MTP head UNQUANTIZED (BF16).

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

This is patch 1 of 2 (see p41_hy3_nextn_finalnorm.py for patch 2 -- same file,
must run after this one, hence the p40 < p41 filename ordering):

1) hunyuan_v3_nextn.py: force the NEXTN/MTP head UNQUANTIZED (BF16).
   Null quant_config at the HYV3ForCausalLMNextN.__init__ top so it covers the whole
   draft (decoder layer-80 experts + shared_mlp + attention + lm_head -- ALL BF16-
   excluded). Same guard glm4_moe_nextn.py / qwen3_5_mtp.py already carry; HYV3 lacks it.

[moved 2026-07-16] Was an inline `python3 - <<'PATCH_HY3_NEXTN_BF16_EOF'` heredoc
inside a bash `if { [[ $SGLANG_MODEL == *Hy3* ]] || [[ $SGLANG_MODEL == *Hunyuan* ]]; }
&& [ "$SGLANG_SPECULATIVE_ENABLED" = "true" ]` gate, shared with p41. That gate is now
`when=gate_model("Hy3", "Hunyuan") and gate_env("SGLANG_SPECULATIVE_ENABLED", "true")`.
"""

from _patchlib import Patch, gate_env, gate_model

patch = Patch(
    name="HY3 NEXTN/MTP head forced unquantized (BF16)",
    target="sglang/srt/models/hunyuan_v3_nextn.py",
    when=gate_model("Hy3", "Hunyuan") and gate_env("SGLANG_SPECULATIVE_ENABLED", "true"),
)

MARKER = "# [patch] _sgl_hy3_nextn_bf16_head_"

ANCHOR = (
    "        nn.Module.__init__(self)\n" "        self.config = config\n" "        self.quant_config = quant_config\n"
)

INJECT = (
    "        nn.Module.__init__(self)\n"
    "        self.config = config\n"
    "        " + MARKER + "\n"
    "        # layer-80 (NEXTN/MTP head) is BF16-excluded in hf_quant_config.json;\n"
    "        # drop the target's NVFP4 quant so create_weights allocates BF16 buffers.\n"
    "        if quant_config is not None and quant_config.get_name() in (\n"
    '            "modelopt_fp4",\n'
    '            "modelopt_mixed",\n'
    "        ):\n"
    "            quant_config = None\n"
    "        self.quant_config = quant_config\n"
)


@patch.run
def apply(p: Patch) -> None:
    p.replace(ANCHOR, INJECT, marker=MARKER, what="HY3 NEXTN/MTP head forced unquantized (BF16)")
