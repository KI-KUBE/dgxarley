"""[dgxarley] models/deepseek_nextn.py: honour the checkpoint's per-module NVFP4
exclude on the built-in MTP head (upstream blindly nulls the whole quant_config).

DeepSeek/GLM NEXTN: honour the checkpoint's per-module NVFP4 exclude on the
built-in MTP head (upstream blindly nulls the whole quant_config).
GlmMoeDsaForCausalLM (0xSero/glm-5.2-reap-504B-v2) and plain DeepSeek-V3 route
their built-in NEXTN/MTP head through deepseek_nextn.py, which UNCONDITIONALLY
nulls a modelopt_fp4 quant_config for the ENTIRE MTP decoder layer. Correct when
the whole MTP layer is BF16 (normal DeepSeek-V3), WRONG for this REAP export:
its hf_quant_config 'ignore' keeps only attn/eh_proj/gate/shared_experts of
layer N BF16 while the ROUTED EXPERTS stay NVFP4. Blindly nulling -> FusedMoE
builds a BF16 (unpacked) w13 buffer -> the NVFP4-packed checkpoint tensor
mismatches in _load_w13 ("size of tensor a (6144) must match b (3072)" = hidden
6144 unpacked vs 3072 packed, 2 fp4/byte) -> head crash-loop. The nextn decoder
is built under the 'model.decoder.*' prefix, which never matches the checkpoint's
'model.layers.N.*' exclude entries, so is_layer_excluded can't discriminate per
submodule on its own. Fix: when the checkpoint actually leaves the MTP experts
quantized, KEEP the fp4 config and add 'model.decoder.*' aliases of the layer-N
excludes -> attn/gate/shared build BF16, experts build NVFP4, exactly the mix the
main model uses (only touches build-time quant choice, load path unchanged).
Self-gated (is_layer_excluded + non-empty aliases) -> inert for all-BF16-MTP
checkpoints, so it falls back to the upstream blanket null. Note:
--speculative-draft-model-quantization=unquant does NOT reach this path
(deepseek_nextn.py never reads that flag; only glm4_moe_nextn / qwen3_*_mtp do).

[moved 2026-07-16] Was an inline `python3 - <<'PATCH_DSNEXTN_MIXED_MTP_EOF'` heredoc
inside a bash `if [ "$SGLANG_SPECULATIVE_ENABLED" = "true" ]` gate. That gate is now
`when=gate_env("SGLANG_SPECULATIVE_ENABLED", "true")`.
"""

from _patchlib import Patch, gate_env

patch = Patch(
    name="DeepSeek/GLM NEXTN: honour per-module NVFP4 exclude on MTP head (mixed-precision)",
    target="sglang/srt/models/deepseek_nextn.py",
    when=gate_env("SGLANG_SPECULATIVE_ENABLED", "true"),
)

MARKER = "# [patch] _sgl_dsnextn_mixed_mtp_"

ANCHOR = (
    '        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":\n'
    "            logger.warning(\n"
    '                "Overriding DeepseekV3ForCausalLMNextN quant config for modelopt_fp4 Deepseek model."\n'
    "            )\n"
    "            quant_config = None\n"
)

INJECT = (
    "        " + MARKER + "  # honour per-module NVFP4 exclude on the MTP head\n"
    "        _dsnextn_kept_fp4 = False\n"
    '        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":\n'
    '            _mtp_experts = f"model.layers.{config.num_hidden_layers}.mlp.experts"\n'
    '            _excl = getattr(quant_config, "exclude_modules", None)\n'
    '            _tag = f".layers.{config.num_hidden_layers}."\n'
    "            if (\n"
    "                isinstance(_excl, list)\n"
    '                and hasattr(quant_config, "is_layer_excluded")\n'
    "                and not quant_config.is_layer_excluded(_mtp_experts)\n"
    "            ):\n"
    "                _aliases = [\n"
    '                    e.replace(_tag, ".decoder.")\n'
    "                    for e in _excl\n"
    '                    if _tag in e and e.replace(_tag, ".decoder.") not in _excl\n'
    "                ]\n"
    "                if _aliases:\n"
    "                    quant_config.exclude_modules = _excl + _aliases\n"
    "                    _dsnextn_kept_fp4 = True\n"
    "                    logger.warning(\n"
    '                        "NextN modelopt_fp4: checkpoint keeps the MTP experts quantized; "\n'
    '                        "aliasing %d layer-%d excludes to model.decoder.* so attn/gate/shared "\n'
    '                        "stay BF16 while experts stay NVFP4.",\n'
    "                        len(_aliases),\n"
    "                        config.num_hidden_layers,\n"
    "                    )\n"
    "            if not _dsnextn_kept_fp4:\n"
    "                logger.warning(\n"
    '                    "Overriding DeepseekV3ForCausalLMNextN quant config for modelopt_fp4 Deepseek model."\n'
    "                )\n"
    "                quant_config = None\n"
)


@patch.run
def apply(p: Patch) -> None:
    p.replace(
        ANCHOR,
        INJECT,
        marker=MARKER,
        what="DeepSeek/GLM NEXTN mixed-precision MTP quant override",
    )
