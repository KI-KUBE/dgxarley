"""[dgxarley] qwen3_5.py: load the baked FP8 KV scales (QUALITY, mirrors the Llama-4
KV-scale patch A+B above, applied to qwen3_5).

A modelopt-NVFP4 Qwen3.5 checkpoint with quantized attention bakes per-layer FP8
KV scales (full-attn layers only: ...self_attn.k_proj.k_scale / ...v_proj.v_scale,
F32 scalars). Two gaps in qwen3_5.py drop them -> SGLang logs "Using FP8 KV cache
but no scaling factors provided. Defaulting to scaling factors of 1.0" and the
flashinfer attn backend uses 1.0 (baked scales are ~0.01-0.04 -> 25-80x off, a
real precision loss). NOT a load-blocker. Relevant only for checkpoints that
quantize attention ( uniform-W4A4); NVIDIA MoE-only NVFP4 has no baked KV
scales so both edits are inert there. No model-name gate (qwen3_5.py is imported
only for this arch).
  A) RadixAttention built WITHOUT quant_config -> FP8-KV quant method never runs
     -> no k_scale/v_scale params. Fix: pass quant_config.
  B) load_weights skips ...k_proj.k_scale/...v_proj.v_scale via ignore_suffixes and
     never calls maybe_remap_kv_scale_name. Fix: remap onto the attn.k_scale/
     v_scale params (which A registers) BEFORE the ignore-skip. Both needed.
"""

from _patchlib import Patch

patch = Patch(name="load baked FP8 KV scales", target="sglang/srt/models/qwen3_5.py")

MARKER_A = "# [patch]  A: RadixAttention quant_config"

OLD_A = """            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
        )"""

NEW_A = """            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,  # [patch]  A: RadixAttention quant_config
            prefix=f"{prefix}.attn",
        )"""

MARKER_B = "# [patch]  B: kv-scale remap"

OLD_B = """            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
"""

NEW_B = """            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            # [patch]  B: kv-scale remap -- load the baked FP8 KV scales.
            # qwen3_5 strips .self_attn (line above) then fuses k_proj/v_proj ->
            # qkv_proj in the stacked loop below, so the stock
            # maybe_remap_kv_scale_name (which keys off .self_attn.) never matches
            # and falls to a wrong generic branch. Map k_proj.k_scale/v_proj.v_scale
            # onto the RadixAttention attn.k_scale/attn.v_scale params that patch A
            # registers, BEFORE the fusion rename, so FP8 KV uses the calibrated
            # scale instead of the 1.0 fallback. Inert if A did not register the
            # param (remapped name not in params_dict -> name left unchanged ->
            # ignore-skipped as before).
            if name.endswith(".k_scale") or name.endswith(".v_scale"):
                _rm = name.replace(".k_proj.k_scale", ".attn.k_scale").replace(
                    ".v_proj.v_scale", ".attn.v_scale"
                )
                if _rm in params_dict:
                    # Load the scale DIRECTLY here and skip the rest of the loop. Just
                    # renaming (name = _rm) is not enough: the non-stacked else-branch
                    # below does not copy a remapped k/v_scale into its param, so the
                    # value stayed at the 1.0 fallback. Mirror the copy the Llama-4 KV
                    # patch B does explicitly.
                    _p = params_dict[_rm]
                    _wl = getattr(_p, "weight_loader", None)
                    if _wl is not None:
                        _wl(_p, loaded_weight)
                    else:
                        _p.data.copy_(loaded_weight.to(_p.dtype))
                    loaded_params.add(_rm)
                    continue
"""


@patch.run
def apply(p: Patch) -> None:
    # replace_all: the original used s.replace(old, new) with no count for both
    # edits; the real 0.5.15 file has 1 hit for A and 4 for B.
    p.replace_all(OLD_A, NEW_A, marker=MARKER_A, what="A-radixattn-quant_config")
    p.replace_all(OLD_B, NEW_B, marker=MARKER_B, what="B-kv-scale-remap")
