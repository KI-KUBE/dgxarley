"""[dgxarley] qwen3_5.py: load the baked FP8 KV scales (QUALITY, mirrors the Llama-4
KV-scale patch A+B above, applied to qwen3_5).

A modelopt-NVFP4 Qwen3.5 checkpoint with quantized attention bakes per-layer FP8
KV scales (full-attn layers only: ...self_attn.k_proj.k_scale / ...v_proj.v_scale,
F32 scalars). Two gaps in qwen3_5.py drop them -> SGLang logs "Using FP8 KV cache
but no scaling factors provided. Defaulting to scaling factors of 1.0" and the
flashinfer attn backend uses 1.0 (baked scales are ~0.01-0.04 -> 25-80x off, a
real precision loss). NOT a load-blocker. Relevant only for checkpoints that
quantize attention (uniform-W4A4); NVIDIA MoE-only NVFP4 has no baked KV
scales so both edits are inert there. No model-name gate (qwen3_5.py is imported
only for this arch).
  A) RadixAttention built WITHOUT quant_config -> FP8-KV quant method never runs
     -> no k_scale/v_scale params. Fix: pass quant_config.
  B) [superseded 2026-07-24, see below] load_weights skipped
     ...k_proj.k_scale/...v_proj.v_scale via ignore_suffixes and never called
     maybe_remap_kv_scale_name. Originally fixed with a bespoke inline
     remap-and-load block inserted right after the ".self_attn" strip in every
     load_weights().

[2026-07-24] Edit B reworked into C+D below, porting sgl-project/sglang PR
#31220 after upstream review (trevor-m): a bespoke bag-on-the-side remap-and-load
helper is the wrong shape for this kind of name mapping -- SGLang already has a
declarative primitive for exactly this, `WeightsMapper` (the same type backing
model classes' `hf_to_sglang_mapper` attribute; see e.g. kimi_k25.py, which
applies its mapper to the incoming weight stream inside load_weights() the same
way C+D do here). `maybe_remap_kv_scale_name()` itself cannot be reused: its
modelopt branch requires ".self_attn."/".mixer." to still be present in the name,
but qwen3_5's load_weights() strips ".self_attn" before this point (no
"self_attn" level exists in the sglang module tree -- RadixAttention hangs
directly off the decoder layer as `.attn`), so the helper's own params_dict
membership check fails in either call position. `packed_modules_mapping` doesn't
fit either: k_scale/v_scale aren't a shard of the fused qkv_proj, they belong to
a different module (RadixAttention) -- routing them through that mapping would
feed them into the qkv shard loader, the very path that silently drops them
today. Hence: a plain WeightsMapper, applied to the weight stream at the top of
every load_weights(), same as before but via the stock mechanism instead of a
bespoke helper. Not exposed as a `hf_to_sglang_mapper` class attribute: the
SGLang loader feeds that attribute into `quant_config.apply_weight_name_mapper()`,
which mutates `ModelOptFp4Config.exclude_modules` (a side effect the Qwen3.5 VL
classes deliberately avoid by pinning `hf_to_sglang_mapper = None`, since
upstream #21234) -- a channel this patch has no reason to touch.
  C) Insert the `WeightsMapper` import + the `QWEN3_5_KV_SCALE_MAPPER` module
     constant right after the `ALL_DECODER_LAYER_TYPES` dict (stable, unique,
     appears once right before the model classes that need it).
  D) Insert `weights = QWEN3_5_KV_SCALE_MAPPER.apply(weights)` as the first line
     of EVERY load_weights() (4 occurrences in 0.5.15: the dense causal-LM, the
     MoE causal-LM, the dense VL, and the MoE VL class). The 0.5.15 image this
     patch runs against only exercises the MoE causal-LM and MoE VL classes
     end-to-end (no dense-model NVFP4-attn checkpoint to verify against), but
     applying the mapper to all four closes the same silent-drop gap for the
     dense classes too (see sgl-project/sglang#29577, a text-only Qwen3.5
     modelopt_fp4 checkpoint) at no extra cost -- the mapper is a no-op for any
     checkpoint that never emits attention k_scale/v_scale keys.

After the rename, the mapped name ("...attn.k_scale") no longer contains
"k_proj"/"v_proj", so the stacked qkv_proj shard matching further down the loop
(which used to consume and silently drop the scale) never fires, and the name
loads through the loop's existing generic weight_loader fallback branch --
RadixAttention's k_scale/v_scale carry no weight_loader
(BaseKVCacheMethod.create_weights), so default_weight_loader's scalar path
(`param.numel()==1 -> fill_`) lands the value. No custom load path needed
anymore; do NOT additionally call maybe_remap_kv_scale_name() on these names --
it would re-append ".attn" onto the already-mapped name
("...attn.attn.k_scale"), miss, and drop the scale again.
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

MARKER_C = "QWEN3_5_KV_SCALE_MAPPER = WeightsMapper("

ANCHOR_C = """ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3_5AttentionDecoderLayer,
    "linear_attention": Qwen3_5LinearDecoderLayer,
}
"""

INJECT_C = """
# [patch]  C: QWEN3_5_KV_SCALE_MAPPER -- see module docstring for the full
# rationale (ports sgl-project/sglang#31220 post-review). ModelOpt FP4
# checkpoints that quantize attention bake per-layer KV-cache scales under the
# HF attention projections ("...self_attn.k_proj.k_scale" /
# "...self_attn.v_proj.v_scale"); in the sglang module tree they live on
# RadixAttention ("...attn.k_scale" / "...attn.v_scale"). Applied to the weight
# stream in every load_weights() (edit D) BEFORE the ".self_attn" strip further
# down, so the stacked qkv_proj shard matching never sees a "k_proj"/"v_proj"
# name here and cannot consume/drop it.
from sglang.srt.models.utils import WeightsMapper

QWEN3_5_KV_SCALE_MAPPER = WeightsMapper(
    orig_to_new_substr={
        ".self_attn.k_proj.k_scale": ".attn.k_scale",
        ".self_attn.v_proj.v_scale": ".attn.v_scale",
    },
)
"""

MARKER_D = "QWEN3_5_KV_SCALE_MAPPER.apply(weights)"

OLD_D = "    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):\n"

NEW_D = OLD_D + "        weights = QWEN3_5_KV_SCALE_MAPPER.apply(weights)  # [patch]  D: kv-scale mapper\n"


@patch.run
def apply(p: Patch) -> None:
    # replace_all: the original used s.replace(old, new) with no count for both
    # edits; the real 0.5.15 file has 1 hit for A and 4 for B.
    p.replace_all(OLD_A, NEW_A, marker=MARKER_A, what="A-radixattn-quant_config")
    # replace: ALL_DECODER_LAYER_TYPES appears exactly once (verified against the
    # real 0.5.15 file), right before the model classes that need the constant.
    p.insert_after(ANCHOR_C, INJECT_C, marker=MARKER_C, what="C-kv-scale-mapper-def")
    # replace_all: 4 load_weights() defs in 0.5.15 (dense causal-LM, MoE
    # causal-LM, dense VL, MoE VL) -- verified against the real file. All four
    # must get the mapper call, not just the two MoE classes the original Edit B
    # covered, to also close the dense-model gap (see docstring).
    p.replace_all(OLD_D, NEW_D, marker=MARKER_D, what="D-kv-scale-mapper-apply-call")
