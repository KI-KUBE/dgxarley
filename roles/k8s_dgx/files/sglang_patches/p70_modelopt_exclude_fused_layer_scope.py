"""[dgxarley] Keep the layer scope of fused-tail entries in ModelOpt's is_layer_excluded.

`ModelOptQuantConfig.is_layer_excluded` (modelopt_quant.py) has a fused-module
branch for exclude entries whose last segment is one of q_a_proj / q_b_proj /
kv_a_proj_with_mqa / kv_b_proj: it compares ONLY that tail against the last
segment of the layer prefix and drops everything before it. An exact entry such
as `model.layers.45.self_attn.q_b_proj` (an MTP layer kept BF16 by the export)
therefore excludes q_b_proj in EVERY layer, and the loader dies on the first
quantized one with `param_data.shape=[4096, 1536] != loaded_weight.shape=[4096, 768]`
(bf16 param vs packed NVFP4 shard). Hit by vroomfondel/GLM-5.3-Flash-NVFP4-W4A4
on 2026-10-08 (image 0.5.21-sm121); the GLM-5.2 REAP requant hit the inverse on
layer 78 (see reference_modelopt_ignore_fused_and_mtp).

Edit (one anchor): the fused branch additionally requires the entry's head (the
part before the tail, glob-expanded) to fullmatch the prefix's head. A tail-only
entry (`q_a_proj`) keeps matching everywhere, `*.self_attn.q_a_proj` still covers
`layers.N.self_attn.fused_qkv_a_proj_with_mqa` in every layer, and
`model.layers.45.self_attn.q_b_proj` now matches layer 45 only.

DELETE WHEN upstream scopes the fused-pattern check to the entry's module path.
"""

from _patchlib import Patch, target_contains

TARGET = "sglang/srt/layers/quantization/modelopt_quant.py"

patch = Patch(
    name="ModelOpt is_layer_excluded fused-tail layer scope",
    target=TARGET,
    when=target_contains(TARGET, "fused_patterns = {"),
)

OLD = """            pattern_tail = pattern.rsplit(".", maxsplit=1)[-1]
            if pattern_tail in fused_patterns:
                for pfx in prefixes_to_check:
                    if pattern_tail in pfx.rsplit(".", maxsplit=1)[-1]:
                        return True
"""
NEW = """            pattern_head, _, pattern_tail = pattern.rpartition(".")  # dgxarley p70
            if pattern_tail in fused_patterns:
                head_regex = pattern_head.replace(".", r"\\.").replace("*", r".*")
                for pfx in prefixes_to_check:
                    pfx_head, _, pfx_tail = pfx.rpartition(".")
                    if pattern_tail not in pfx_tail:
                        continue
                    if not pattern_head or re.fullmatch(head_regex, pfx_head):
                        return True
"""


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD, NEW, marker="# dgxarley p70", what="fused-tail layer scope")
