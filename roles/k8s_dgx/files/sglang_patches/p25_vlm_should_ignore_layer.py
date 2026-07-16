"""[dgxarley] compressed_tensors/utils.py: keep VLM vision tower + multimodal
projector in BF16 for compressed-tensors NVFP4 checkpoints (e.g. Mistral3/Pixtral
such as RecViking/Mistral-Medium-3.5-128B-NVFP4).

These checkpoints quantize ONLY the text transformer and list the entire
vision_tower + multi_modal_projector in the `ignore` list (-> BF16). But
should_ignore_layer() compares the checkpoint `ignore` names against the RUNTIME
module names, and SGLang's Mistral3/Pixtral loader renames four families on load:
  q/k/v_proj            -> qkv_proj         (fusion; pixtral.py stacked_params_mapping)
  gate/up_proj          -> gate_up_proj     (fusion)
  attention.o_proj      -> attention.proj   (rename; pixtral.py)
  model.vision_tower.X          -> vision_tower.X           (model. strip; mistral.py/llava.py)
  model.multi_modal_projector.X -> multi_modal_projector.X  (model. strip; mistral.py/llava.py)
(the inner Mistral3/Llava model mounts vision_tower + projector WITHOUT the
 checkpoint's leading "model." -- so the runtime prefix must be toggled to match)
AND Pixtral/Mistral3ForConditionalGeneration define NO packed_modules_mapping, so
the fused-decomposition branch never runs (fused_mapping is empty). Net effect:
the vision tower is quantized to NVFP4 and crashes at warmup:
  ValueError: mm_fp4 accepts 2d tensors, got torch.Size([1, 16, 832]) ...
Upstream PR #24929 (OPEN) only fixes the unrelated ".linear" suffix variant.

Fix: wrap should_ignore_layer to (1) inject the canonical fused mapping so the
vision qkv_proj/gate_up_proj decompose to the checkpoint's unfused names, and
(2) also test the inverse-remapped checkpoint-namespace candidates (model.-prefix
for the projector, o_proj for attention.proj). The wrapper appends to utils.py so
the `from .utils import should_ignore_layer` in compressed_tensors.py (executed
later) binds to the patched function. The try/except keeps non-VLM compressed-
tensors models (asymmetric fused-shard ignore) on the original behaviour.
Verified at the matcher level against the real 340-entry ignore list (all vision +
projector linears -> BF16, all text linears -> quantized).

Implementation note: the original inline heredoc unconditionally appended the
wrapper block to the end of the file (`p.write_text(src + block)`), gated only on
`should_ignore_layer` existing somewhere in the source (not on being adjacent to
any particular anchor). `_patchlib.insert_after` needs a literal anchor to insert
after, so this patch anchors on the verbatim body of `_match_fused_layer` -- the
last function in utils.py -- which reproduces the exact same byte position (true
EOF) as the original append. If a future SGLang version reorders utils.py so
`_match_fused_layer` is no longer last, this patch will correctly ANCHOR-DRIFT
rather than silently inserting mid-file.
"""

from _patchlib import Patch

patch = Patch(
    name="VLM vision/projector ignore-list fix",
    target="sglang/srt/layers/quantization/compressed_tensors/utils.py",
)

# Verbatim body of `_match_fused_layer`, the last function in utils.py at the time
# of writing -- used purely as the "true EOF" anchor (see docstring above).
ANCHOR = '''def _match_fused_layer(
    layer_name: str,
    target_layers: Iterable[str],
    fused_mapping: Mapping[str, List[str]],
) -> Optional[str]:
    """
    Match a fused layer name to its corresponding individual layer in
    target_layers. Returns first value in fused_mapping which matches targets

    Implements an "all" matching strategy where a fused layer matches iff
    "all" of its components match

    :param layer_name: layer name
    :param target_layers: list of targets to match the layer against
    :param fused_mapping: map from fused layer names to its components

    Examples:
        layer_name = "model.layers.0.self_attn.qkv_proj"
        target_layers = ["model.layers.0.self_attn.q_proj",
                        "model.layers.0.self_attn.k_proj",
                        "model.layers.0.self_attn.v_proj"]
    """
    # find layer_name in mapping
    fused = next((key for key in fused_mapping if layer_name.endswith(key)), None)
    if fused is None:
        return None

    # expand path of unfused components
    unfused_paths = [
        layer_name.replace(fused, unfused) for unfused in fused_mapping[fused]
    ]

    # for each unfused component, find a match in targets
    unfused_matches: List[Optional[str]] = []
    for unfused in unfused_paths:
        for target in target_layers:
            if _is_equal_or_regex_match(unfused, target):
                unfused_matches.append(target)
                break
        else:
            unfused_matches.append(None)

    return unfused_matches[0] if all(unfused_matches) else None
'''

BLOCK = """

# [patch] _sgl_vlm_ignore_fix — keep VLM vision tower + multimodal projector in BF16.
# See sglang_launch.sh for the full rationale (4 weight-loader name remaps + missing
# packed_modules_mapping cause compressed-tensors NVFP4 VLMs to quantize the vision
# tower -> "mm_fp4 accepts 2d tensors" crash). Wrap should_ignore_layer to inject the
# canonical fused mapping and test inverse-remapped checkpoint-namespace candidates.
_sgl_orig_should_ignore_layer = should_ignore_layer
_sgl_vlm_fused_default = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _sgl_vlm_ignore_candidates(rt):
    base = {rt}
    # SGLang's Mistral3/Llava loader strips a leading "model." from the inner
    # multimodal submodels (vision_tower, multi_modal_projector) — they are mounted
    # at "vision_tower." / "multi_modal_projector." (no "model.") while the checkpoint
    # ignore list keeps the "model." prefix. Toggle it so names line up either way.
    if rt.startswith("model."):
        base.add(rt[len("model."):])
    else:
        base.add("model." + rt)
    names = set(base)
    for n in base:
        if n.endswith(".attention.proj"):  # vision attn output: runtime proj <- ckpt o_proj
            names.add(n[: -len(".attention.proj")] + ".attention.o_proj")
    return names


def should_ignore_layer(layer_name, ignore=tuple(), fused_mapping=None):
    base_fm = fused_mapping if fused_mapping else {}
    fm = dict(base_fm)
    if layer_name:
        _proj = layer_name.rsplit(".", 1)[-1]
        if _proj in _sgl_vlm_fused_default and _proj not in fm:
            fm[_proj] = _sgl_vlm_fused_default[_proj]
    try:
        return any(
            _sgl_orig_should_ignore_layer(c, ignore=ignore, fused_mapping=fm)
            for c in _sgl_vlm_ignore_candidates(layer_name or "")
        )
    except ValueError:  # mixed-scheme fused group with injected mapping -> defer to original
        return _sgl_orig_should_ignore_layer(
            layer_name, ignore=ignore, fused_mapping=base_fm
        )
"""


@patch.run
def apply(p: Patch) -> None:
    p.insert_after(
        ANCHOR, BLOCK, marker="# [patch] _sgl_vlm_ignore_fix", what="_sgl_vlm_ignore_fix should_ignore_layer"
    )
