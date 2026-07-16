"""[dgxarley] modelopt_quant.py: EP-aware input_scale slicing (SGLang 0.5.9 bug).

In process_weights_after_loading(), the fallback else-branch computes
w13_input_scale and w2_input_scale with shape (num_experts,) but multiplies
them with w13_weight_scale_2 / w2_weight_scale_2 which are (num_local_experts,).
With EP=2 on MiniMax-M2.5 (256 experts): (256,) * (128,) -> RuntimeError.
The cutedsl branch has _slice_scale() but the else-branch is missing it.
Fix: slice input_scale to local experts in the else-branch.

The original bash guard also grep'd the file for the (unchanged-by-this-patch)
`w13_input_scale.max(dim=-1)` substring before even attempting the edit, as a
coarse "does this function still look like this" pre-check, printing a
dedicated "outer probe missing" ANCHOR-DRIFT line on failure. That grep target
is a strict subset of the `old` anchor `_patchlib.Patch.replace()` already
checks below (with a more precise error), so it is redundant here and dropped;
the behaviour (skip + one ANCHOR-DRIFT line when the anchor is gone) is
unchanged, only the exact wording differs.
"""

from _patchlib import Patch

patch = Patch(name="EP-aware input_scale slicing", target="sglang/srt/layers/quantization/modelopt_quant.py")

MARKER = "# EP-aware slicing: input_scale has shape"

OLD = """        else:
            w13_input_scale = layer.w13_input_scale.max(dim=-1).values.to(torch.float32)
            w2_input_scale = layer.w2_input_scale"""

NEW = (
    """        else:
            w13_input_scale = layer.w13_input_scale.max(dim=-1).values.to(torch.float32)
            w2_input_scale = layer.w2_input_scale
            """
    + MARKER
    + """ (num_experts,) but must match
            # weight_scale_2 which is (num_local_experts,). No-op when ep_size=1.
            if layer.moe_ep_size > 1:
                _ep_start = layer.moe_ep_rank * layer.num_local_experts
                _ep_end = _ep_start + layer.num_local_experts
                w13_input_scale = w13_input_scale[_ep_start:_ep_end]
                w2_input_scale = w2_input_scale[_ep_start:_ep_end]"""
)


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD, NEW, marker=MARKER)
