"""[dgxarley] modelopt_quant.py: CutlassMoEParams uses num_local_experts for EP.

CutlassMoEParams uses layer.num_experts (global=256) instead of
layer.num_local_experts (128 with EP=2). The cutlass_moe_fp4 forward function
then asserts num_experts == weight expert dim, which fails because weights are
EP-sliced to num_local_experts. Fix: use num_local_experts.

Upstream PR #20869 is the adjacent work and does contain this fix (plus the
input_scale slicing of p23); it then sidesteps the remaining cutlass_moe_fp4 EP
bugs by auto-routing SM120 to flashinfer_cutlass. Drop this file on an image that
ships #20869 (the already-applied guard makes it a no-op then).

Ordering: p22, p23 and this patch all edit modelopt_quant.py. The p-numbers keep
them in the order the original sglang_launch.sh applied them; do not renumber
without checking their anchors do not overlap.

Converted from two `sed -i` calls. `sed s///` without /g replaces the first match
per LINE (i.e. potentially several times per file), while p.replace() replaces
exactly the first match in the file. Verified against the 0.5.15-sm121 image that
each anchor occurs exactly once, so the two are equivalent here.

Behaviour note (deliberate): the original applied the first sed and then ran the
second unguarded, so a missing second anchor silently left the file half-patched.
_patchlib is all-or-nothing per file, so a drifted second anchor now aborts both
edits and writes nothing. That is the safer failure mode; it only differs from
the original once upstream actually moves the anchor.
"""

from _patchlib import Patch

patch = Patch(
    name="CutlassMoEParams uses num_local_experts for EP",
    target="sglang/srt/layers/quantization/modelopt_quant.py",
)

OLD_PARAMS = "num_experts=layer.num_experts,  # global num experts"
NEW_PARAMS = "num_experts=layer.num_local_experts,  # EP-aware: use local expert count"

OLD_ASSERT = "existing_params.num_experts != layer.num_experts"
NEW_ASSERT = "existing_params.num_experts != layer.num_local_experts"


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD_PARAMS, NEW_PARAMS, what="CutlassMoEParams num_local_experts EP fix")
    p.replace(OLD_ASSERT, NEW_ASSERT, what="CutlassMoEParams num_local_experts assert fix")
