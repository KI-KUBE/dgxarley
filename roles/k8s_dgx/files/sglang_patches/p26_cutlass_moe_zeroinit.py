"""[dgxarley] cutlass_moe.py: cutlass_moe_fp4 EP correctness fix. Two independent
bugs in the same function, both triggered by `topk_ids == -1` non-local
sentinels that `StandardDispatcher.local_expert_mapping` writes under EP>1.

BUG 1: a_map / c_map allocated with torch.empty (uninitialized). The native
kernel prepare_moe_input.compute_arg_sorts iterates blockIdx.x over
[0, num_experts) and only writes a_map[slot] / c_map[slot] where
topk_ids[i] == blockIdx.x. The -1 entries match no block and leave those
slots with torch.empty garbage. Downstream a.index_select(0, a_map) in
_shuffle_rows_torch reads the garbage as row indices and trips torch's
vectorized_gather_kernel bounds check — surfaces as device-side assert at
nvfp4_blockwise_moe.cuh:78 (via next cudaMallocAsync sync point).
Fix 1: zero-init. Zero is always a valid row index into `a`.

BUG 2: topk_weights are NOT zeroed for -1 slots. Fix 1 alone eliminates the
crash but produces garbage output. The dispatcher remaps topk_ids to local
IDs + -1 sentinels, but leaves topk_weights carrying the original softmax
weights. After the grouped GEMM path, shuffle_rows(c2, c_map, ...) at
line 493 finds c_map[slot] == 0 for non-local slots (from our Fix 1
zero-init) and reads c2[0] — the first ACTIVE expert's output for the
first local token — into those slots. The non-local slots then go into
`c2 * topk_weights.view(m, num_topk, 1)` carrying real-but-wrong finite
values multiplied by real (non-zero) weights, and `sum(dim=1)` aggregates
them into the output alongside the correct local-slot contributions.
Fix 2: mask topk_weights where topk_ids < 0 at the start of
cutlass_moe_fp4 — a .masked_fill before any math runs propagates through
both the `apply_router_weight_on_input=False` path (line 496 final
multiply) and the `=True` path (weights baked into input earlier).

Two separate PATCH_*_EOF blocks so each patch's grep guard runs
independently and failure of one doesn't prevent the other. [Historical note:
that was true of the original inline bash heredocs (this fix + the sibling
modelopt_quant.py num_local_experts fix); as separate files under
sglang_patches/ they are now already independent of each other by
construction.]

Upstream PR #20869 is the adjacent work but only fixes the first two bugs
in this chain (input-scale slicing + num_local_experts for CutlassMoEParams);
it then sidesteps the third+fourth bugs here by auto-routing SM120 to
flashinfer_cutlass. Our monkey-patches are the first real fix for the
cutlass_moe_fp4 codepath under EP that we are aware of. See
SGLANG_NVFP4_SHUFFLE_ROWS_OOB_UPSTREAM_BUG.md for the full debug ordeal.

Only patches the cutlass_moe_fp4 call site — there is also a torch.empty at
~line 145 inside cutlass_fused_experts_fp8 which is the FP8 MoE path (not
affected by this bug; left alone). The anchor below is discriminated by the
surrounding `num_topk = topk_ids.shape[1]` line that exists only in
cutlass_moe_fp4.
"""

from _patchlib import Patch

patch = Patch(
    name="a_map/c_map zero-init + topk_weights mask for EP",
    target="sglang/srt/layers/moe/cutlass_moe.py",
)

OLD = """    num_topk = topk_ids.shape[1]
    device = a.device
    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)"""

NEW = """    num_topk = topk_ids.shape[1]
    device = a.device
    # EP-aware: mask topk_weights where topk_ids == -1 (non-local sentinels
    # from StandardDispatcher.local_expert_mapping). Without this, non-local
    # slots carry real softmax weights into the final c2 * topk_weights
    # multiply and pollute the output reduction with the wrong expert's
    # values (see Fix 2 in the patch header).
    topk_weights = topk_weights.masked_fill(topk_ids < 0, 0)
    # EP-aware: zero-init instead of torch.empty. prepare_moe_input only
    # writes slots for non-(-1) topk_ids; -1 entries leave slots as garbage
    # and the downstream a.index_select(0, a_map) trips torch's bounds
    # check. Zero is a valid row index into `a`; the fake-gathered rows
    # then multiply by our newly-masked (zero) topk_weights above and
    # vanish in the reduction (see Fix 1 in the patch header).
    a_map = torch.zeros((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.zeros((topk_ids.numel()), dtype=torch.int32, device=device)"""


@patch.run
def apply(p: Patch) -> None:
    p.replace(
        OLD, NEW, marker="EP-aware: mask topk_weights where topk_ids", what="a_map/c_map zero-init + topk_weights mask"
    )
