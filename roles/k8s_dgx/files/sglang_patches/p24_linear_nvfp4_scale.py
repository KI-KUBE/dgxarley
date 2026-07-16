"""[dgxarley] linear.py: NVFP4 scalar-scale loading for fused/merged linears.

Upstream fix PR #29151 / commit cfc3d0555e ("Fix ModelOpt NVFP4 scalar scales for
merged linears"), merged 2026-07-13, MAIN-ONLY (not in the 0.5.15 image).
MergedColumnParallelLinear / QKVParallelLinear weight_loader_v2 hardcode shard_id=0
for the PerTensorScaleParameter (NVFP4 weight_scale_2 / input_scale scalars) when
loaded_shard_id is None OR a tuple, leaving the other logical slots UNINITIALIZED;
the shared ModelOpt NVFP4 path then .max()es over the slots, so a garbage slot
becomes the runtime global alpha -> catastrophic dequant -> word-salad. Worst on
GDN-hybrid models whose fused linear_attn.in_proj_qkv loads via a TUPLE shard_id
(Ornith-1.0-9B: 24 GDN layers dominate -> total garbage). Fix: fill EVERY logical
slot with the scalar. INERT for non-NVFP4 checkpoints (PerTensorScaleParameter path
only taken for modelopt_fp4 scalar scales).
"""

from _patchlib import Patch

patch = Patch(
    name="NVFP4 scalar-scale loading for fused/merged linears (PR #29151)",
    target="sglang/srt/layers/linear.py",
)

OLD_MERGED = """            if isinstance(param, PerTensorScaleParameter):
                param.load_merged_column_weight(
                    loaded_weight=loaded_weight,
                    shard_id=0,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
                return
"""

NEW_MERGED = """            if isinstance(param, PerTensorScaleParameter):
                # [patch]: PR #29151 -- fill every logical slot with the
                # scalar NVFP4 scale instead of only slot 0, else the unfilled slots
                # feed garbage into the .max() global alpha (fused merged linears,
                # e.g. GDN in_proj_qkv with tuple shard_id).
                if loaded_weight.numel() != 1:
                    raise ValueError(
                        "Expected scalar scale for fused-in-checkpoint "
                        "merged-column checkpoint load, got shape "
                        f"{tuple(loaded_weight.shape)}"
                    )
                if loaded_shard_id is None:
                    shard_ids = range(param.data.shape[0])
                else:
                    shard_ids = loaded_shard_id
                for shard_id in shard_ids:
                    param.load_merged_column_weight(
                        loaded_weight=loaded_weight,
                        shard_id=shard_id,
                        tp_rank=self.tp_rank,
                        tp_size=self.tp_size,
                    )
                return
"""

OLD_QKV = """            if isinstance(param, PerTensorScaleParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight, shard_id=0)
                return
"""

NEW_QKV = """            if isinstance(param, PerTensorScaleParameter):
                # [patch]: PR #29151 -- fill all q/k/v slots with the scalar scale
                if loaded_weight.numel() != 1:
                    raise ValueError(
                        "Expected scalar scale for fused-in-checkpoint QKV "
                        "checkpoint load when loaded_shard_id is None, got "
                        f"shape {tuple(loaded_weight.shape)}"
                    )
                for shard_id in param.qkv_idxs:
                    param.load_qkv_weight(
                        loaded_weight=loaded_weight, shard_id=shard_id
                    )
                return
"""


@patch.run
def apply(p: Patch) -> None:
    p.replace_all(
        OLD_MERGED,
        NEW_MERGED,
        marker="# [patch]: PR #29151 -- fill every logical slot",
        what="nvfp4-scale D-merged-col-scalar-scale",
    )
    p.replace_all(
        OLD_QKV,
        NEW_QKV,
        marker="# [patch]: PR #29151 -- fill all q/k/v slots",
        what="nvfp4-scale D-qkv-scalar-scale",
    )
