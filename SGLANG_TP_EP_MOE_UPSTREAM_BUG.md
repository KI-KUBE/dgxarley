# SGLang/vLLM Upstream Bug: moe_wna16 qzeros + Expert Parallelism

## Status

**Unreported** as of 2026-03-19. Bug exists in both SGLang and vLLM (code originated in vLLM PR #14447).

- SGLang: `sglang/srt/layers/quantization/moe_wna16.py`, lines 491–504 (v0.5.9)
- vLLM: `vllm/model_executor/layers/quantization/moe_wna16.py`, lines 492–505

## Affected Configuration

- Quantization: `moe_wna16` (AWQ/GPTQ 4-bit MoE models with `zero_point: true`)
- Expert Parallelism: `ep_size > 1`
- Tensor Parallelism: `tp_size > 1`
- Tested with: Qwen3-235B-A22B MoE (128 experts), TP=2, EP=2, AWQ 4-bit

Models with `zero_point: false` (symmetric quantization) are **not affected** — qzeros loading is skipped entirely via early return.

## The Bug

The `moe_wna16_weight_loader` closure in `MoeWNA16Method.get_weight_loader()` has three code paths:

1. `if "w13_qzeros" in weight_name` — custom inline logic for gate/up qzeros
2. `elif "w2_qzeros" in weight_name` — custom inline logic for down qzeros
3. `else` — delegates to `FusedMoE.weight_loader` (handles qweight, scales)

The `else` path correctly calls `layer._map_global_expert_id_to_local_expert_id()` for EP remapping and uses the MoE-local TP rank for slicing. The qzeros paths bypass this and have **two bugs**:

### Bug 1: Global expert_id instead of local EP index

```python
# BUGGY: expert_id is global (0-127), param.data has shape [64, ...] with EP=2
param.data[expert_id, : shard_size // 2] = tensor   # IndexError when expert_id >= 64
```

The weight loader is called for all 128 experts on every rank. `FusedMoE.weight_loader` maps global → local and skips non-local experts. The qzeros branches index directly with the global id.

### Bug 2: Global tp_rank instead of MoE-local tp_rank

```python
# BUGGY: tp_rank is global (0 or 1), but moe_tp_size = tp_size/ep_size = 1 with EP=2
tensor = loaded_weight.view(layer.moe_tp_size, -1, loaded_weight.size(1))[tp_rank]
# view creates dimension of size 1, tp_rank=1 → IndexError
```

With EP, MoE layers don't use TP splitting (`moe_tp_size = 1`). The correct index is `layer.moe_tp_rank` (or equivalently `tp_rank % layer.moe_tp_size`), which is 0 on both ranks.

## Root Cause

The qzeros branches were written as special cases that bypass the generic `FusedMoE.weight_loader`. They handle the tensor reshaping differently from qweight/scales (different view dimensions), which is why they can't simply delegate. But they failed to replicate the EP-aware expert remapping and MoE-local TP rank logic that `FusedMoE.weight_loader` provides.

## Fix

For each qzeros branch, add EP remapping and use MoE-local TP rank:

```python
if "w13_qzeros" in weight_name:
    _local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
    if _local_id == -1:
        return
    _moe_tp_rank = tp_rank % layer.moe_tp_size  # or: layer.moe_tp_rank
    tensor = loaded_weight.view(
        layer.moe_tp_size, -1, loaded_weight.size(1)
    )[_moe_tp_rank]
    if shard_id == "w1":
        param.data[_local_id, : shard_size // 2] = tensor
    else:
        param.data[_local_id, shard_size // 2 :] = tensor
elif "w2_qzeros" in weight_name:
    _local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
    if _local_id == -1:
        return
    _moe_tp_rank = tp_rank % layer.moe_tp_size
    param.data[_local_id] = loaded_weight.view(
        loaded_weight.size(0), layer.moe_tp_size, -1
    )[:, _moe_tp_rank]
```

The fix is a no-op when `ep_size=1`: `_local_id == expert_id` (identity mapping) and `_moe_tp_rank == tp_rank` (modulo has no effect).

## Our Workaround

We monkey-patch `moe_wna16.py` at container startup in `sglang_launch.sh` and `sglang_shard_launch.sh` (Python string-replace before SGLang starts). Same pattern as the existing ShardedStateLoader progress-logging patch.

## Caveat: Is moe_wna16 + EP even the right combination?

The reason `moe_wna16` was originally chosen is to avoid the **Marlin repack memory peak**.
Without `--quantization moe_wna16`, SGLang auto-detects `AWQMarlinConfig`, which repacks
AWQ weights into Marlin format at load time. During repack, old and new tensors coexist in
GPU memory — for Qwen3-235B-A22B AWQ with TP=2, this peaks at ~109 GB per GPU (vs. 128 GB
available on DGX Spark).

However, this calculation assumes **TP=2 without EP**. With EP=2, the memory situation changes
fundamentally:

- **TP=2 only**: each GPU holds all 128 experts (TP-split) → ~62 GB weights, ~109 GB repack peak
- **TP=2 + EP=2**: each GPU holds only 64 experts (full weight per expert) → ~31–35 GB weights, ~55–60 GB repack peak → **fits comfortably in 128 GB**

This means the original motivation for `moe_wna16` (avoiding repack OOM) **disappears with EP**.
The standard `AWQMarlinConfig` code path goes through `FusedMoE.weight_loader`, which is
fully EP-aware and well-tested. The `moe_wna16` code path, by contrast, has the qzeros bug
documented above — precisely because it bypasses `FusedMoE.weight_loader` for a niche case
that few people apparently test with EP.

**Practical takeaway**: When using EP, consider dropping `quantization: "moe_wna16"` and
letting auto-detection use `AWQMarlinConfig` instead. This avoids the bug entirely and uses
the mainstream, well-tested code path. The monkey-patch documented here remains valid for
anyone who does need `moe_wna16 + EP` (e.g., on GPUs with less memory headroom), but it may
be solving a problem that doesn't need to exist.

## Additional Bug: EPLB crashes with Qwen3MoE

When `--enable-eplb` is active with EP, the `EPLBManager` crashes after its first rebalance
interval (default: 1000 forward passes):

```
File ".../sglang/srt/eplb/eplb_manager.py", line 110, in _compute_update_layer_ids_chunks
    list(self._model_runner.model.routed_experts_weights_of_layer.keys())
AttributeError: 'Qwen3MoeForCausalLM' object has no attribute 'routed_experts_weights_of_layer'
```

The EPLB rebalancer needs models to expose a `routed_experts_weights_of_layer` property
(a dict mapping layer IDs to their expert weight tensors) so it can transfer weights between
GPUs. `Qwen3MoeForCausalLM` does not implement this — likely only `DeepseekV3ForCausalLM`
(or similar) was tested with EPLB.

**Impact**: The crash kills the scheduler, which triggers SIGQUIT → full restart of both nodes.
This happens reliably after ~1000 inference passes.

**Workaround**: Disable EPLB (`--enable-eplb` removed). EP=2 still works with static expert
assignment (experts 0–63 → GPU 0, experts 64–127 → GPU 1). The static assignment is suboptimal
if expert activation is highly skewed, but in practice Qwen3-235B shows ~0.82 balancedness
which is acceptable.

## Related Issues (none address these bugs)

- vLLM #12647 — moe_wna16 AssertionError (KV cache conflict, unrelated)
- vLLM #22961 — TypeError in moe_wna16_weight_loader (return_success param, unrelated)
- vLLM PR #14447 — introduced moe_wna16 marlin kernel (origin of this code)
- SGLang PR #17137 — non-Marlin WNA16MoE port (does not fix EP bug)
- SGLang #14158 — update_weights_from_tensor for WNA16MoE (unrelated)
