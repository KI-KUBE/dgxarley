# SGLang/vLLM Upstream Bugs: MoE + Expert Parallelism (moe_wna16 + modelopt_quant)

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

## Additional Bug: EPLB crashes with Qwen3MoE and Qwen3.5MoE

When `--enable-eplb` is active with EP, the `EPLBManager` crashes after its first rebalance
interval (default: 1000 forward passes):

```
File ".../sglang/srt/eplb/eplb_manager.py", line 110, in _compute_update_layer_ids_chunks
    list(self._model_runner.model.routed_experts_weights_of_layer.keys())
AttributeError: 'Qwen3MoeForCausalLM' object has no attribute 'routed_experts_weights_of_layer'
```

The EPLB rebalancer needs models to expose a `routed_experts_weights_of_layer` property
(a dict mapping layer IDs to their expert weight tensors) so it can transfer weights between
GPUs. Neither `Qwen3MoeForCausalLM` nor `Qwen3_5MoeForConditionalGeneration` implements this —
likely only `DeepseekV3ForCausalLM` (or similar) was tested with EPLB.

**Impact**: The crash kills the scheduler, which triggers SIGQUIT → full restart of both nodes.
This happens reliably after ~1000 inference passes (~8 min wall time under moderate load).

**Confirmed failing on both architectures:**

| Model class | Image | Date | Pod |
|---|---|---|---|
| `Qwen3MoeForCausalLM` (Qwen3-235B-A22B) | 0.5.9-t5 | 2026-03-19 | sglang-head-855c5799c4 |
| `Qwen3_5MoeForConditionalGeneration` (Qwen3.5-122B-A10B-FP8) | 0.5.9-dev2-acab24a7-t5 | 2026-03-20 | sglang-head-5d7585955 |

The Qwen3.5 crash on dev2-acab24a7-t5 proves that [PR #19767](https://github.com/sgl-project/sglang/pull/19767)
("Fix qwen3.5 mtp eplb related issues", merged 2026-03-09) is either not included in
commit `acab24a7` (2026-03-11), or does not actually fix the `routed_experts_weights_of_layer`
attribute for `Qwen3_5MoeForConditionalGeneration` despite the claim. The exact same
`AttributeError` on the same code path (`eplb_manager.py:110`) occurs.

**Workaround**: Disable EPLB (`--enable-eplb` removed). EP=2 still works with static expert
assignment (experts 0–63 → GPU 0, experts 64–127 → GPU 1). The static assignment is suboptimal
if expert activation is highly skewed, but in practice Qwen3-235B shows ~0.82 balancedness
which is acceptable.

## Additional Bug: modelopt_quant.py NVFP4 input_scale not EP-aware

### Status

**Unreported** as of 2026-03-27. Bug exists in SGLang `sglang/srt/layers/quantization/modelopt_quant.py`, class `ModelOptNvFp4FusedMoEMethod`.

### Affected Configuration

- Quantization: `modelopt_fp4` (NVFP4-quantized MoE models)
- Expert Parallelism: `ep_size > 1`
- Backend: the `else` fallback branch in `process_weights_after_loading` (i.e., when neither `enable_flashinfer_cutlass_moe`, `enable_flashinfer_trtllm_moe`, nor `enable_flashinfer_cutedsl_moe` is active — this is the path hit by the **shard job** which doesn't configure a MoE runner backend)
- Tested with: nvidia/MiniMax-M2.5-NVFP4 (256 experts), TP=2, EP=2, shard job on 0.5.9-dev2-acab24a7-t5

The `flashinfer_cutlass` and `trtllm` branches are **not affected** (they reduce input_scale to a scalar via `.max()`). The `cutedsl` branch is **not affected** (it has a `_slice_scale()` helper that correctly slices to local experts).

### The Bug

In `process_weights_after_loading()`, the `else` branch computes:

```python
w13_input_scale = layer.w13_input_scale.max(dim=-1).values.to(torch.float32)  # shape: (num_experts,)
w2_input_scale = layer.w2_input_scale                                          # shape: (num_experts,)
```

These are then multiplied with EP-local weight scales:

```python
(w13_input_scale * w13_weight_scale_2).to(torch.float32)  # (256,) * (128,) → RuntimeError
(w2_input_scale * layer.w2_weight_scale_2).to(torch.float32)  # same
```

`w13_weight_scale_2` has shape `(num_local_experts,)` = 128 with EP=2, but `w13_input_scale` remains at `(num_experts,)` = 256.

### Root Cause

`w13_input_scale` and `w2_input_scale` are allocated as global tensors (flagged with `_sglang_require_global_experts = True`) because the weight loader fills them for all experts. The `cutedsl` branch correctly slices them to local experts via `_slice_scale()`, but this helper is defined inside the `elif` block and is not available to the `else` branch. The `else` branch was never tested with EP > 1.

### Crash Output

```
[2026-03-27 18:41:27 TP0 EP0] Scheduler hit an exception:
  File ".../modelopt_quant.py", line 1560, in process_weights_after_loading
    (w13_input_scale * w13_weight_scale_2).to(torch.float32),
RuntimeError: The size of tensor a (256) must match the size of tensor b (128) at non-singleton dimension 0
```

### Fix

Add EP-aware slicing in the `else` branch, same logic as `_slice_scale()`:

```python
        else:
            w13_input_scale = layer.w13_input_scale.max(dim=-1).values.to(torch.float32)
            w2_input_scale = layer.w2_input_scale
            # EP-aware slicing: no-op when ep_size=1
            if layer.moe_ep_size > 1:
                _ep_start = layer.moe_ep_rank * layer.num_local_experts
                _ep_end = _ep_start + layer.num_local_experts
                w13_input_scale = w13_input_scale[_ep_start:_ep_end]
                w2_input_scale = w2_input_scale[_ep_start:_ep_end]
```

### Our Workaround

Monkey-patched in `sglang_launch.sh` and `sglang_shard_launch.sh` (same string-replace pattern as the `moe_wna16` patch). The patch inserts the EP slicing block after the two assignments in the `else` branch.

## Related Issues (none address these bugs)

- vLLM #12647 — moe_wna16 AssertionError (KV cache conflict, unrelated)
- vLLM #22961 — TypeError in moe_wna16_weight_loader (return_success param, unrelated)
- vLLM PR #14447 — introduced moe_wna16 marlin kernel (origin of this code)
- SGLang PR #17137 — non-Marlin WNA16MoE port (does not fix EP bug)
- SGLang #14158 — update_weights_from_tensor for WNA16MoE (unrelated)
