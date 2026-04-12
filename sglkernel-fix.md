# cutlass_moe_fp4 EP Fix — Hypotheses & Plan

## Current state (2026-04-12)

**Root cause isolated: EP dispatch/combine layer is broken for the
cutlass_moe_fp4 code path.** The CUTLASS GEMM kernel itself is numerically
correct on SM121 — confirmed by EP=1 test producing perfect output with
vanilla scitrera image + triton backend.

Two monkey-patches in `sglang_launch.sh` prevent the crash:
1. `a_map/c_map = torch.zeros` (prevent uninitialized-memory OOB in `_shuffle_rows_torch`)
2. `topk_weights.masked_fill(topk_ids < 0, 0)` (zero out non-local expert weights)

**Output is garbage under EP>1.** Both the vanilla scitrera image and our custom
xomoxcc-sm121 image produce incoherent tokens (`.RY95 mun am`, `Hello ModernTAmen`,
etc.) when EP=4. The mask helped (degenerate `!!!!!` → non-degenerate) but didn't
fix the semantic correctness. The bug is NOT in the GEMM kernel; it's in how
sglang's `StandardDispatcher` + `cutlass_moe_fp4` combine partial EP results.

## What we've confirmed

| Fact | Evidence |
|------|----------|
| `topk_ids` contains -1 sentinels for non-local experts | `StandardDispatcher.dispatch()` uses `local_expert_mapping` with -1 fill |
| `prepare_moe_input` only writes a_map/c_map for non-(-1) entries | `compute_arg_sorts` iterates blockIdx.x over [0, num_experts), no block for -1 |
| `topk_weights` are NOT zeroed by the dispatcher for -1 slots | confirmed by garbage output: pre-mask → degenerate, post-mask → less degenerate |
| EP all-reduce EXISTS in `forward_normal` line 328 | `moe_expert_parallel_all_reduce(final_hidden_states)` fires when `self.ep_size > 1` |
| No DOUBLE all-reduce (EP+TP over same group) | `self.tp_size = get_moe_tensor_parallel_world_size()` → moe_tp_size, which is 1 when tp=ep=4 |
| `sgl-kernel-sm121.patch` not needed for crash avoidance | vanilla scitrera runs without cuh:78 crash |
| `sgl-kernel-sm121.patch` not needed for numerical correctness | EP=1 + scitrera vanilla + triton = correct output |
| `cutlass_moe_fp4` + triton + SM121 is CORRECT at EP=1 | coherent reasoning output, perfect text generation |
| Bug is 100% in EP dispatch/combine layer | EP=1 correct, EP=4 garbage, same kernel, same image |
| `CutlassMoEParams.num_experts = num_local_experts = 32` | existing monkey-patch, applied successfully |

## Active hypotheses (ranked by likelihood)

### H1: sgl-kernel-sm121.patch needed for NUMERICAL correctness (not crash avoidance)

**Theory:** Vanilla Pingpong + StageCountAutoCarveout on SM121 doesn't crash
(auto-carveout picks a valid stage count under the 99 KiB budget) but produces
**numerically incorrect** GEMM results — subtle bit-level errors from wrong
pipeline staging, shared-memory layout mismatch, or warp-schedule-dependent
data hazards that only affect SM120/121 with the Pingpong schedule.

Our sgl-kernel-sm121.patch (Cooperative + StageCount<2>) explicitly controls
both the schedule and stage count, potentially avoiding a silent correctness
issue that auto-carveout + Pingpong introduces.

**Test:** Run xomoxcc-sm121 image (has our CUTLASS patch) with the full
monkey-patch set (mask + zero-init). Compare output quality to the scitrera
vanilla test. Three possible outcomes: (a) correct → H1 confirmed, (b) same
garbage → schedule is irrelevant, bug is in Python EP layer, (c) different
garbage → both schedules are broken on SM121, need EP=1 test to isolate.

**Caveat:** Even if H1 is confirmed, it could be that our patch is ALSO wrong
but happens to produce correct-looking output by coincidence. The real test
of correctness is comparison against flashinfer_cutlass baseline (known good).

**Status:** Test pending (image switched back to xomoxcc, deploy + curl needed).

### H2: `silu_and_mul` garbage tail corrupting block-scale computation

**Theory:** `silu_and_mul(c1, intermediate)` iterates ALL `m_a * num_topk`
rows, reading garbage from c1's non-active tail (rows beyond
`expert_offsets[num_local_experts]`). If `scaled_fp4_experts_quant`'s native
kernel computes block scales using a global statistic (max/absmax over ALL
rows, not per-expert) then the garbage rows pollute the quantization
scale for the ENTIRE tensor — including the active range. This would produce
subtly wrong FP4 values everywhere, leading to "real but wrong" token output.

**Test:** Zero-init `c1` before `cutlass_fp4_group_mm` writes to it (or
zero-init `intermediate`). If output quality improves → H2 confirmed.

**Difficulty:** Requires another monkey-patch on line 459/471 of cutlass_moe.py.

### H3: `c2` scatter pollution from `c_map` zero-init

**Theory:** After `shuffle_rows(c2, c_map, ...)` with zero-init c_map, the
non-active positions in the reshuffled c2 all point to c2[0] (first active
expert's first token output). Even though topk_weights for those positions
are masked to 0, there might be a numerical edge case where BF16 zero ×
large c2 value → nonzero (denormalized) contribution. Or there's an
`apply_router_weight_on_input=True` path we haven't checked.

**Test:** Verify `apply_router_weight_on_input` value at runtime (should be
False for Qwen3-235B). Print c2 values at non-active positions after
shuffle. If they're NaN/Inf → the 0 × NaN path might produce NaN.

**Difficulty:** Low, just logging.

### H4: EP all-reduce over wrong group or missing for some layers

**Theory:** `moe_expert_parallel_all_reduce` uses `get_moe_ep_group()`. If
the EP group is incorrectly configured (e.g., only size 1, or wrong rank
mapping), the all-reduce is a no-op or sums the wrong processes. Each rank
would return its partial 25% contribution.

**Test:** In the running pod, instrument `moe_expert_parallel_all_reduce` to
print the group's world_size and rank. Or check `get_moe_ep_group().world_size`.

**Difficulty:** Requires kubectl exec with a python one-liner that hooks into
the sglang process.

### H5: `cutlass_fp4_group_mm` internal output tensor not fully written

**Theory:** The CUTLASS grouped GEMM kernel might have a tile-boundary issue
where the last tile of an expert's row range writes beyond `expert_offsets[e+1]`
or leaves the last few columns unwritten for non-aligned hidden sizes. On
SM121 with the specific tile shape <_128, _128, _128>, this could produce
partial garbage within the "active" range itself.

**Test:** Run with a single expert (EP=1, no EP) and verify correctness. If
EP=1 works but EP=4 doesn't, this hypothesis is less likely (issue is
EP-specific, not tile-boundary-specific).

**Difficulty:** Config change to ep_size=1, redeploy.

### H6: FP4 quantization precision loss on SM121

**Theory:** The FP4 (NVFP4) quantization format might have reduced precision
on SM121 vs SM100 due to hardware-level differences in the tensor core FP4
multiply-accumulate path. This would be an architectural limitation, not a
software bug — the same code works on SM100 but produces garbage on SM121.

**Test:** Run the same model with `moe_runner_backend=flashinfer_cutlass`
(bypasses cutlass_moe_fp4 entirely) to confirm FP4 inference quality is fine
on SM121 via a different code path. We already know flashinfer_cutlass works
(test matrix winner at 11.28 / 34.60 / 42.70 tok/s). If that works → SM121
FP4 hardware is fine, the bug is in the software path.

**Status:** Already confirmed via the test matrix.

## Test plan (in order)

### Test 1: xomoxcc-sm121 + full monkey-patches (H1)

```bash
# Image already switched to xomoxcc/dgx-spark-sglang:0.5.10-sm121
ansible-playbook k8s_dgx.yml -t sglang
# wait for pods ready, then:
curl -sS --max-time 180 -X POST https://sglang.dgx.elasticc.io/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"nvidia/Qwen3-235B-A22B-NVFP4","messages":[{"role":"user","content":"Say hello in exactly one sentence."}],"max_tokens":64,"temperature":0.6}'
```

- **Correct output → H1 confirmed.** Our CUTLASS patch is needed for numerical
  correctness. The scitrera vanilla garbage was from the Pingpong schedule, not
  from our Python fixes.
- **Same garbage as scitrera → Schedule is irrelevant.** Both Pingpong and
  Cooperative produce the same (wrong or right) GEMM result. The bug is above
  the kernel level — in the EP dispatch/combine/allreduce Python layer. Move
  to Test 2.
- **Different garbage than scitrera → Both schedules are numerically broken
  on SM121**, just in different ways. Need Test 5 (EP=1) to distinguish
  "SM121 GEMM is fundamentally broken" from "EP layer is broken".

### Test 2: Add intermediate zero-init (H2)

If H1 refuted: add another monkey-patch in `sglang_launch.sh` that zero-inits
`intermediate` at line 471:

```python
# Was:
intermediate = torch.empty((m_a * num_topk, w1_fp4.shape[1] // 2), ...)
# Becomes:
intermediate = torch.zeros((m_a * num_topk, w1_fp4.shape[1] // 2), ...)
```

Also consider zero-init for the `output` tensor inside `cutlass_fp4_group_mm`
(line 146 of nvfp4.py) — but this is inside a different function and harder to
monkey-patch. Start with `intermediate`.

### Test 3: Flashinfer_cutlass sanity (H6 / baseline)

If Tests 1+2 both garbage: switch to `moe_runner_backend: flashinfer_cutlass`
(known winner) to confirm the rest of the inference stack is healthy. This has
already been validated in the test matrix but should be re-confirmed after all
our monkey-patches.

### Test 4: EP group introspection (H4)

If flashinfer_cutlass works correctly but cutlass_moe_fp4 doesn't, add a
one-shot print probe to verify EP group membership:

```python
# Add to cutlass_moe_fp4 via sglang_launch.sh, runs once:
import sglang.srt.distributed as d
_ep_group = d.get_moe_ep_group()
print(f"[EP-debug] ep_group world_size={_ep_group.world_size} rank={_ep_group.rank_in_group}")
```

### Test 5: EP=1 baseline (H5)

If all else fails: run with EP=1 (tp_size=4, ep_size=1 in defaults) to confirm
the cutlass_moe_fp4 path produces correct output without expert parallelism.
If EP=1 works → bug is specifically in the EP dispatch/combine/allreduce path.
If EP=1 also garbage → bug is in cutlass_moe_fp4 itself on SM121.

## Files involved

- `roles/k8s_dgx/files/sglang_launch.sh` — all monkey-patches live here
- `roles/k8s_dgx/defaults/main.yml` — sglang_image (switch scitrera ↔ xomoxcc)
- `roles/k8s_dgx/tasks/sglang.yml` — ConfigMap env vars (CUDA_LAUNCH_BLOCKING, DGXARLEY_SM121_DEBUG)
- `SGLANG_NVFP4_SHUFFLE_ROWS_OOB_UPSTREAM_BUG.md` — full debug narrative
- `scripts/patches/sgl-kernel-sm121.patch` — the CUTLASS StageCount<2> + Cooperative patch
- `scripts/patches/sgl-kernel-sm121-debug.patch` — the pre/post-workspace stream-error diagnostic probes
