"""[dgxarley] Qwen3.8 QSA sparse decode on GB10/SM121: backport of upstream #36845.

WHY
  The day-0 qwen4-main-squashed branch (@ 99c9362e, what our 0.5.18-sm121 image
  carries via scripts/patches/sglang-qwen4exp-pr36497.patch) has no correct QSA
  sparse-decode kernel for SM121. p65 EDIT 2 makes the engine BOOT by routing the
  packed varlen fallback to SGLang's FA4 dispatcher, but p65's own header says
  that kernel was never output-validated on SM121. Measured 2026-09-16 on
  RadixArk/Qwen3.8-Flash-Next-NVFP4, TP4, cuda graphs on: 3/16 short (<1K token)
  generations collapsed mid-answer into 370-400 consecutive "!" (token id 0).
  An SM120 reporter saw the same FA4 varlen fallback collapse 5-19% of
  generations (sgl-project/sglang#36537).

WHAT
  Byte-exact backport of sgl-project/sglang#36845 ("fix(qsa): restore SM121
  correctness", merged 2026-08-30 as 78c5024e, the ONLY commit between our pin
  99c9362e and it):
    1. new package sglang/kernels/kda_kernels/qwen38_qsa_sm121/ (a Triton
       kernel for the exact packed-QSA decode contract; kernel.py SHA256
       a0fd72586b341a1405cf5a4aa98686cabf3965270ae1e5f70a93ff7db0b104d1,
       identical to upstream at 78c5024e),
    2. its registry entry + wrapper in sglang/kernels/ops/attention/__init__.py
       (that file is byte-identical in v0.5.18 and in #36845's base, blob 4030ea3),
    3. an SM121 early-return in _resolve_flash_attn_varlen_func. Placed BEFORE
       p65 EDIT 2's family-12 FA4 branch, so on SM121 this kernel wins and FA4 is
       no longer reached; SM120 keeps p65's routing. p65 EDIT 1 (trtllm veto) is
       unaffected and stays.

CONTRACT LIMIT, READ BEFORE CHANGING PARALLELISM
  The kernel only accepts BF16, head_dim 256, 12:1 GQA with (24 q, 2 kv) = TP1
  or (12 q, 1 kv) = TP2, batch rows <= 128 and selected KV <= 2055. Anything
  else RAISES ValueError at decode ("unsupported SM121 QSA call"). TP4 (6 q per
  0.5 kv) is NOT covered, so this patch is only usable with tp_size 1 or 2. With
  NEXTN 3/1/4 and cuda_graph_max_bs 32 the verify pass hits exactly 128 rows.

GATES
  target_contains on the QSA backend, so every image without qwen4exp logs one
  "gate not matched" line. The two source edits run only when all three modules
  were written AND imported, since routing to a missing kernel would crash decode.
  The routing edit additionally requires env TP in {1, 2, 4} (set on head and
  worker from sglang_tp). Patches run once per fresh container, so a TP change
  always re-evaluates this gate.

LOCAL DEVIATION FROM UPSTREAM: TP4 (6 q, 1 replicated kv)
  Upstream's contract is a hard allowlist `_SUPPORTED_HEAD_TOPOLOGIES =
  frozenset({(12, 1), (24, 2)})`, i.e. only the shapes they captured on one and
  two Sparks. The Triton kernel itself is generic in the head ratio:
  NUM_Q_HEADS/NUM_KV_HEADS are constexprs from the live shapes,
  queries_per_kv = NUM_Q_HEADS // NUM_KV_HEADS, and the per-kv-head grouping is
  a masked arange over BLOCK_M = 16 (m < queries_per_kv), so 6:1 fits the same
  scheme. Only when env TP is "4" the written contract check gets (6, 1) added.
  The use_bk32 block-size heuristic was tuned on TP1/TP2 data only. Validated
  EMPIRICALLY only (2026-09-16 decision: no FP32 reference comparison); see the
  model profile STATUS for the probe result before relying on TP4.

DELETE WHEN the image is built from a qwen4exp source that already contains
78c5024e (or upstream main with #36845's kernel), i.e. when the backend file
already imports qwen38_qsa_sm121_varlen; the markers below then report
"already applied".
"""

import os

from _patchlib import DIST_PACKAGES, Patch, target_contains, write_module

BACKEND = "sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
OPS = "sglang/kernels/ops/attention/__init__.py"
PKG = os.path.join(DIST_PACKAGES, "sglang/kernels/kda_kernels")

GATE = target_contains(BACKEND, "def _resolve_flash_attn_varlen_func") and os.path.isfile(
    os.path.join(DIST_PACKAGES, "sglang/kernels/registry.py")
)

_KDA_INIT = r'''"""Kernels produced by Kernel Design Agent workflows."""
'''

_PKG_INIT = r'''# SPDX-License-Identifier: Apache-2.0

# KDA provenance: optimized by Codex and Kimi K3 agents through KDA-1.5.
# Task: https://github.com/radixark/KDA-1.5/pull/4 @
# 414ce456e14ae8546f77d9356d2c4d955c5bb7f1.
# Winning submission: b4181149c8884ddb.

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_SUPPORTED_HEAD_TOPOLOGIES = frozenset({(12, 1), (24, 2)})
# Largest batch qualified by the extended GB10 baseline sweep.
_MAX_BATCH = 128
_MAX_SELECTED_KV = 2055
_logged_fast_path = False


def can_use_qwen38_qsa_sm121(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
) -> bool:
    """Return whether this call matches the captured Qwen3.8 SM121 contract."""
    if not q.is_cuda or q.ndim != 3 or q.dtype != torch.bfloat16:
        return False
    batch, num_q_heads, head_dim = q.shape
    if not (0 < batch <= _MAX_BATCH) or head_dim != 256:
        return False
    if k.ndim != 3 or v.shape != k.shape or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    num_kv_heads = k.shape[1]
    if (num_q_heads, num_kv_heads) not in _SUPPORTED_HEAD_TOPOLOGIES:
        return False
    if k.shape[2] != head_dim or not (0 < max_seqlen_k <= _MAX_SELECTED_KV):
        return False
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        return False
    if q.device != k.device or q.device != v.device:
        return False
    if cu_seqlens_q.device != q.device or cu_seqlens_k.device != q.device:
        return False
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        return False
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        return False
    if not cu_seqlens_q.is_contiguous() or not cu_seqlens_k.is_contiguous():
        return False
    if cu_seqlens_q.numel() != batch + 1 or cu_seqlens_k.numel() != batch + 1:
        return False
    properties = torch.cuda.get_device_properties(q.device)
    return (properties.major, properties.minor) == (12, 1)


def qwen38_qsa_sm121(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Run the KDA-generated Qwen3.8 packed QSA decode kernel."""
    global _logged_fast_path
    if not can_use_qwen38_qsa_sm121(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_k):
        raise ValueError("unsupported call for the KDA Qwen3.8 SM121 QSA kernel")

    from .kernel import qwen38_qsa_sm121 as run_kernel

    if not _logged_fast_path:
        logger.info(
            "Using the Codex/Kimi K3 KDA Qwen3.8 QSA kernel on SM121 "
            "(radixark/KDA-1.5#4, submission b4181149c8884ddb)"
        )
        _logged_fast_path = True
    return run_kernel(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)


__all__ = ["can_use_qwen38_qsa_sm121", "qwen38_qsa_sm121"]
'''

_KERNEL = r'''# SPDX-License-Identifier: Apache-2.0

# KDA provenance: optimized by Codex and Kimi K3 agents through KDA-1.5
# (https://github.com/radixark/KDA-1.5).
# Task: https://github.com/radixark/KDA-1.5/pull/4 @
# 414ce456e14ae8546f77d9356d2c4d955c5bb7f1.
# Winning submission: b4181149c8884ddb; byte-exact submitted source SHA256:
# 4f9977f88abfea4393a2add3a2c9255699f7e13b981dbc1a976b024b3b00e909.
"""Shape-specialized Qwen3.8 packed QSA decode kernel for SM121."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_split_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    cu_q_ptr,
    cu_k_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_acc_ptr,
    counter_ptr,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
):
    sequence_idx = tl.program_id(0)
    split_program = tl.program_id(1)
    kv_head_idx = split_program // MAX_SPLITS
    split_idx = split_program - kv_head_idx * MAX_SPLITS
    slot = sequence_idx * NUM_KV_HEADS + kv_head_idx
    queries_per_kv = NUM_Q_HEADS // NUM_KV_HEADS

    query_idx = tl.load(cu_q_ptr + sequence_idx)
    kv_begin = tl.load(cu_k_ptr + sequence_idx)
    kv_end = tl.load(cu_k_ptr + sequence_idx + 1)
    kv_count = kv_end - kv_begin
    tile_count = tl.cdiv(kv_count, BLOCK_KV)

    # The split count depends only on live device metadata and the static
    # launch geometry, so CUDA Graph replay needs no host readback.
    batch = tl.num_programs(0)
    n_splits = 1
    if tile_count >= 1536 // BLOCK_KV:
        n_splits = 2
    elif tile_count >= 512 // BLOCK_KV and batch * NUM_KV_HEADS <= 4:
        n_splits = 4
    if batch == 1:
        if tile_count >= 1024 // BLOCK_KV:
            n_splits = 8
        elif tile_count >= 512 // BLOCK_KV:
            n_splits = 4
        elif tile_count >= 256 // BLOCK_KV:
            n_splits = 2
    if split_idx >= n_splits:
        return

    tile_lo = (tile_count * split_idx) // n_splits
    tile_hi = (tile_count * (split_idx + 1)) // n_splits
    kv_start = kv_begin + tile_lo * BLOCK_KV
    kv_stop = tl.minimum(kv_begin + tile_hi * BLOCK_KV, kv_end)

    m = tl.arange(0, BLOCK_M)
    d = tl.arange(0, HEAD_DIM)
    q_head = kv_head_idx * queries_per_kv + m
    q_mask = m < queries_per_kv
    query = tl.load(
        q_ptr + query_idx * q_stride_t + q_head[:, None] * q_stride_h + d[None, :],
        mask=q_mask[:, None],
        other=0.0,
    )

    n0 = tl.arange(0, BLOCK_KV)
    k_rows = (
        k_ptr
        + kv_head_idx * k_stride_h
        + kv_start.to(tl.int64) * k_stride_t
        + n0[:, None] * k_stride_t
    )
    v_rows = (
        v_ptr
        + kv_head_idx * v_stride_h
        + kv_start.to(tl.int64) * v_stride_t
        + n0[:, None] * v_stride_t
    )

    running_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    kv_len = kv_stop - kv_start
    full_end = kv_start + (kv_len // BLOCK_KV) * BLOCK_KV
    for block_start in range(kv_start, full_end, BLOCK_KV):
        keys = tl.load(k_rows + d[None, :])
        scores = tl.dot(query, tl.trans(keys)) * softmax_scale
        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
        values = tl.load(v_rows + d[None, :])
        accumulator = accumulator * old_scale[:, None] + tl.dot(
            probabilities.to(tl.bfloat16), values
        )
        running_max = new_max
        k_rows += BLOCK_KV * k_stride_t
        v_rows += BLOCK_KV * v_stride_t

    if full_end < kv_stop:
        n = full_end + n0
        n_mask = n < kv_stop
        keys = tl.load(k_rows + d[None, :], mask=n_mask[:, None], other=0.0)
        scores = tl.dot(query, tl.trans(keys)) * softmax_scale
        scores = tl.where(n_mask[None, :], scores, -float("inf"))
        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
        values = tl.load(v_rows + d[None, :], mask=n_mask[:, None], other=0.0)
        accumulator = accumulator * old_scale[:, None] + tl.dot(
            probabilities.to(tl.bfloat16), values
        )
        running_max = new_max

    if n_splits == 1:
        output = accumulator / tl.where(running_sum > 0.0, running_sum, 1.0)[:, None]
        tl.store(
            out_ptr
            + query_idx * out_stride_t
            + q_head[:, None] * out_stride_h
            + d[None, :],
            output.to(out_ptr.dtype.element_ty),
            mask=q_mask[:, None],
        )
        return

    partial_row = (slot * MAX_SPLITS + split_idx) * BLOCK_M + m
    tl.store(partial_max_ptr + partial_row, running_max)
    tl.store(partial_sum_ptr + partial_row, running_sum)
    tl.store(
        partial_acc_ptr + partial_row[:, None] * HEAD_DIM + d[None, :],
        accumulator,
    )
    tl.debug_barrier()
    arrival = tl.atomic_add(counter_ptr + slot, 1, sem="acq_rel", scope="gpu")
    if arrival == n_splits - 1:
        merged_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
        for j in tl.static_range(MAX_SPLITS):
            j_ok = (j < n_splits) & (m < BLOCK_M)
            row = (slot * MAX_SPLITS + j) * BLOCK_M + m
            mj = tl.load(partial_max_ptr + row, mask=j_ok, other=-float("inf"))
            merged_max = tl.maximum(merged_max, mj)
        merged_sum = tl.zeros([BLOCK_M], tl.float32)
        merged_acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
        for j in tl.static_range(MAX_SPLITS):
            j_ok = (j < n_splits) & (m < BLOCK_M)
            row = (slot * MAX_SPLITS + j) * BLOCK_M + m
            mj = tl.load(partial_max_ptr + row, mask=j_ok, other=-float("inf"))
            lj = tl.load(partial_sum_ptr + row, mask=j_ok, other=0.0)
            weight = tl.exp(mj - merged_max)
            merged_sum += weight * lj
            partial = tl.load(
                partial_acc_ptr + row[:, None] * HEAD_DIM + d[None, :],
                mask=j_ok[:, None],
                other=0.0,
            )
            merged_acc += weight[:, None] * partial
        output = merged_acc / tl.where(merged_sum > 0.0, merged_sum, 1.0)[:, None]
        tl.store(
            out_ptr
            + query_idx * out_stride_t
            + q_head[:, None] * out_stride_h
            + d[None, :],
            output.to(out_ptr.dtype.element_ty),
            mask=q_mask[:, None],
        )
        tl.atomic_xchg(counter_ptr + slot, 0, sem="release", scope="gpu")


# The worst-case TP1 scratch allocation at the qualified limit is 32.3 MiB.
_MAX_BATCH = 128
_MAX_KV_HEADS = 2
_BLOCK_M = 16
_MAX_SPLITS = 8
_MAX_SLOTS = _MAX_BATCH * _MAX_KV_HEADS
_HEAD_DIM = 256
_scratch: dict[int, tuple[torch.Tensor, ...]] = {}


def _get_scratch(device: torch.device) -> tuple[torch.Tensor, ...]:
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    scratch = _scratch.get(device_index)
    if scratch is None:
        rows = _MAX_SLOTS * _MAX_SPLITS * _BLOCK_M
        scratch = (
            torch.empty(rows, dtype=torch.float32, device=device),
            torch.empty(rows, dtype=torch.float32, device=device),
            torch.empty(rows * _HEAD_DIM, dtype=torch.float32, device=device),
            torch.zeros(_MAX_SLOTS, dtype=torch.int32, device=device),
        )
        _scratch[device_index] = scratch
    return scratch


def qwen38_qsa_sm121(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run the KDA-generated SM121 Qwen3.8 QSA kernel."""
    output = torch.empty_like(q)
    partial_max, partial_sum, partial_acc, counters = _get_scratch(q.device)
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]

    # This is the measured shape/topology schedule. The TP1 q_rows=4 shape is
    # intentionally kept on BK64 because its short and saturated rows are
    # host-indistinguishable; the live cu_seqlens still choose their split count.
    use_bk32 = (num_kv_heads == 1 and batch < 12) or (num_kv_heads == 2 and batch < 4)
    block_kv = 32 if use_bk32 else 64
    stages = 3 if use_bk32 else 2
    _qsa_split_kernel[(batch, num_kv_heads * _MAX_SPLITS)](
        q,
        k,
        v,
        output,
        cu_seqlens_q,
        cu_seqlens_k,
        partial_max,
        partial_sum,
        partial_acc,
        counters,
        softmax_scale,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_KV=block_kv,
        MAX_SPLITS=_MAX_SPLITS,
        q_stride_t=q.stride(0),
        q_stride_h=q.stride(1),
        k_stride_t=k.stride(0),
        k_stride_h=k.stride(1),
        v_stride_t=v.stride(0),
        v_stride_h=v.stride(1),
        out_stride_t=output.stride(0),
        out_stride_h=output.stride(1),
        num_warps=4,
        num_stages=stages,
    )
    return output
'''


def _pkg_init_source() -> str:
    if os.environ.get("TP", "") != "4":
        return _PKG_INIT
    old = "_SUPPORTED_HEAD_TOPOLOGIES = frozenset({(12, 1), (24, 2)})"
    if old not in _PKG_INIT:
        print("ANCHOR-DRIFT: qwen38_qsa_sm121/__init__.py: TP4 (6, 1) topology widening anchor missing")
        return _PKG_INIT
    return _PKG_INIT.replace(
        old, "_SUPPORTED_HEAD_TOPOLOGIES = frozenset({(12, 1), (24, 2), (6, 1)})  # [dgxarley] TP4"
    )


def _write_modules() -> bool:
    files = [
        (os.path.join(PKG, "__init__.py"), _KDA_INIT, "kda_kernels package"),
        (os.path.join(PKG, "qwen38_qsa_sm121", "__init__.py"), _pkg_init_source(), "qwen38_qsa_sm121 contract check"),
        (os.path.join(PKG, "qwen38_qsa_sm121", "kernel.py"), _KERNEL, "qwen38_qsa_sm121 Triton kernel (#36845)"),
    ]
    try:
        os.makedirs(os.path.join(PKG, "qwen38_qsa_sm121"), exist_ok=True)
    except OSError as exc:
        print(f"ANCHOR-DRIFT: kda_kernels: cannot create package dir ({exc})")
        return False
    ok = True
    for path, source, what in files:
        try:
            with open(path) as fh:
                if fh.read() == source:
                    print(f"[patch] {os.path.basename(path)}: {what} already written, skipping")
                    continue
        except OSError:
            pass
        ok = write_module(path, source, what) and ok
    return ok


_modules_ok = _write_modules() if GATE else False
if not GATE:
    print("[patch] QSA SM121 KDA kernel (#36845): gate not matched, skipping")

OPS_MARKER = 'op="attention.kda_qwen38_qsa_sm121"'

patch_ops = Patch(
    name="register the SM121 QSA KDA kernel (#36845)",
    target=OPS,
    when=_modules_ok,
)


@patch_ops.run
def apply_ops(p: Patch) -> None:
    p.replace(
        """from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import KernelBackend, KernelSpec
""",
        """from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch
""",
        marker="from sglang.kernels.selector import get_kernel",
        what="imports",
    )
    p.replace(
        """del _mod, _fn

__all__ = []
""",
        '''del _mod, _fn

register_kernel(
    KernelSpec(
        op="attention.kda_qwen38_qsa_sm121",
        backend=KernelBackend.TRITON,
        target=("sglang.kernels.kda_kernels.qwen38_qsa_sm121:" "qwen38_qsa_sm121"),
        capabilities=frozenset(
            {CapabilityRequirement.cuda(min_sm=(12, 1), max_sm=(12, 1))}
        ),
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            description=(
                "Qwen3.8 packed QSA decode: D=256, 12:1 GQA, " "1 <= q_rows <= 128"
            ),
        ),
        description=(
            "SM121 Qwen3.8 QSA decode optimized by Codex/Kimi K3 through " "KDA-1.5."
        ),
    )
)


def can_use_kda_qwen38_qsa_sm121(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
) -> bool:
    """Check the exact E2E-qualified Qwen3.8/SM121 QSA contract."""
    from sglang.kernels.kda_kernels.qwen38_qsa_sm121 import (
        can_use_qwen38_qsa_sm121,
    )

    return can_use_qwen38_qsa_sm121(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_k)


def qwen38_qsa_sm121_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = 1,
    max_seqlen_k: int = 0,
    softmax_scale: float = 1.0,
    causal: bool = True,
    **_: object,
) -> torch.Tensor:
    """Run the only SM121 packed-QSA kernel for its qualified contract."""
    del causal
    if max_seqlen_q != 1:
        raise ValueError(f"QSA requires max_seqlen_q=1, got {max_seqlen_q}")
    if not can_use_kda_qwen38_qsa_sm121(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_k
    ):
        raise ValueError(
            "unsupported SM121 QSA call: expected BF16 D=256, 12:1 GQA, "
            "TP1 24Q/2KV or TP2 12Q/1KV, bs<=128, and selected KV<=2055"
        )
    return get_kernel("attention.kda_qwen38_qsa_sm121", KernelBackend.TRITON)(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_k,
        softmax_scale,
    )


__all__ = ["can_use_kda_qwen38_qsa_sm121", "qwen38_qsa_sm121_varlen"]
''',
        marker=OPS_MARKER,
        what="kernel registration + wrapper",
    )


BACKEND_MARKER = "from sglang.kernels.ops.attention import (\n            qwen38_qsa_sm121_varlen,"

_SM121_BRANCH = """    from sglang.srt.utils import is_sm121

    if is_sm121():
        from sglang.kernels.ops.attention import (
            qwen38_qsa_sm121_varlen,
        )

        return qwen38_qsa_sm121_varlen
"""

patch_backend = Patch(
    name="route SM121 QSA decode to the KDA kernel (#36845)",
    target=BACKEND,
    when=_modules_ok and os.environ.get("TP", "") in ("1", "2", "4"),
)


@patch_backend.run
def apply_backend(p: Patch) -> None:
    # Variant 1: after p65 EDIT 2 (the normal case, p65 runs first).
    # Variant 2: pristine 99c9362e body, if p65 was skipped or drifted.
    p65_branch = """    from sglang.srt.utils import is_sm120_supported

    if is_sm120_supported():
        from sglang.kernels.ops.attention.flash_attention_v4 import (
"""
    pristine = """    flash-attn-4's cute interface serves the same call shape on Blackwell.
    \"\"\"
    try:
        from flash_attn import flash_attn_varlen_func
"""
    p.replace_any(
        [
            (p65_branch, _SM121_BRANCH + p65_branch),
            (pristine, pristine.replace("    try:\n", _SM121_BRANCH + "    try:\n", 1)),
        ],
        marker=BACKEND_MARKER,
        what="SM121 early-return in _resolve_flash_attn_varlen_func",
    )
