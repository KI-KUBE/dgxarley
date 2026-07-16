"""[dgxarley] DSA indexer: Triton-fused paged-MQA-logits fast path on the p30
torch fallback (the decode perf floor on GB10/SM121).

WHY (measured 2026-07-16, spark5 GB10, exact live shapes): with p34 the sparse
ATTENTION is native and fast, and decode sits at 8.4 tok/s = ~119 ms/token. The
p30 torch logits kernel costs 1.476 ms/layer-call x 79 layers = 116.6 ms/token
at the LIVE decode shape -- i.e. the indexer IS the whole floor, and the reason
is structural: under cuda-graph, `max_seq_len = block_tables.shape[1] * 64`
(dsa_indexer.py) is the CAPTURE-width of the page table (2048 pages = the full
131072-token KV budget), and the torch impl gathers + fp32-dequants + bmms the
FULL width for every request on every step, regardless of the true seq len
(~300). torch cannot do better under graph capture (shapes must be static).

Neither flashinfer 0.6.14 (image) nor flashinfer main (upstream tree scanned
2026-07-16) ships an indexer/fp8-paged-MQA-logits kernel; sgl_kernel covers only
the topk side (fast_topk*) and a DSV4 q-prep fusion. So this Triton kernel is
the remaining lever ("Option 2" of the user-approved plan in dsalogitrework.md
NEXT).

WHAT: one Triton program per (request, 64-token KV page). The launch grid stays
STATIC over the full page-table width (cuda-graph-safe), but each program
EARLY-EXITS on the true `seq_lens[b]` read at replay time -- cost tracks the
real context, not the capture width. Fused fp8-load + per-token-scale dequant +
q.k dot + relu + weighted head-sum + scale, no fp32 HBM intermediates.

GPU-verified on spark5 vs the p30 torch reference: BIT-EXACT (max|diff| = 0.0,
identical -inf masks) across bs 1/4/32, seq 300/2048/131072 at width 131072;
cuda-graph capture + replay correct INCLUDING a seq-len change in the static
buffer between capture and replay. Perf at the live decode point (bs=1, width
131072, seq 300): 0.024 ms/call vs 1.476 ms torch (61x) -> ~1.9 ms/token
indexer cost instead of ~116.6.

DELIVERY: (1) creates dsa/triton_paged_mqa_logits.py (new file), (2) inserts a
dispatch into the p30-generated dsa/torch_paged_mqa_logits.py right after its
shape asserts, gated by env `SGLANG_DSA_INDEXER_TRITON` (default ON; set "0" to
revert to the pure-torch path without a redeploy of code). The activation
contract is unchanged: everything still keys off
`dsa_paged_mqa_logits_backend: torch` -- this only makes that backend fast.
Falls back to torch when triton is unavailable or num_heads < 16 (tl.dot
minimum; GLM=32, DSv3.2=64, so never in practice).

ORDER: must run AFTER p30 (edits the file p30 creates) -- the p35 number
encodes that. RE-SYNC on image bump: if upstream ships a native SM12x indexer
logits kernel (watch deep_gemm SM121 support and flashinfer), DELETE this and
re-point the backend.
"""

from _patchlib import Patch

MARKER = "# [patch] _sgl_dsa_indexer_triton_"

TRITON_MODULE = '''# SPDX-License-Identifier: Apache-2.0
"""Triton-fused DSA paged-MQA-logits (drop-in fast path for the torch fallback).

[dgxarley] _sgl_dsa_indexer_triton_ -- see p35_dsa_indexer_triton_logits.py for
provenance, measurements and the re-sync rule. Bit-exact vs
fp8_paged_mqa_logits_torch_dsa (GPU-verified, spark5 GB10 2026-07-16).

Static launch grid over the full page-table width (cuda-graph-safe); per-block
early exit on the true seq len read at replay time.
"""

from typing import Any, Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _dsa_indexer_logits_kernel(
    q_ptr,        # fp8 [B, H, D] (contiguous)
    kv_ptr,       # fp8 flat: per 64-token block, 64*D values then 64 fp32 scales
    scale_ptr,    # SAME storage viewed as fp32, element-indexed
    w_ptr,        # fp32 [B, H]
    seq_ptr,      # int32 [B]
    pt_ptr,       # int32 [B, W]
    out_ptr,      # fp32 [B, MAX_SEQ]
    W: tl.constexpr,
    MAX_SEQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    pblk = tl.program_id(1)
    base = pblk * BLOCK
    rows = base + tl.arange(0, BLOCK)
    out_off = b * MAX_SEQ + rows
    out_mask = rows < MAX_SEQ

    seq = tl.load(seq_ptr + b)
    if base >= seq:
        tl.store(out_ptr + out_off, tl.full((BLOCK,), float("-inf"), tl.float32), mask=out_mask)
        return

    page = tl.load(pt_ptr + b * W + pblk).to(tl.int64)
    blk_base = page * (BLOCK * (D + 4))
    offs = blk_base + tl.arange(0, BLOCK)[:, None] * D + tl.arange(0, D)[None, :]
    kv = tl.load(kv_ptr + offs).to(tl.float32)                    # [BLOCK, D]
    s_base = page * (BLOCK * (D + 4) // 4) + (BLOCK * D // 4)
    scale = tl.load(scale_ptr + s_base + tl.arange(0, BLOCK))     # [BLOCK]

    q = tl.load(
        q_ptr + b * H * D + tl.arange(0, H)[:, None] * D + tl.arange(0, D)[None, :]
    ).to(tl.float32)                                              # [H, D]
    w = tl.load(w_ptr + b * H + tl.arange(0, H))                  # [H]

    score = tl.dot(kv, tl.trans(q))                               # [BLOCK, H]
    score = tl.maximum(score, 0.0)
    score = tl.sum(score * w[None, :], axis=1)                    # [BLOCK]
    score = score * scale

    valid = rows < seq
    score = tl.where(valid, score, float("-inf"))
    tl.store(out_ptr + out_off, score, mask=out_mask)


def fp8_paged_mqa_logits_triton_dsa(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Same signature/semantics as fp8_paged_mqa_logits_torch_dsa (bit-exact)."""
    _ = deep_gemm_metadata
    _ = clean_logits
    batch_size, _one, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    assert head_dim == 128 and block_size == 64
    if seq_lens.dim() > 1:
        seq_lens = seq_lens.squeeze(-1)
    table_width = page_table.shape[1]
    max_pages = (max_seq_len + block_size - 1) // block_size
    grid_w = min(table_width, max_pages)

    kv_flat = kvcache_fp8.reshape(-1)
    scale_view = kvcache_fp8.view(torch.uint8).view(torch.float32).reshape(-1)
    q = q_fp8.reshape(batch_size, num_heads, head_dim).contiguous()
    out = torch.empty(batch_size, max_seq_len, dtype=torch.float32, device=q_fp8.device)
    if grid_w * block_size < max_seq_len:
        out.fill_(float("-inf"))

    _dsa_indexer_logits_kernel[(batch_size, grid_w)](
        q,
        kv_flat,
        scale_view,
        weight.to(torch.float32).contiguous(),
        seq_lens.to(torch.int32).contiguous(),
        page_table.contiguous(),
        out,
        W=table_width,
        MAX_SEQ=max_seq_len,
        H=num_heads,
        D=head_dim,
        BLOCK=block_size,
        num_warps=4,
    )
    return out
'''

patch_new_file = Patch(
    name="create dsa/triton_paged_mqa_logits.py (fused indexer logits kernel)",
    target="sglang/srt/layers/attention/dsa/triton_paged_mqa_logits.py",
)

# Patch.run requires the target to exist; create it here (idempotent by content).
import os  # noqa: E402

_target_path = patch_new_file.path
if not os.path.isfile(_target_path):
    with open(_target_path, "w") as _fh:
        _fh.write(TRITON_MODULE)
    print(f"Patched triton_paged_mqa_logits.py: created ({len(TRITON_MODULE)} bytes)")
elif open(_target_path).read() != TRITON_MODULE:
    with open(_target_path, "w") as _fh:
        _fh.write(TRITON_MODULE)
    print("Patched triton_paged_mqa_logits.py: refreshed to current content")
else:
    print("[patch] triton_paged_mqa_logits.py: already current, skipping")


patch_torch_dispatch = Patch(
    name="torch indexer fallback: Triton fast-path dispatch (env SGLANG_DSA_INDEXER_TRITON)",
    target="sglang/srt/layers/attention/dsa/torch_paged_mqa_logits.py",
)


@patch_torch_dispatch.run
def apply_torch_dispatch(p: Patch) -> None:
    # Import + env gate at module level (after the FP8_DTYPE line p30 writes).
    p.replace(
        "FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn\n",
        "FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn\n\n" + MARKER + """
# Triton fused fast path (bit-exact, 61x at the live decode shape). Env
# SGLANG_DSA_INDEXER_TRITON=0 reverts to the pure-torch path without a code
# change. Guarded import: no triton -> silently stay on torch.
import os as _sgl_os

try:
    from sglang.srt.layers.attention.dsa.triton_paged_mqa_logits import (
        fp8_paged_mqa_logits_triton_dsa as _sgl_triton_logits,
    )
except Exception:  # pragma: no cover - triton unavailable
    _sgl_triton_logits = None
_SGL_USE_TRITON_LOGITS = (
    _sgl_triton_logits is not None
    and _sgl_os.environ.get("SGLANG_DSA_INDEXER_TRITON", "1") == "1"
)
""",
        marker=MARKER,
        what="triton import + env gate",
    )

    # Dispatch AFTER the shape asserts so all layout guarantees hold for the
    # kernel too. num_heads >= 16 is tl.dot's minimum (GLM=32, DSv3.2=64).
    p.replace(
        "    assert clean_logits == False\n",
        """    assert clean_logits == False
    if _SGL_USE_TRITON_LOGITS and num_heads >= 16:  # _sgl_dsa_indexer_triton_ dispatch
        return _sgl_triton_logits(
            q_fp8, kvcache_fp8, weight, seq_lens, page_table, None, max_seq_len
        )
""",
        what="triton dispatch after asserts",
    )
