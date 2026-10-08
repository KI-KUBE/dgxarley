"""[dgxarley] Retune the TileLang DSA sparse-attention v1 tile on SM12x (GB10).

`tilelang_sparse_fwd()` calls `sparse_attention_fwd_kernel_v1/_v2(num_heads, d_v,
tail_dim, topk, sm_scale=...)` with the factory defaults. For v1 those are
block_I=64, num_stages=2, threads=256, which request 151552 B (H=16) / 169984 B
(H=32) of dynamic shared memory; GB10 allows 101376 B per block, so the kernel
launch fails.

This patch makes the v1 path (tail_dim == 0) pass block_I=32, num_stages=1,
threads=128 when the device is SM12x. GPU-verified on spark5 (2026-10-08,
image 0.5.21-sm121): launches for H=16 and H=32, relative error ~2.2e-3 against
a float32 gathered-attention reference. The v2 path (tail_dim > 0) has threads
hardcoded to 384 and no working tile on GB10; it is left untouched, as is every
non-SM12x device.

Gated on SGLANG_DSA_DECODE_BACKEND or SGLANG_DSA_PREFILL_BACKEND == "tilelang".

Anchor is identical in the v0.5.21 source tree and in the image's
sglang/kernels/ops/attention/dsa/tilelang_kernel.py.

DELETE WHEN upstream picks SM12x-safe v1 tile defaults (or the wrapper selects
them by device capability).
"""

from _patchlib import Patch, gate_env

TARGET = "sglang/kernels/ops/attention/dsa/tilelang_kernel.py"

patch = Patch(
    name="TileLang DSA v1 SM12x tile (block_I=32, stages=1, threads=128)",
    target=TARGET,
    when=gate_env("SGLANG_DSA_DECODE_BACKEND", "tilelang") or gate_env("SGLANG_DSA_PREFILL_BACKEND", "tilelang"),
)

OLD = """        kernel = kernel_factory(num_heads, d_v, tail_dim, topk, sm_scale=sm_scale)
        out = kernel(q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))  # type: ignore
"""
NEW = """        _tile_kwargs = (
            dict(block_I=32, num_stages=1, threads=128)  # dgxarley p68
            if tail_dim == 0 and torch.cuda.get_device_capability()[0] == 12
            else {}
        )
        kernel = kernel_factory(num_heads, d_v, tail_dim, topk, sm_scale=sm_scale, **_tile_kwargs)
        out = kernel(q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))  # type: ignore
"""


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD, NEW, marker="# dgxarley p68", what="tilelang_sparse_fwd SM12x v1 tile kwargs")
