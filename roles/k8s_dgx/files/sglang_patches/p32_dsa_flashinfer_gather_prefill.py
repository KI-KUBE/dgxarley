"""[dgxarley] DSA PREFILL fallback: reuse the SAME gather+dense-fa2 flashinfer_gather
implementation as decode (p31_dsa_flashinfer_gather.py, patch_dsa_backend). Every
dedicated DSA prefill kernel is dead on GB10/SM121 for the identical reason
(trtllm-gen FMHA ISA wall -- LIVE crash: "TllmGenFmhaRunner ... Unsupported
architecture" at forward_batch warmup/first request, since dsa_prefill_backend
defaults/resolves to "trtllm" whenever unset). flashmla/fa3/tilelang/aiter are
equally dead (DSA_speedup.md survey).

KEY FINDING (source-verified, 2026-07-16): SGLANG_DSA_FUSE_TOPK defaults to
TRUE, and with dsa_topk_backend="sgl-kernel" (our config),
_get_fused_topk_page_table() just returns topk_indices UNCHANGED -- i.e. the
page_table_1 handed to BOTH forward_decode's AND forward_extend's dispatch
chain is the SAME kind of tensor in both modes: [num_query_tokens, topk]
physical KV-cache-slot indices, -1-padded past the real (topk-clamped) context
length. forward_decode's num_query_tokens == batch_size (1 query token per
request); forward_extend's num_query_tokens == the total number of prefill/
extend tokens across the batch (many query tokens per request, ragged). The
ALREADY-VERIFIED-AND-DEPLOYED _forward_flashinfer_gather (decode patch above)
is written generically against q_nope.shape[0]/page_table_1.shape[-1] -- it
does not assume "one token per request" anywhere -- and its kv_len_arr source,
metadata.dsa_cache_seqlens_int32, is ALSO populated per-query-token for EXTEND
mode (dsa_backend.py's non-speculative-extend branch: dsa_cache_seqlens_int32
= compute_dsa_seqlens(seqlens_expanded, topk), where seqlens_expanded already
has one real-context-length entry per query token, matching decode's
per-request semantics exactly when extend_len==1). Net effect: NO changes to
_forward_flashinfer_gather are needed to reuse it for prefill -- only a new
dispatch branch in forward_extend.

There is no true "dense" code path on GB10 to fall back to: MHA_ONE_SHOT (the
only actually-dense prefill impl in this backend) is gated to
`device_sm == 90 or (100 <= device_sm < 110)` in set_dsa_prefill_impl -- SM121
never qualifies, so self.use_mha is unconditionally False here regardless of
sequence length. This is not a limitation for correctness: for a real context
length <= topk (2048; true for every short prompt, e.g. GSM8K/smoke tests),
the indexer's top-k selection has nothing to exclude and (by construction of
compute_dsa_seqlens/the -1 padding) degenerates to selecting the FULL causal
context, so gather+dense-fa2 computes the exact same result a true dense
causal prefill would. Genuine sparse selection (real context > topk) reuses
the identical code path but is UNVERIFIED this session (see docstring below)
-- deliberately NOT hard-blocked with a NotImplementedError, since doing so
safely would need extra per-request host-sync bookkeeping in a hot dispatch
path that itself risks a new correctness bug; the existing decode fallback
documents its own analogous open point (short-sequence kv_len_arr) the same
way rather than gating it, and this follows that precedent.
"""

from _patchlib import Patch

patch = Patch(
    name="flashinfer_gather PREFILL dispatch (reuses decode's gather + dense fa2 impl, phase 1)",
    target="sglang/srt/layers/attention/dsa_backend.py",
)

MARKER = "# [patch] _sgl_dsa_flashinfer_gather_prefill_"

OLD = """        elif dsa_impl == "aiter":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter_extend(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
            )
        else:
            raise ValueError(
                f"Unsupported {dsa_impl = } for forward_extend. Consider using an other attention backend."
            )"""

NEW = (
    """        elif dsa_impl == "aiter":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter_extend(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
            )

        elif dsa_impl == "flashinfer_gather":
            """
    + MARKER
    + """
            # Reuses the decode implementation UNCHANGED: page_table_1 here is
            # the same [num_query_tokens, topk] fused-topk physical-slot-index
            # tensor (see the sglang_launch.sh patch comment above this class
            # for the source trace proving this), and metadata.dsa_cache_seqlens_int32
            # is populated per-query-token for EXTEND mode by the caller of
            # forward_extend, matching _forward_flashinfer_gather's existing
            # kv_len_arr derivation with no changes needed.
            return self._forward_flashinfer_gather(
                q_nope=q_nope,
                q_rope=q_rope,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                metadata=metadata,
                k_scale=(
                    layer.k_scale_float
                    if getattr(layer, "k_scale_float", None) is not None
                    else 1.0
                ),
            )

        else:
            raise ValueError(
                f"Unsupported {dsa_impl = } for forward_extend. Consider using an other attention backend."
            )"""
)


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD, NEW, marker=MARKER, what="flashinfer_gather PREFILL dispatch")
