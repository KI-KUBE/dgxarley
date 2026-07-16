"""[dgxarley] dsa_backend.py: flashinfer_gather CUDA-GRAPH plan/run-split (dsa_cuda_graph_plan.md).

The decode/prefill flashinfer_gather patches above (p31_dsa_flashinfer_gather.py,
p32_dsa_flashinfer_gather_prefill.py) call wrapper.plan() INLINE in
_forward_flashinfer_gather every forward. plan() does host-side stream sync /
alloc that is NOT cuda-graph-recordable -> crash at decode-graph capture (the
reason disable_cuda_graph was the eager workaround). This block splits it exactly
like the native FlashInferMLAAttnBackend: build a per-bs wrapper(use_cuda_graph=
True) OUT of the captured region, call the REAL .plan() once, monkeypatch
wrapper.plan -> fast_mla_decode_plan (a module-level, causal-generic fast variant
that skips the stream sync), then INSIDE the captured region call ONLY
wrapper.run(). Validated bit-exact vs eager on synthetic tensors (scratchpad/
microtest.py, incl. causal=True and short-seq kv_len) before deploy.

INERT unless attention_backend=dsa + dsa_decode_backend=flashinfer_gather (the
gated dispatch is only reached then). Eager path (prefill/extend, or
disable_cuda_graph decode) keeps the original inline-plan behavior UNCHANGED.

Ordering: this patch runs after p31 (adds _forward_flashinfer_gather + the eager
wrapper init slot + the decode dispatch elif) and p32 (the PREFILL dispatch,
reusing the same method via is_decode). Every anchor below except S6 exists in
the file ONLY because of those two patches; S6
(init_forward_metadata_out_graph's _apply_cuda_graph_metadata call) is untouched
by them and matches pristine SGLang directly. Seven edits, same S1-S7 tags the
original bash block used:

  S1 __init__     -- expand the single eager wrapper slot into per-bs graph state.
  S2 signature    -- add is_decode so the prefill dispatch can force the eager path.
  S3 head         -- drop the build-eager-wrapper-at-top (moved into the eager branch).
  S4 tail         -- replace the inline plan()+run() with the graph(run-only)/eager
                     (plan+run) split.
  S5 new methods  -- _fig_build_graph_wrapper + _fig_replan_graph, inserted right
                     before _forward_flashinfer_gather.
  S6 out_graph    -- init_forward_metadata_out_graph: out-of-graph plan/replan hook.
  S7 prefill      -- forward_extend's flashinfer_gather call passes is_decode=False
                     (anchored on the forward_extend ValueError, unique to prefill;
                     decode raises an assert instead).

Full design + rationale: dsa_cuda_graph_plan.md.
"""

from _patchlib import Patch

patch = Patch(
    name="flashinfer_gather CUDA-GRAPH plan/run-split",
    target="sglang/srt/layers/attention/dsa_backend.py",
)

MARKER = "# [patch] _sgl_dsa_fig_graph_split_"

# S1: __init__ -- expand the single eager wrapper slot into the per-bs graph state.
S1_OLD = """        self._flashinfer_gather_wrapper = None"""

S1_NEW = (
    S1_OLD
    + """
        """
    + MARKER
    + """
        # per-bs cuda-graph wrappers (plan/run split, dsa_cuda_graph_plan.md). The
        # eager slot above stays for prefill/extend + disable_cuda_graph decode
        # (inline plan); these back the CAPTURED-decode path (run-only in graph).
        self._flashinfer_gather_wrappers = {}   # bs -> BatchMLAPagedAttentionWrapper(use_cuda_graph=True)
        self._fig_static = {}                    # bs -> {qo_cpu, kv_indptr_cpu, kv_indices, kv_len_buf}
        self._fig_plan_params = None             # (num_heads, ckv_d, kpe_d, sm_scale, q_dtype, kv_dtype); model-wide const"""
)

# S2: signature -- add is_decode so the prefill dispatch can force the eager path.
S2_OLD = """        metadata: "DSAMetadata",
        k_scale: float = 1.0,
    ) -> torch.Tensor:"""

S2_NEW = """        metadata: "DSAMetadata",
        k_scale: float = 1.0,
        is_decode: bool = True,
    ) -> torch.Tensor:"""

# S3: head -- drop the build-eager-wrapper-at-top (moved into the eager branch).
S3_OLD = """        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        if self._flashinfer_gather_wrapper is None:
            self._flashinfer_gather_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer, backend="fa2"
            )
        wrapper = self._flashinfer_gather_wrapper

        num_tokens_q = q_nope.shape[0]"""

S3_NEW = """        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        num_tokens_q = q_nope.shape[0]"""

# S4: tail -- replace the inline plan()+run() with the graph(run-only)/eager(plan+run) split.
S4_OLD = """        qo_indptr = torch.arange(0, num_tokens_q + 1, device=device, dtype=torch.int32)
        # page_size=1 post-gather: the freshly gathered/dequantized buffer is
        # already dense per request, so a plain sequential index is correct
        # (no second indirection into the original packed KV cache needed).
        kv_indptr = qo_indptr * topk
        kv_indices = torch.arange(
            0, num_tokens_q * topk, device=device, dtype=torch.int32
        )
        # Real per-request valid KV count (<=topk; short sequences leave a
        # padded tail in page_table_1 -- see dsalogitrework.md "Open points").
        kv_len_arr = metadata.dsa_cache_seqlens_int32.clamp(max=topk).to(torch.int32)

        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            num_heads,
            v_head_dim,
            kpe.shape[-1],
            1,
            True,
            sm_scale,
            q_nope.dtype,
            ckv.dtype,
        )
        return wrapper.run(q_nope, q_rope, ckv, kpe, return_lse=False)"""

S4_NEW = """        bs = num_tokens_q
        # Stash the model-wide-constant plan params from the layer (available HERE,
        # not in init_forward_metadata_out_graph). MLA scaling/dims are identical
        # across all attention layers, so a one-time capture is correct.
        if self._fig_plan_params is None:
            self._fig_plan_params = (
                num_heads, v_head_dim, kpe.shape[-1], sm_scale, q_nope.dtype, ckv.dtype
            )

        # decode_cuda_graph_metadata[bs] exists ONLY for cuda-graph-captured decode
        # batch sizes (never for prefill/extend, never under disable_cuda_graph).
        # is_decode excludes the prefill dispatch (which reuses this method but must
        # always take the eager inline-plan path).
        graph_meta = getattr(self, "decode_cuda_graph_metadata", None)
        is_graph_decode = (
            is_decode
            and graph_meta is not None
            and graph_meta.get(bs, None) is not None
        )

        if is_graph_decode:
            # plan/run-split: the wrapper for this bs was planned OUT of the captured
            # region (either here during the uncaptured warmup run_once, or in
            # init_forward_metadata_out_graph at capture-prep). Inside the captured
            # region we call ONLY wrapper.run().
            if bs not in self._flashinfer_gather_wrappers:
                assert not torch.cuda.is_current_stream_capturing(), (
                    "flashinfer_gather graph wrapper missing for bs=%d inside capture "
                    "(warmup run_once / out_graph should have built it)" % bs
                )
                self._fig_build_graph_wrapper(
                    bs, num_heads, v_head_dim, kpe.shape[-1], sm_scale,
                    q_nope.dtype, ckv.dtype, metadata, topk,
                )
            wrapper = self._flashinfer_gather_wrappers[bs]
            return wrapper.run(q_nope, q_rope, ckv, kpe, return_lse=False)

        # Eager path (prefill/extend, or disable_cuda_graph decode): reusable
        # non-graph wrapper, plan() inline every call (original verified behavior).
        if self._flashinfer_gather_wrapper is None:
            self._flashinfer_gather_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer, backend="fa2"
            )
        wrapper = self._flashinfer_gather_wrapper
        qo_indptr = torch.arange(0, num_tokens_q + 1, device=device, dtype=torch.int32)
        # page_size=1 post-gather: the gathered/dequantized buffer is already dense
        # per request, so a plain sequential index is correct.
        kv_indptr = qo_indptr * topk
        kv_indices = torch.arange(
            0, num_tokens_q * topk, device=device, dtype=torch.int32
        )
        kv_len_arr = metadata.dsa_cache_seqlens_int32.clamp(max=topk).to(torch.int32)
        wrapper.plan(
            qo_indptr, kv_indptr, kv_indices, kv_len_arr,
            num_heads, v_head_dim, kpe.shape[-1], 1, True, sm_scale,
            q_nope.dtype, ckv.dtype,
        )
        return wrapper.run(q_nope, q_rope, ckv, kpe, return_lse=False)"""

# S5: two new methods, inserted right before _forward_flashinfer_gather.
S5_OLD = """    def _forward_flashinfer_gather(
        self,
        q_nope: torch.Tensor,"""

S5_NEW = '''    def _fig_build_graph_wrapper(
        self, bs, num_heads, head_dim_ckv, head_dim_kpe,
        sm_scale, q_dtype, kv_dtype, metadata, topk,
    ):
        """Build + REAL-plan a per-bs cuda-graph flashinfer_gather wrapper ONCE, then
        monkeypatch its .plan to fast_mla_decode_plan (skips the non-graph stream sync
        on every subsequent replay). Mirrors FlashInferMLAAttnBackend's capture path.
        Post-gather addressing is fully static given (bs, topk); only kv_len_arr is
        dynamic (updated in place each replay by _fig_replan_graph)."""
        from functools import partial
        from flashinfer.mla import BatchMLAPagedAttentionWrapper
        from sglang.srt.layers.attention.flashinfer_mla_backend import fast_mla_decode_plan

        dev = metadata.dsa_cache_seqlens_int32.device
        qo_indptr = torch.arange(0, bs + 1, device=dev, dtype=torch.int32)
        kv_indptr = qo_indptr * topk
        kv_indices = torch.arange(0, bs * topk, device=dev, dtype=torch.int32)
        kv_len_buf = torch.empty(bs, device=dev, dtype=torch.int32)
        kv_len_buf.copy_(
            metadata.dsa_cache_seqlens_int32[:bs].clamp(max=topk).to(torch.int32)
        )
        wrapper = BatchMLAPagedAttentionWrapper(
            self.workspace_buffer, use_cuda_graph=True,
            qo_indptr=qo_indptr, kv_indptr=kv_indptr,
            kv_indices=kv_indices, kv_len_arr=kv_len_buf, backend="fa2",
        )
        # REAL plan once (populates wrapper._cached_module for the fast variant).
        wrapper.plan(
            qo_indptr, kv_indptr, kv_indices, kv_len_buf,
            num_heads, head_dim_ckv, head_dim_kpe, 1, True, sm_scale, q_dtype, kv_dtype,
        )
        wrapper.plan = partial(fast_mla_decode_plan, wrapper)
        self._flashinfer_gather_wrappers[bs] = wrapper
        self._fig_static[bs] = {
            "qo_cpu": qo_indptr.cpu(),
            "kv_indptr_cpu": kv_indptr.cpu(),
            "kv_indices": kv_indices,
            "kv_len_buf": kv_len_buf,
        }

    def _fig_replan_graph(self, bs, metadata):
        """Out-of-graph capture-prep / replay-prep: refresh kv_len (the one dynamic
        quantity) and re-run the FAST plan (no stream sync). Builds the wrapper lazily
        if the params are already stashed (capture-prep before the warmup run_once);
        no-op until either the wrapper exists or params are known."""
        if self.dsa_index_topk is None:
            return
        topk = self.dsa_index_topk
        if bs not in self._flashinfer_gather_wrappers:
            if self._fig_plan_params is None:
                return  # will be built by the uncaptured warmup run_once instead
            nh, ckv_d, kpe_d, sm, qd, kd = self._fig_plan_params
            self._fig_build_graph_wrapper(bs, nh, ckv_d, kpe_d, sm, qd, kd, metadata, topk)
            return
        nh, ckv_d, kpe_d, sm, qd, kd = self._fig_plan_params
        st = self._fig_static[bs]
        st["kv_len_buf"].copy_(
            metadata.dsa_cache_seqlens_int32[:bs].clamp(max=topk).to(torch.int32)
        )
        wrapper = self._flashinfer_gather_wrappers[bs]
        wrapper.plan(
            st["qo_cpu"], st["kv_indptr_cpu"], st["kv_indices"], st["kv_len_buf"].cpu(),
            nh, ckv_d, kpe_d, 1, True, sm, qd, kd,
        )

    def _forward_flashinfer_gather(
        self,
        q_nope: torch.Tensor,'''

# S6: init_forward_metadata_out_graph -- add the out-of-graph plan/replan hook.
S6_OLD = """        self._apply_cuda_graph_metadata(
            bs=forward_batch.batch_size,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            forward_mode=forward_batch.forward_mode,
            spec_info=forward_batch.spec_info,
            out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
            actual_forward_mode=getattr(forward_batch, "actual_forward_mode", None),
        )"""

S6_NEW = (
    S6_OLD
    + """
        """
    + MARKER
    + """
        # Out-of-graph plan/replan for the flashinfer_gather captured-decode wrapper
        # (dsa_cuda_graph_plan.md): builds it at capture-prep if params are known, and
        # fast-replans (fresh kv_len) before every replay. INERT for any other backend.
        if (
            self.dsa_decode_impl == "flashinfer_gather"
            and forward_batch.forward_mode.is_decode_or_idle()
        ):
            _fig_meta = self.decode_cuda_graph_metadata.get(forward_batch.batch_size)
            if _fig_meta is not None:
                self._fig_replan_graph(forward_batch.batch_size, _fig_meta)"""
)

# S7: prefill dispatch -- force the eager path (is_decode=False). Anchored on the
# forward_extend ValueError, which is UNIQUE to the prefill method (decode raises an
# assert), so this targets the prefill call, not the identical decode call.
S7_OLD = """                metadata=metadata,
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

S7_NEW = """                metadata=metadata,
                is_decode=False,
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


@patch.run
def apply(p: Patch) -> None:
    # No shared `marker=` across calls: S1 and S6 both inject MARKER text, and the
    # framework checks the probe BEFORE the anchor -- passing MARKER to both would
    # make S6 silently report "already applied" the instant S1 lands in the same
    # run (same failure mode p31's docstring warns about). Each edit's own (unique,
    # verbatim) new-text is a safe default probe instead.
    p.replace(S1_OLD, S1_NEW, what="S1-init-state")
    p.replace(S2_OLD, S2_NEW, what="S2-signature")
    p.replace(S3_OLD, S3_NEW, what="S3-head")
    p.replace(S4_OLD, S4_NEW, what="S4-tail-split")
    p.replace(S5_OLD, S5_NEW, what="S5-new-methods")
    p.replace(S6_OLD, S6_NEW, what="S6-out_graph-hook")
    p.replace(S7_OLD, S7_NEW, what="S7-prefill-is_decode")
