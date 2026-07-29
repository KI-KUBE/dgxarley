"""[dgxarley] nvfp4-KV + speculation (NEXTN/EAGLE): fix silent corruption caused
by a dequant workspace that is never filled. Verify runs natively on FP4 (XQA),
the draft-extend prefix is dequantized position-preservingly.

THE FINDING (2026-07-29, image 0.5.16-dev-sm121, GB10/SM121, Qwen3.6-35B-A3B
NVFP4-MTP): `--kv-cache-dtype nvfp4` + NEXTN produces word salad without any
error message (GSM8K 0/10, `accept len` constantly at the maximum), while the
same configuration without speculation (18/20) and fp8-KV with speculation
(10/10) are healthy. Full chain of evidence: nvfp4_kv_mtp_plan.md sections 14-19.

CAUSE (upstream `main` identical, unfixed):
`flashinfer_backend.py::_prepare_dequant_workspace_metadata_for_extend` is gated
on `forward_mode.is_extend_without_speculative()`. TARGET_VERIFY and
DRAFT_EXTEND_V2 fall through: `dq_page_table` stays None, and the call site says
`prepare_workspace=self.dq_page_table is not None`, so under speculation the
workspace is NEVER filled but still read (a wrapper without custom_kv_indices
uses pool slot indices; the last prefill fill sits in the COMPACTED layout at
different offsets: stale AND shifted). In eager mode it crashes instead with
`TypeError: 'NoneType' object is not subscriptable`, because verify inherits the
STALE table of the last prefill and the extend fill then indexes
`extend_prefix_lens_cpu=None`.

FIX (three parts, verified live in E6/E7: GSM8K 10/10, accept len scattering
2.30-2.70, 47.2 tok/s with CUDA graphs):
  1. hybrid_attn_backend: route TARGET_VERIFY to the decode backend when that
     one reads FP4 natively. The prefill backend CANNOT read the FP4 cache
     correctly under speculation; the XQA decode kernel can do it natively
     (proven at kernel level: cos 1.0000 against the dequantized reference,
     including with q_len=3 + draft mask).
  2. trtllm_mha: exempt forward_extend for target_verify/draft_extend_v2 from
     the blanket raise and call the (anyway used there) decode kernel with
     FP4 cache + block scales + a bit-packed causal draft mask (XQA requires
     the mask from q_len>1 on; linear topk=1 chain). KV write goes through
     `_kv_write_scales` (the nvfp4 pool expects None -> scale table; fp8/bf16
     unchanged at layer.k_scale).
  3. flashinfer: instead of the (never/incorrectly running) extend fill,
     DRAFT_EXTEND_V2 gets a position-preserving PREFIX fill via
     `get_flashinfer_decode_dequant_workspace_kv_buffer` (the draft pool holds
     only the single MTP layer, hence cheap; in the ragged case the paged part
     reads only the prefix, the current chunk runs raw/ragged).

OPERATIONAL NOTE: as long as the prefix fill runs inside forward, the
draft-extend CUDA graph must stay off (`SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1`),
otherwise the host fill does not run on graph replay. Verify and draft-step
graphs are capture-safe (static mask/pool tensors).

RUNTIME GATES: all inserted paths only engage for `is_nvfp4_kvcache` resp.
`prefill_uses_dequant_workspace` AND speculative forward modes, so for fp8/bf16
configurations the patch is a no-op even when applied.

DELETABLE as soon as upstream fills the workspace under speculation OR computes
verify/draft-extend natively. Upstream issue planned (silent corruption +
eager TypeError, minimal repro: Llama-8B-NVFP4 + NEXTN on SM120).
"""

from _patchlib import Patch, target_contains

_TRTLLM = "sglang/srt/layers/attention/trtllm_mha_backend.py"
_HYBRID = "sglang/srt/layers/attention/hybrid_attn_backend.py"
_FLASHINFER = "sglang/srt/layers/attention/flashinfer_backend.py"

# Without the native-FP4 machinery (older images) there is nothing to fix.
_HAS_NATIVE_FP4 = target_contains(_TRTLLM, "decode_uses_native_fp4")


# --- Part 1: verify routing -------------------------------------------------

route = Patch(
    name="nvfp4 spec: route verify to the native FP4 decode backend",
    target=_HYBRID,
    when=_HAS_NATIVE_FP4,
)

OLD_ROUTE = """        elif forward_mode.is_target_verify():
            return (
                self.decode_backend
                if self.spec_attn_is_decode
                else self.prefill_backend
            )"""

NEW_ROUTE = """        elif forward_mode.is_target_verify():
            if getattr(self.decode_backend, "decode_uses_native_fp4", False):
                # [dgxarley-nvfp4-spec] the prefill backend cannot read the
                # FP4 cache under speculation (the dequant workspace is never
                # filled for verify); the decode kernel reads it natively.
                return self.decode_backend
            return (
                self.decode_backend
                if self.spec_attn_is_decode
                else self.prefill_backend
            )"""


@route.run
def apply_route(p: Patch) -> None:
    p.replace(OLD_ROUTE, NEW_ROUTE, marker="[dgxarley-nvfp4-spec]", what="verify routing")


# --- Part 2: trtllm_mha native FP4 verify -----------------------------------

trtllm = Patch(
    name="nvfp4 spec: native FP4 verify/draft-extend in the trtllm_mha backend",
    target=_TRTLLM,
    when=_HAS_NATIVE_FP4,
)

OLD_GUARD = """        if self.decode_uses_native_fp4:
            raise RuntimeError(
                "TRTLLM MHA with native FP4 KV cache supports decode only; "
                "use a separate prefill backend such as flashinfer or triton."
            )"""

NEW_GUARD = """        if self.decode_uses_native_fp4 and not (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):  # [dgxarley-nvfp4-guard] verify/draft-extend-v2 use the decode
            # kernel below, which reads FP4 natively; only real prefill stays
            # forbidden.
            raise RuntimeError(
                "TRTLLM MHA with native FP4 KV cache supports decode only; "
                "use a separate prefill backend such as flashinfer or triton."
            )"""

OLD_WRITE = """                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                        k,
                        v,
                        layer.k_scale,
                        layer.v_scale,
                    )"""

NEW_WRITE = """                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                        k,
                        v,
                        # [dgxarley-nvfp4-write] the nvfp4 pool expects None ->
                        # scale table; identical to before for fp8/bf16.
                        *self._kv_write_scales(layer),
                    )"""

OLD_READ = """        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        # [num_pages, page_size, num_kv_heads, head_dim] -> [num_pages, num_kv_heads, page_size, head_dim]
        k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_cache = k_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)
        v_cache = v_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)

        if layer.tp_k_head_num == 1:
            k_cache = canonicalize_stride(k_cache)
        if layer.tp_v_head_num == 1:
            v_cache = canonicalize_stride(v_cache)

        kv_cache = (k_cache, v_cache)"""

NEW_READ = """        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        # [dgxarley-nvfp4-read] native FP4 path, same as in forward_decode
        kv_cache_sf = None
        if self.is_nvfp4_kvcache:
            kv_cache, kv_cache_sf = self._get_nvfp4_decode_kv_cache(layer)
        else:
            # [num_pages, page_size, num_kv_heads, head_dim] -> [num_pages, num_kv_heads, page_size, head_dim]
            k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            k_cache = k_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            ).permute(0, 2, 1, 3)
            v_cache = v_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            ).permute(0, 2, 1, 3)

            if layer.tp_k_head_num == 1:
                k_cache = canonicalize_stride(k_cache)
            if layer.tp_v_head_num == 1:
                v_cache = canonicalize_stride(v_cache)

            kv_cache = (k_cache, v_cache)"""

OLD_SCALES = """        # sink: additional value per head in the denominator of the softmax.
        attention_sink = kwargs.get("sinks", None)
        bmm1_scale, bmm2_scale = self._get_bmm_scales(layer, q_scale)"""

NEW_SCALES = """        # sink: additional value per head in the denominator of the softmax.
        attention_sink = kwargs.get("sinks", None)
        if self.is_nvfp4_kvcache:  # [dgxarley-nvfp4-scales]
            _fp4_k_scale, _fp4_v_scale = self._get_nvfp4_bmm_scales(layer)
            bmm1_scale = q_scale * _fp4_k_scale * layer.scaling
            bmm2_scale = _fp4_v_scale
        else:
            bmm1_scale, bmm2_scale = self._get_bmm_scales(layer, q_scale)"""

OLD_CALL = """                o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                    query=q,
                    kv_cache=kv_cache,
                    workspace_buffer=self.workspace_buffer,
                    block_tables=page_table,
                    seq_lens=self.forward_metadata.cache_seqlens_int32,
                    max_seq_len=self.max_context_len,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    window_left=layer.sliding_window_size,
                    sinks=attention_sink,
                    skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
                    out_dtype=self.q_data_type,
                    q_len_per_req=self.forward_metadata.max_seq_len_q,
                )"""

NEW_CALL = """                _q_len = int(self.forward_metadata.max_seq_len_q)
                o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                    query=q,
                    kv_cache=kv_cache,
                    workspace_buffer=self.workspace_buffer,
                    block_tables=page_table,
                    seq_lens=self.forward_metadata.cache_seqlens_int32,
                    max_seq_len=self.max_context_len,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    window_left=layer.sliding_window_size,
                    sinks=attention_sink,
                    skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
                    out_dtype=self.q_data_type,
                    q_len_per_req=_q_len,
                    # [dgxarley-nvfp4-call] from q_len>1 on, XQA requires the
                    # bit-packed draft block mask; without FP4 both stay None
                    # (behaviour unchanged).
                    kv_cache_sf=kv_cache_sf,
                    mask=(
                        self._dgxarley_causal_draft_mask(q.shape[0] // _q_len, _q_len)
                        if self.is_nvfp4_kvcache and _q_len > 1
                        else None
                    ),
                )
                if self.is_nvfp4_kvcache and o.dtype != self.q_data_type:
                    o = o.to(self.q_data_type)"""

OLD_HELPER = """    def _get_nvfp4_bmm_scales(self, layer: RadixAttention) -> tuple[float, float]:
        assert self.is_nvfp4_kvcache
        return self.kv_cache_quant_method.get_bmm_scales(layer.layer_id)"""

NEW_HELPER = """    def _get_nvfp4_bmm_scales(self, layer: RadixAttention) -> tuple[float, float]:
        assert self.is_nvfp4_kvcache
        return self.kv_cache_quant_method.get_bmm_scales(layer.layer_id)

    def _dgxarley_causal_draft_mask(self, bs: int, q_len: int) -> "torch.Tensor":
        # [dgxarley-nvfp4-mask] XQA speculation mask: bit-packed, uint16,
        # [bs, q_len, ((q_len+31)//32)*2]; row i allows draft positions
        # 0..i (linear topk=1 chain). Cached per (bs, q_len) so the tensor
        # stays stable under CUDA-graph capture.
        assert q_len <= 31, "draft mask only implemented for q_len <= 31"
        cache = getattr(self, "_dgxarley_mask_cache", None)
        if cache is None:
            cache = {}
            self._dgxarley_mask_cache = cache
        m = cache.get((bs, q_len))
        if m is None:
            msr32 = (q_len + 31) // 32
            m32 = torch.zeros(bs, q_len, msr32, dtype=torch.int32)
            for i in range(q_len):
                m32[:, i, 0] = (1 << (i + 1)) - 1
            m = m32.view(torch.uint16).to(self.device)
            cache[(bs, q_len)] = m
        return m"""


@trtllm.run
def apply_trtllm(p: Patch) -> None:
    p.replace(OLD_GUARD, NEW_GUARD, marker="[dgxarley-nvfp4-guard]", what="guard")
    p.replace(OLD_WRITE, NEW_WRITE, marker="[dgxarley-nvfp4-write]", what="kv write")
    p.replace(OLD_READ, NEW_READ, marker="[dgxarley-nvfp4-read]", what="kv read")
    p.replace(OLD_SCALES, NEW_SCALES, marker="[dgxarley-nvfp4-scales]", what="scales")
    p.replace(OLD_CALL, NEW_CALL, marker="[dgxarley-nvfp4-call]", what="kernel call")
    p.replace(OLD_HELPER, NEW_HELPER, marker="[dgxarley-nvfp4-mask]", what="mask helper")


# --- Part 3: draft-extend prefix fill ---------------------------------------

dfill = Patch(
    name="nvfp4 spec: draft-extend prefix fill (DRAFT_EXTEND_V2)",
    target=_FLASHINFER,
    when=target_contains(
        "sglang/srt/mem_cache/memory_pool.py",
        "get_flashinfer_decode_dequant_workspace_kv_buffer",
    ),
)

OLD_DFILL = """                prepare_workspace=self.dq_page_table is not None,
                use_ragged=self.forward_metadata.use_ragged,
                k_cur=k,
                v_cur=v,
            )"""

NEW_DFILL = """                prepare_workspace=self.dq_page_table is not None
                and not forward_batch.forward_mode.is_draft_extend_v2(),
                use_ragged=self.forward_metadata.use_ragged,
                k_cur=k,
                v_cur=v,
            )
            if forward_batch.forward_mode.is_draft_extend_v2():
                # [dgxarley-nvfp4-dfill] DRAFT_EXTEND_V2 falls through the
                # is_extend_without_speculative() gate of the extend fill:
                # the workspace would be read stale. Instead dequantize the
                # PREFIX position-preservingly, since slot indices == read
                # indices of the paged wrapper (no custom_kv_indices under
                # speculation); the current chunk runs ragged over the raw
                # k/v. Under V2, seq_lens already include the new tokens.
                kv_cache = pool.get_flashinfer_decode_dequant_workspace_kv_buffer(
                    layer,
                    self.req_to_token_pool.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens - forward_batch.extend_seq_lens,
                )"""


@dfill.run
def apply_dfill(p: Patch) -> None:
    p.replace(OLD_DFILL, NEW_DFILL, marker="[dgxarley-nvfp4-dfill]", what="draft prefix fill")
