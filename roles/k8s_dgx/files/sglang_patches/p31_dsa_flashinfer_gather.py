"""[dgxarley] DSA attention decode: gather + reuse dense fa2 (GB10/SM121, gather+reuse, not a new kernel).

Every DEDICATED DSA attention kernel is dead on GB10/SM121: trtllm-gen FMHA is a
datacenter-Blackwell-only ISA (live crash: "Unsupported architecture" at
TllmGenFmhaRunner autotune), flashmla_sparse/flashmla_kv's sgl_kernel extension is
not built in this image, fa3 has a hard SM90/SM100-only gate, and tilelang compiles
on SM121 but has a proven smem-vs-compile contradiction (no block_I both fits the
~99 KB budget and compiles). Full survey + verdict: DSA_speedup.md.

Instead of a new kernel: gather the indexer's top-k selected KV (the "gather" prep
ALREADY exists -- dsa_backend.py::forward_decode builds
page_table_1 = transform_index_page_table_decode(page_table, topk_indices, page_size=1)
for every backend) and run flashinfer's DENSE MLA decode (backend="fa2", the SAME
kernel that already serves this model's dense baseline on SM121) over the small
gathered subset. flashinfer's MLA wrapper rejects fp8 kv_data_type outside SM90/fa3,
so the gathered KV must be dequantized to bf16 first -- reusing SGLang's OWN
dequantize_k_cache_paged (already imported in dsa_backend.py, already used by the
flashmla_sparse RAGGED-prefill path for exactly this purpose), not reinvented.
Full design + numeric verification: dsalogitrework.md PART 2.

New OPT-IN decode backend value "flashinfer_gather" (dsa_decode_backend), analogous
to the paged-mqa-logits "torch" backend above: not selected by auto-detection, no
arch gate, zero behavior change for every model/arch that doesn't explicitly select
it. PHASE 1 ONLY: plain single-token decode. MTP/NEXTN target-verify (next_n>=2, via
forward_extend's dsa_decode_impl reuse) is not wired here and falls through to
forward_extend's existing "unsupported" handling -- moot for now since MTP stays off
until this path is live-validated.

Two files, two Patch objects:
  A) server_args.py -- adds "flashinfer_gather" to DSA_CHOICES.
  B) dsa_backend.py -- three edits (init/dispatch/method), gated together in the
     original bash guard by a single marker on the init edit; here each edit
     carries its own idempotency probe (the framework buffers edits per-file, so
     reusing the init marker as the probe for the later edits would falsely
     report "already applied" the moment the init edit lands in the same run).
"""

from _patchlib import Patch

# --- File A: server_args.py -- add "flashinfer_gather" to DSA_CHOICES ---

patch_server_args = Patch(name="DSA_CHOICES: add flashinfer_gather", target="sglang/srt/server_args.py")

MARKER_A = "# [patch] _sgl_dsa_flashinfer_gather_choice_"

OLD_A = """DSA_CHOICES = [
    "flashmla_sparse",
    "flashmla_kv",
    "flashmla_auto",
    "fa3",
    "tilelang",
    "aiter",
    "trtllm",
]"""

NEW_A = MARKER_A + """
DSA_CHOICES = [
    "flashmla_sparse",
    "flashmla_kv",
    "flashmla_auto",
    "fa3",
    "tilelang",
    "aiter",
    "trtllm",
    "flashinfer_gather",
]"""


@patch_server_args.run
def apply_a(p: Patch) -> None:
    p.replace(OLD_A, NEW_A, marker=MARKER_A, what="DSA_CHOICES flashinfer_gather")


# --- File B: dsa_backend.py ---

patch_dsa_backend = Patch(
    name="flashinfer_gather decode backend (gather + dense fa2, phase 1)",
    target="sglang/srt/layers/attention/dsa_backend.py",
)

MARKER_B_INIT = "# [patch] _sgl_dsa_flashinfer_gather_init_"

# B1: __init__ -- cache slot for the lazily-built wrapper.
OLD_INIT = """        # Allocate global workspace buffer for TRT-LLM kernels (ragged attention on SM100/B200, or trtllm decode)
        if self.device_sm_major >= 10 or self.dsa_decode_impl == "trtllm":
            global global_workspace_buffer
            if global_workspace_buffer is None:
                global_workspace_buffer = torch.empty(
                    envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                    dtype=torch.uint8,
                    device=model_runner.device,
                )
            self.workspace_buffer = global_workspace_buffer
        else:
            self.workspace_buffer = None"""

NEW_INIT = (
    OLD_INIT
    + """

        """
    + MARKER_B_INIT
    + """
        # gather+dense-fa2 fallback (dsalogitrework.md PART 2): reuse the working
        # dense MLA decode kernel over the indexer's gathered+dequantized top-k KV,
        # since every dedicated DSA attention kernel is dead on GB10/SM121. Built
        # lazily on first use in _forward_flashinfer_gather (needs self.workspace_buffer,
        # set just above).
        self._flashinfer_gather_wrapper = None"""
)

# B2: forward_decode dispatch -- new elif branch before the final else/assert.
OLD_DISPATCH = """elif self.dsa_decode_impl == "aiter":
            if q_all is None or not _is_hip:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
                metadata=metadata,
                bs=forward_batch.batch_size,
            )

        else:
            assert False, f"Unsupported {self.dsa_decode_impl = }\""""

NEW_DISPATCH = """elif self.dsa_decode_impl == "aiter":
            if q_all is None or not _is_hip:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
                metadata=metadata,
                bs=forward_batch.batch_size,
            )

        elif self.dsa_decode_impl == "flashinfer_gather":
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
            assert False, f"Unsupported {self.dsa_decode_impl = }\""""

# B3: new method, inserted right before _forward_fa3.
OLD_METHOD_ANCHOR = "    def _forward_fa3(\n"

NEW_METHOD = '''    def _forward_flashinfer_gather(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        sm_scale: float,
        v_head_dim: int,
        metadata: "DSAMetadata",
        k_scale: float = 1.0,
    ) -> torch.Tensor:
        """Phase 1 (dsalogitrework.md PART 2, plain decode / next_n==1 only).

        Every dedicated DSA attention kernel is dead on GB10/SM121 (trtllm-gen
        FMHA = datacenter-only ISA, flashmla = extension not built in this
        image, fa3 = hard SM90/SM100 gate, tilelang = proven smem/compile
        contradiction). Instead of a new kernel: gather the indexer's top-k
        selected KV and run flashinfer's DENSE MLA decode (backend="fa2", the
        kernel that already serves this model's dense baseline on SM121) over
        the small gathered+dequantized subset.

        flashinfer's MLA wrapper rejects fp8 kv_data_type outside SM90/fa3
        (dsalogitrework.md Section 2 "THE blocker"), which is why the dequant
        to bf16 happens BEFORE the wrapper, not inside it.

        FIXED 2026-07-16 (live crash: "AssertionError: dim_quant: 576 != 656
        detected in dequantize_k_cache_paged"): the KV pool's byte layout is
        NOT always the 656-byte packed/block-quantized layout that
        dequantize_k_cache_paged hardcodes. Per
        model_runner_kv_cache_mixin.py::calculate_mla_kv_cache_dim, that packed
        layout (dsa_kv_cache_store_fp8=True, 512 fp8 nope + 16 scale bytes +
        128 bf16-rope bytes = 656) is used ONLY when dsa_prefill_backend and
        dsa_decode_backend are both NOT "trtllm" (and, on HIP, not
        tilelang/aiter). Our deployment keeps dsa_prefill_backend="trtllm", so
        the pool is ALWAYS the plain layout (dim = kv_lora_rank +
        qk_rope_head_dim = 576): nope and rope are simply cast to fp8_e4m3
        directly at write time (memory_pool.py set_mla_kv_buffer's "else"
        branch), no per-block scale stored -- a single scalar k_scale (mirrors
        _forward_trtllm's own bmm1_scale derivation) applies uniformly on
        dequant. Branch on self.dsa_kv_cache_store_fp8 so BOTH pool layouts are
        handled correctly (not just our deployment's config) -- the original
        dequantize_k_cache_paged path is kept for when the packed layout really
        is in use, per the parent investigation's "do not force the 656
        assumption" directive.

        Numerically verified (2026-07-16, GPU pod) against an independent
        manual gather + dequant + softmax reference in BOTH byte layouts (the
        formerly-tested 656 packed layout AND, after this fix, the real
        production 576 plain layout): max abs diff ~0.0008-0.0012 (bf16-level),
        no NaN/all-zero, seed-varying. Open point (dsalogitrework.md):
        kv_len_arr masking for requests with real context < topk is wired
        (clamp + flashinfer's own kv_len_arr) but not proven correct
        end-to-end, only plumbing-tested.
        """
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        if self._flashinfer_gather_wrapper is None:
            self._flashinfer_gather_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer, backend="fa2"
            )
        wrapper = self._flashinfer_gather_wrapper

        num_tokens_q = q_nope.shape[0]
        num_heads = q_nope.shape[1]
        topk = page_table_1.shape[-1]
        device = q_nope.device

        # (num_tokens_q * topk, 1, kv_lora_rank + qk_rope_head_dim), bf16.
        if self.dsa_kv_cache_store_fp8:
            # Packed block-quantized layout (656 bytes/token). See docstring.
            gathered = dequantize_k_cache_paged(kv_cache, page_table_1.reshape(-1))
        else:
            # Plain raw layout (576 = kv_lora_rank + qk_rope_head_dim here, but
            # derived from the buffer itself, not hardcoded): flat per-token
            # fp8 slots, gather by the same flattened page_table_1 index
            # dequantize_k_cache_paged would have used, then a single-scalar
            # fp8->bf16 dequant (no per-block scale to unpack).
            flat_kv_cache = kv_cache.view(-1, kv_cache.shape[-1])
            gathered_fp8 = flat_kv_cache[page_table_1.reshape(-1).long()]
            # fp32 intermediate for the scale multiply (matches the existing
            # _dequantize_k_cache_fast_kernel Triton convention: load+cast to
            # fp32, multiply by scale, THEN cast down to bf16) -- avoids
            # extra bf16 rounding when k_scale != 1.0. A no-op precision-wise
            # when k_scale == 1.0 (today's default; see docstring).
            gathered = (
                (gathered_fp8.to(torch.float32) * k_scale)
                .to(torch.bfloat16)
                .unsqueeze(1)
            )
        ckv = gathered[..., :v_head_dim].contiguous()
        kpe = gathered[..., v_head_dim:].contiguous()

        qo_indptr = torch.arange(0, num_tokens_q + 1, device=device, dtype=torch.int32)
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
        return wrapper.run(q_nope, q_rope, ckv, kpe, return_lse=False)

    def _forward_fa3(
'''


@patch_dsa_backend.run
def apply_b(p: Patch) -> None:
    # ONE group guard up front for all three B-edits, exactly as the original did
    # (`if markerB_init in srcB: ... else: apply all three`). This is NOT cosmetic:
    # p33 (the cuda-graph plan/run-split) later REWRITES the very text B2/B3 inject,
    # so their own `new` text is not a durable already-applied probe. Relying on
    # per-edit probes made the second run of the runner re-apply B2/B3 and corrupt
    # dsa_backend.py -- caught by the idempotency check on 2026-07-16, invisible to
    # a single-run tree-diff. MARKER_B_INIT survives p33 untouched, which is exactly
    # why the original keyed the whole group on it.
    if MARKER_B_INIT in p.code:
        return
    p.replace(OLD_INIT, NEW_INIT, marker=MARKER_B_INIT, what="B1-init")
    p.replace(OLD_DISPATCH, NEW_DISPATCH, what="B2-dispatch")
    p.replace(OLD_METHOD_ANCHOR, NEW_METHOD, what="B3-method")
