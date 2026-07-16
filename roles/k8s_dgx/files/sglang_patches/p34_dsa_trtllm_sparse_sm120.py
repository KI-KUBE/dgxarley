"""[dgxarley] DSA on GB10/SM121: route the "trtllm" impl to flashinfer's NATIVE
SM120/121 sparse-MLA kernels (decode AND prefill) instead of trtllm-gen.

THE FINDING (2026-07-16, image xomoxcc/dgx-spark-sglang:0.5.15-sm121): flashinfer
0.6.14 ships `flashinfer/mla/_sparse_mla_sm120.py`, a native sparse-MLA paged
attention with `@supported_compute_capability([120, 121])`, PREBUILT in this image
(no JIT wall), with an explicit GLM_NSA model type (d_qk=576, arbitrary-fp32 inline
scales), `(16, 2048)` in the decode dispatch set (= our TP4 head count + index_topk),
native `-1`-padding skip, dedicated warp-spec decode kernels for num_tokens<=64 and
a prefill orchestrator above that. Its dispatcher,
`flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`, routes cc==12 +
sparse_mla_top_k>0 to it automatically -- but ONLY with `backend="auto"`. SGLang's
`_forward_trtllm` hardcodes `backend="trtllm-gen"`, which on SM121 dies in
`TllmGenFmhaRunner ... Unsupported architecture` (the "trtllm wall" that forced the
whole flashinfer_gather workaround, p31-p33). Upstream sglang main still hardcodes
it as of 2026-07-16 -> this patch is an upstream-PR candidate.

GPU-VERIFIED on spark5 (podman, GB10, sglang's REAL production quantize_k_cache
building the 656-byte packed pool, vs a pure-torch dequant+softmax reference):
  decode bs=4 topk=2048:        max|diff| 0.008, 0.072 ms/call (gather impl: 0.26 ms)
  decode bs=32:                 0.236 ms/call
  decode seq_lens>topk:         PASS (kernel clamps; sglang passes UNCLIPPED
                                cache_seqlens for decode -- verified safe)
  prefill 2400 extend tokens:   max|diff| 0.016, 14.4 ms/layer-call -- the exact
                                GSM8K concurrency-8 shape that OOM-killed the
                                p32 gather prefill (11.3 GB fp32 intermediate)
  prefill 8192 (chunk cap):     48.5 ms/layer-call
  cuda-graph capture+replay:    WORKS directly (0.239 ms/replay, finite) -- the
                                p33 plan/run-split is NOT needed on this path
Full trail: dsalogitrework.md, dsa_cuda_graph_plan.md, DSA_speedup.md.

FIRST LIVE DEPLOY LESSON (2026-07-16, all 4 pods restarted once at decode graph
capture): "ValueError: SM120 sparse MLA v32/GLM expects BF16 query, got
torch.float8_e4m3fn". Cause: for dsa+trtllm+fp8, `_fuse_rope_for_trtllm_mla`
(forward_mla.py) skips rope upstream and `_forward_trtllm` fuse-ropes AND
fp8-QUANTIZES the query for trtllm-gen. The sparse kernel wants a bf16 query.
Fix = the forward_mla edit (rope stays upstream on SM12x) + skipping the fp8
query-quantize branch for `_sparse_sm120` -- the exact flow the live-proven
flashinfer_gather path used (and set_mla_kv_buffer's packed store needs bf16
k/k_rope input anyway; its quantize_k_cache asserts bf16).

LIVE-PROVEN 2026-07-16 (second deploy, with the fix): boot clean (0 restarts, 0
ANCHOR-DRIFT), decode graph capture 18 s (the previous crash point), smoke
coherent, decode 8.4 tok/s cuda-graph (indexer-bound, as predicted), and the
gather-prefill killer shape survived live: a 7-seq/2240-token prefill batch ran
at 873 tok/s input (was 1-5 tok/s + OOM crash).

THREE EDITS make the native path reachable:

1. model_runner_kv_cache_mixin.py::calculate_mla_kv_cache_dim early-returns the
   PLAIN 576 layout whenever a dsa backend is "trtllm" (correct for SM100
   trtllm-gen, which dequants via a scalar bmm1 k_scale). The SM120 sparse kernel
   instead consumes the 656-byte packed layout (512 fp8 nope + 4x fp32 tile scales
   + 64 bf16 rope) -- the SAME layout sglang's quantize_k_cache already writes
   (and which the flashinfer_gather deploys live-tested). So: skip that early
   return on SM12x. `dsa_kv_cache_store_fp8` then flips True automatically
   (memory_pool derives it from override_kv_cache_dim), which also keeps the
   p31/p33 gather fallback's dequant branch consistent.

2. dsa_backend.py::_forward_trtllm: on SM12x with the packed pool, call the
   flashinfer dispatcher with backend="auto" (-> "sparse"), a uint8 view of the
   KV buffer (the sm120 checker requires torch.uint8), and
   kv_scale_format="arbitrary_fp32": sglang's quantize_k_cache writes amax/448
   fp32 tile scales (NOT pow2/ue8m0), which is exactly flashinfer's GLM_NSA scale
   semantics; the default "auto" would misread them as DSv3.2 pow2 scales.
   skip_softmax is not supported by the sparse backend -> force None there.
   Plus: skip the fused rope+fp8-query-quantize branch for `_sparse_sm120`
   (see FIRST LIVE DEPLOY LESSON above).
   SM100/SM103 behaviour is byte-identical (all conditionals keep the upstream
   values on the else side).

3. deepseek_common/attention_forward_methods/forward_mla.py::
   `_fuse_rope_for_trtllm_mla`: return False for the dsa branch on SM12x, so
   rope is applied normally in forward_absorb_prepare and the query reaches
   `_forward_trtllm` in bf16.

ACTIVATION: inert unless attention_backend=dsa AND dsa_prefill_backend/
dsa_decode_backend resolve to "trtllm" AND the device is SM12x with fp8 KV cache
(the packed pool). The indexer side is unchanged and still needs
dsa_paged_mqa_logits_backend=torch on SM121 (p30). MTP target-verify/draft-extend
use the decode impl and go through the same kernel (q reshaped per token).

RE-SYNC on image bump: if upstream lands SM120 routing in _forward_trtllm (watch
for backend="auto" or a cc==12 branch near the trtllm_batch_decode_with_kv_cache_mla
call) and in calculate_mla_kv_cache_dim, DELETE this patch.
"""

from _patchlib import Patch

MARKER = "# [patch] _sgl_dsa_trtllm_sparse_sm120_"

patch_mixin = Patch(
    name="MLA KV dim: keep the 656-byte packed layout for trtllm on SM12x",
    target="sglang/srt/model_executor/model_runner_kv_cache_mixin.py",
)


@patch_mixin.run
def apply_mixin(p: Patch) -> None:
    p.replace(
        """        if (
            self.server_args.dsa_prefill_backend == "trtllm"
            or self.server_args.dsa_decode_backend == "trtllm"
        ):
            return kv_cache_dim""",
        """        if (
            self.server_args.dsa_prefill_backend == "trtllm"
            or self.server_args.dsa_decode_backend == "trtllm"
        ):
            """
        + MARKER
        + """
            # On SM120/SM121 the trtllm impl dispatches to flashinfer's packed
            # sparse-MLA backend (backend="auto" in _forward_trtllm, this patch),
            # which consumes the 656-byte inline-scale layout -- do NOT early-
            # return to the plain 576 layout there. Datacenter Blackwell (SM100)
            # keeps the early return and the plain layout for trtllm-gen.
            if not (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] == 12
            ):
                return kv_cache_dim""",
        marker=MARKER,
        what="trtllm early-return SM12x bypass",
    )


patch_dsa_backend = Patch(
    name="_forward_trtllm: backend=auto + GLM_NSA scale format on SM12x",
    target="sglang/srt/layers/attention/dsa_backend.py",
)


@patch_dsa_backend.run
def apply_dsa_backend(p: Patch) -> None:
    # Edit 0: derive the SM12x-sparse condition ONCE at the top of the function.
    # `dsa_kv_cache_store_fp8` is the precise gate: True exactly when the pool
    # holds the 656-byte packed layout the sm120 kernel consumes (with the mixin
    # edit above, that is SM12x + fp8 KV + trtllm backends). A bf16-KV or plain-
    # layout config keeps the upstream call unchanged.
    p.replace(
        """        \"\"\"Forward using TRT-LLM sparse MLA kernel.\"\"\"
        import flashinfer.decode

        metadata = self.forward_metadata
""",
        """        \"\"\"Forward using TRT-LLM sparse MLA kernel.\"\"\"
        import flashinfer.decode

        metadata = self.forward_metadata
        """
        + MARKER
        + """
        # Native SM120/121 sparse-MLA route (decode <=64 tokens: warp-spec
        # kernels; more: prefill orchestrator). GPU-verified vs torch reference;
        # captures directly under cuda-graph. See the patch docstring.
        _sparse_sm120 = self.device_sm_major == 12 and self.dsa_kv_cache_store_fp8
""",
        what="_sparse_sm120 gate at function top",
    )

    # Edit 1: the fp8 branch fuse-ropes AND QUANTIZES the query to fp8 for the
    # trtllm-gen kernel -- the sm120 sparse kernel instead requires a BF16 query
    # (live crash at decode graph capture: "SM120 sparse MLA v32/GLM expects
    # BF16 query, got torch.float8_e4m3fn"). Skip the branch for _sparse_sm120:
    # rope then runs normally upstream (the forward_mla.py edit below un-skips
    # it), q stays bf16 via the merge_query concat, and k/k_rope reach
    # set_mla_kv_buffer in bf16, which the packed store's quantize_k_cache
    # requires anyway (it asserts bf16 input). This mirrors exactly how the
    # live-proven flashinfer_gather path flows.
    p.replace(
        """        merge_query = q_rope is not None
        if self.kv_cache_dtype == torch.float8_e4m3fn:
""",
        """        merge_query = q_rope is not None
        if self.kv_cache_dtype == torch.float8_e4m3fn and not _sparse_sm120:
""",
        what="skip fused rope+fp8-quantize of q on SM12x",
    )

    # Edit 2: uint8 view of the packed pool (the sm120 checker requires
    # torch.uint8; the pool's store dtype is an fp8 view of the same bytes).
    p.replace(
        """        out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv,
""",
        """        out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv.view(torch.uint8) if _sparse_sm120 else kv,
""",
        what="uint8 KV view",
    )

    # Edit 3: backend/scale-format/skip-softmax selection. arbitrary_fp32 =
    # flashinfer's GLM_NSA semantics, matching sglang's quantize_k_cache scales
    # (amax/448 fp32, not pow2). skip_softmax raises on the sparse backend.
    p.replace(
        """            sparse_mla_top_k=self.dsa_index_topk,
            bmm1_scale=bmm1_scale,
            backend="trtllm-gen",
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
""",
        """            sparse_mla_top_k=self.dsa_index_topk,
            bmm1_scale=bmm1_scale,
            backend="auto" if _sparse_sm120 else "trtllm-gen",
            kv_scale_format="arbitrary_fp32" if _sparse_sm120 else "auto",
            skip_softmax_threshold_scale_factor=(
                None
                if _sparse_sm120
                else envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get()
            ),
""",
        what="backend=auto + kv_scale_format on SM12x",
    )


patch_forward_mla = Patch(
    name="_fuse_rope_for_trtllm_mla: no fused rope+fp8-q on SM12x (sparse wants bf16 q)",
    target="sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py",
)


@patch_forward_mla.run
def apply_forward_mla(p: Patch) -> None:
    # The dsa branch of _fuse_rope_for_trtllm_mla makes the model SKIP rope in
    # forward_absorb_prepare and defer it into _forward_trtllm's fused
    # rope+fp8-quantize (mla_quantize_and_rope_for_fp8) -- which hands the sm120
    # sparse kernel an fp8 query it rejects. On SM12x return False so rope runs
    # normally upstream and q reaches _forward_trtllm in bf16 (the same flow the
    # flashinfer_gather path used, live-proven with the packed pool). SM100
    # keeps the fused path. The second call site (extra cos_sin_cache args for
    # the attention call) is gated by the same function, so it stays consistent.
    p.replace(
        """        if self.current_attention_backend in ("dsa", "nsa"):
            return (
                get_global_server_args().dsa_decode_backend == "trtllm"
                or get_global_server_args().dsa_prefill_backend == "trtllm"
            ) and get_attn_backend().kv_cache_dtype == torch.float8_e4m3fn
""",
        """        if self.current_attention_backend in ("dsa", "nsa"):
            """
        + MARKER
        + """
            # SM12x routes trtllm to the native sparse kernel (BF16 query,
            # inline-scale packed KV): keep rope in forward_absorb_prepare.
            if torch.cuda.get_device_capability()[0] == 12:
                return False
            return (
                get_global_server_args().dsa_decode_backend == "trtllm"
                or get_global_server_args().dsa_prefill_backend == "trtllm"
            ) and get_attn_backend().kv_cache_dtype == torch.float8_e4m3fn
""",
        marker=MARKER,
        what="dsa fuse-rope off on SM12x",
    )
