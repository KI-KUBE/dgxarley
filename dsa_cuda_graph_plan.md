# DSA `flashinfer_gather` CUDA-Graph Plan/Run Split — Implementation Plan

Plan only, no implementation. Written 2026-07-16 against image
`xomoxcc/dgx-spark-sglang:0.5.15-sm121`. Companion to `dsalogitrework.md`
(the gather+dense-fa2 attention decode fallback this document fixes the
CUDA-graph story for) and `DSA_speedup.md`.

## 1. The problem (live crash, 2026-07-16)

`_forward_flashinfer_gather` (`sglang_launch.sh` patch block
`PATCH_DSA_FLASHINFER_GATHER`, method added to `DeepseekSparseAttnBackend` in
`dsa_backend.py`) calls `wrapper.plan(...)` (flashinfer
`BatchMLAPagedAttentionWrapper.plan`) **inline**, every forward call. This
crashes during decode CUDA-graph capture:

```
decode_cuda_graph_runner.py:821 run_once
  -> _forward_flashinfer_gather (dsa_backend.py:2344)
  -> wrapper.plan(...)
  -> flashinfer/mla/_core.py:1648 plan
Scheduler hit an exception ...
Hint: 2. set --cuda-graph-max-bs-decode to a smaller value
      3. disable decode CUDA graph by --cuda-graph-backend-decode=disabled
```

`plan()` does host-side dynamic work (stream sync / allocation) that is not
CUDA-graph-recordable. Every other SM121 hardware wall for DSA is already
cleared (indexer via the torch fallback, no `trtllm-gen` FMHA assert, the
576-byte KV dequant fix) — this is the last implementation gap, not a
hardware dead end. Current workaround: `disable_cuda_graph: true` in the
profile, which proves functional correctness (GSM8K) at eager (slow) speed
while this fix is pending.

## 2. Correction to the assumed hook interface

The task that spawned this plan assumed the classic (deprecated)
`init_forward_metadata_capture_cuda_graph` / `init_forward_metadata_replay_cuda_graph`
override pair. **This SGLang version has migrated away from those** — they
are removed from the ABC. The current contract
(`sglang/srt/layers/attention/base_attn_backend.py:18-99`, `AttentionBackend`
docstring, verbatim):

- `init_forward_metadata(fb)` — eager entry point; default wraps
  `_out_graph(fb)` + `_in_graph(fb)`.
- `init_forward_metadata_out_graph(fb, in_capture=False)` — per-iteration
  metadata prep, runs **outside** `with graph.capture():`. Called at capture
  time (`in_capture=True`, once per shape, before `run_once`/the graph
  region) **and** at every replay (`in_capture=False`, before
  `graph.replay()`). Host ops / dynamic-shape / `.item()` / stream-sync logic
  belongs here.
- `init_forward_metadata_in_graph(fb)` — graph-recordable, static-shape GPU
  ops only, runs inside `with graph.capture():` and auto-replays. Lint
  contract: **must not** call `.item()`/`.cpu()`/`.tolist()`/dynamic-shape
  `torch.empty()`.

`DeepseekSparseAttnBackend` (our target class) already implements both
`init_forward_metadata_out_graph` (`dsa_backend.py:697`, dispatches to
`self._apply_cuda_graph_metadata(...)`) and `init_cuda_graph_state`
(`dsa_backend.py:1085`) for its existing decode impls (trtllm, flashmla,
tilelang, aiter). We need to extend these, not invent a parallel mechanism.

## 3. How the native `FlashInferMLAAttnBackend` gets this right

Read in full from `flashinfer_mla_backend.py`. The mechanism, precisely:

**`forward_decode` (line 595) never calls `.plan()`.** It only calls
`decode_wrapper.run(q_nope, q_rope, k_buffer_nope, k_buffer_rope, out=o, ...)`
— `run()` is the only wrapper method invoked inside the graph region.

**`.plan()` is called exclusively from `FlashInferMLAIndicesUpdaterDecode
.call_begin_forward`** (line ~727), itself only reachable via
`indices_updater_decode.update(...)`, itself only called from
`init_forward_metadata_out_graph` / `_apply_cuda_graph_metadata` — i.e.
always outside the graph, both at capture-prep and at every replay-prep:

- **Capture** (`init_forward_metadata_out_graph(fb, in_capture=True)`,
  line 308): builds a fresh `BatchMLAPagedAttentionWrapper(use_cuda_graph=True,
  qo_indptr=self.cuda_graph_qo_indptr[...], kv_indptr=self.cuda_graph_kv_indptr[...],
  kv_indices=self.cuda_graph_kv_indices, kv_len_arr=self.cuda_graph_kv_lens[...])`
  — **static, pre-allocated buffers**, sized once in `init_cuda_graph_state`
  (line 426) for `max_bs`. Calls `indices_updater_decode.update(...,
  init_metadata_replay=False)` → the **real** `.plan()` runs once, which
  populates `wrapper._cached_module` (the resolved/dispatched kernel handle).
  Only **after** that call completes:
  `decode_wrapper.plan = partial(fast_mla_decode_plan, decode_wrapper)`
  — the wrapper's `.plan` attribute is monkey-patched to a fast variant for
  every subsequent call on this wrapper instance.
- **Replay** (`in_capture=False`): calls `_apply_cuda_graph_metadata(...)`
  (line 457), which recomputes `kv_len_arr_cpu` fresh from
  `seq_lens_cpu[:bs]` (host, but this call is *outside* the graph, so a host
  op here is fine), writes it and the cumsum-derived `kv_indptr_cpu` **in
  place** into the same buffers allocated in `init_cuda_graph_state`
  (`self.cuda_graph_kv_indptr_cpu[1:bs+1] = torch.cumsum(...)`), then calls
  `indices_updater_decode.update(..., decode_wrapper=self.decode_cuda_graph_metadata[bs],
  init_metadata_replay=True, **fast_decode_kwargs)` → this reaches
  `wrapper.plan(fast_decode_kwargs["qo_indptr_cpu"], fast_decode_kwargs["kv_indptr_cpu"],
  kv_indices, fast_decode_kwargs["kv_len_arr_cpu"], ...)` — but `wrapper.plan`
  is now `fast_mla_decode_plan`, a **module-level, backend-agnostic function**
  (`flashinfer_mla_backend.py:1085`) that skips the real `.plan()`'s stream
  sync and just calls `self._cached_module.plan(...)` directly with the
  already-resolved kernel handle. This runs on every decode step (every
  replay), still outside the graph, cheaply.

**Why `.plan()` must be re-invoked every step at all:** `kv_len_arr` (real
per-request context length) genuinely changes every decode step (context
grows by 1 token each time) even for the native dense backend, so the
schedule really is recomputed every replay — just cheaply, via the fast path.

**`init_cuda_graph_state`** (line 426) is where the static buffers live:
`cuda_graph_kv_indices` (`[max_bs * max_context_len]`), `cuda_graph_qo_indptr`
/ `cuda_graph_kv_indptr` (cloned from the base `q_indptr_decode`/`kv_indptr`),
`cuda_graph_kv_lens`, plus CPU mirrors (`cuda_graph_qo_indptr_cpu`,
`cuda_graph_kv_indptr_cpu`) bundled into `self.fast_decode_kwargs`.

`DSAMetadata.page_table_1` (the pre-indexer, per-token page table, **not**
the post-topk-selection gather indices) already follows exactly this
static-buffer + in-place-`.copy_()` pattern generically for every existing
DSA decode impl (`dsa_backend.py:_apply_cuda_graph_metadata`,
`_build_forward_metadata_cuda_graph`) — this machinery is inherited "for
free"; the topk-selection / indexer output feeding into our
`_forward_flashinfer_gather`'s `page_table_1` argument is the thing this
plan is scoped to (see Section 6 open risk: unverified whether the indexer
itself is graph-safe on this stack, since no DSA decode impl has ever
reached graph capture successfully on SM121 before this backend).

## 4. The concrete fix for `_forward_flashinfer_gather`

**Core change: move `wrapper.plan(...)` entirely out of
`_forward_flashinfer_gather`.** The method should only build `ckv`/`kpe`
(gather + dequant, as today) and call `wrapper.run(q_nope, q_rope, ckv, kpe,
...)`, reading a wrapper that was already `.plan()`-ed by the out-of-graph
hook for this batch size.

Concretely, in the `sglang_launch.sh` `PATCH_DSA_FLASHINFER_GATHER` block
(extend, keep the existing marker/idempotency pattern):

1. **State (extend `__init__`, B1):** replace the single
   `self._flashinfer_gather_wrapper = None` slot with a per-bs cache dict
   (`self._flashinfer_gather_wrappers: dict[int, BatchMLAPagedAttentionWrapper] = {}`),
   mirroring `decode_cuda_graph_metadata`. Also add static buffers sized for
   `max_bs * topk` in a new `init_cuda_graph_state` extension (new patch
   anchor into `DeepseekSparseAttnBackend.init_cuda_graph_state`,
   `dsa_backend.py:1085`): `self._fig_kv_indptr_cpu`, `self._fig_qo_indptr`
   (both are pure `arange`-derived from `bs`/`topk`, no dynamic content —
   see Section 5), and `self._fig_kv_len_arr_cpu` (a pinned/CPU buffer,
   `[max_bs]`, the one genuinely-dynamic quantity).

2. **New `init_forward_metadata_out_graph` branch:** `DeepseekSparseAttnBackend
   .init_forward_metadata_out_graph` (`dsa_backend.py:697`) already calls
   `self._apply_cuda_graph_metadata(...)` for every decode impl. Add a
   post-step there (or a sibling private method called right after) gated on
   `self.dsa_decode_impl == "flashinfer_gather"` and
   `forward_mode.is_decode_or_idle()`:
   - Compute `kv_len_arr_cpu = metadata.dsa_cache_seqlens_int32[:bs].clamp(max=topk).cpu()`
     (or, if a CPU mirror of `dsa_cache_seqlens_int32` already exists on
     `metadata` from the shared DSA cuda-graph metadata build, reuse it
     instead of a fresh `.cpu()` sync — check `DSAMetadata` fields before
     assuming a new sync is needed).
   - Write it into `self._fig_kv_len_arr_cpu[:bs]` in place.
   - On first use for this `bs` (capture, `bs not in self._flashinfer_gather_wrappers`):
     construct `BatchMLAPagedAttentionWrapper(self.workspace_buffer,
     use_cuda_graph=True, qo_indptr=self._fig_qo_indptr[:bs+1],
     kv_indptr=self._fig_kv_indptr_cpu-derived-GPU-indptr[:bs+1],
     kv_indices=<static index buffer, see Section 5>, kv_len_arr=self._fig_kv_len_arr_cpu[:bs])`,
     call the **real** `.plan(...)` once (via the still-unpatched method),
     then monkey-patch: `wrapper.plan = partial(fast_mla_decode_plan, wrapper)`
     (import `fast_mla_decode_plan` from
     `sglang.srt.layers.attention.flashinfer_mla_backend` — it is a plain
     module-level function, generic to any `BatchMLAPagedAttentionWrapper`,
     **not** MLA-backend-specific; reuse it rather than reimplementing).
     Store in `self._flashinfer_gather_wrappers[bs]`.
   - On every call (capture-prep **and** every replay-prep): call
     `wrapper.plan(qo_indptr_cpu, kv_indptr_cpu, kv_indices, kv_len_arr_cpu, ...)`
     — now the fast variant — with the freshly-updated `kv_len_arr_cpu`.
     This mirrors `_apply_cuda_graph_metadata`'s per-replay `wrapper.plan()`
     call for the native backend exactly.

3. **`_forward_flashinfer_gather` becomes graph-body-only:** drop the
   `wrapper.plan(...)` call entirely; look up
   `wrapper = self._flashinfer_gather_wrappers[bs]` (bs from
   `q_nope.shape[0]` or passed through `metadata`), keep the gather + dequant
   (unchanged, produces `ckv`/`kpe`), call `wrapper.run(q_nope, q_rope, ckv,
   kpe, return_lse=False)`. This is the only wrapper method now invoked
   inside `run_once`/the graph region.

4. **Eager (non-graph) path must keep working identically:** when
   `disable_cuda_graph: true` (today's functional-test config) or for shapes
   that never get captured, `init_forward_metadata` (the eager entry, not
   `_out_graph`) still needs a `.plan()` to have run before `forward_decode`.
   Cleanest: have the plan step live in a small shared helper called from
   both `init_forward_metadata_out_graph` (graph path) and a
   `dsa_decode_impl == "flashinfer_gather"` branch added to
   `DeepseekSparseAttnBackend.init_forward_metadata`'s eager body (the
   override at `dsa_backend.py:716`) — do **not** duplicate the plan logic;
   factor it into one private method (e.g. `self._plan_flashinfer_gather(bs,
   metadata, use_fast=in_capture_or_replay)`) called from both entry points.

## 5. `kv_indices`/`qo_indptr`/`kv_indptr` for the gathered buffer: mostly static, one dynamic piece

Unlike the native backend (whose `kv_indices` point into the persistent,
never-reallocated KV cache and are genuinely per-request/per-step dynamic),
our post-gather addressing is **almost entirely compile-time-static** given
a fixed captured batch size `bs` and fixed `topk` (2048, a config constant):

- `qo_indptr = arange(0, bs+1)` — static, depends only on `bs`.
- `kv_indptr = qo_indptr * topk` — static, depends only on `bs`/`topk`.
- `kv_indices = arange(0, bs*topk)` — static: post-gather, the dequantized
  buffer is already dense/sequential per request (see the existing
  `_forward_flashinfer_gather` docstring: "page_size=1 post-gather: the
  freshly gathered/dequantized buffer is already dense per request").

**The one genuinely dynamic quantity is `kv_len_arr`**, because a request's
*real* context length can be less than `topk` (early decode steps, or short
prompts) — the DSA indexer's `topk(min(topk, end_pos))` returns fewer than
2048 valid indices in that regime, and `metadata.dsa_cache_seqlens_int32`
(already clamped to `topk` in the existing eager code) legitimately grows by
1 each decode step until it saturates at `topk`, exactly analogous to the
native backend's real `kv_len_arr`. **This means the "plan once per shape,
reuse forever" simplification is NOT valid** — `wrapper.plan()` must be
re-invoked (via the fast path) on every replay with the current
`kv_len_arr_cpu`, matching Section 4's design, not a cheaper "static-plan"
shortcut. (An initial hypothesis before this line of investigation was that
`flashinfer_gather`'s indices might be fully static and could skip the
per-step `fast_mla_decode_plan` call entirely — this section is where that
hypothesis is falsified; do not re-attempt that shortcut without first
proving `kv_len_arr` is truly step-invariant for this workload, which it is
not once short sequences are considered.)

## 6. Open risks

- **Indexer graph-safety, unverified.** No DSA decode backend has ever
  reached CUDA-graph capture successfully on this SM121 stack before
  `flashinfer_gather` (trtllm/tilelang/flashmla all died on kernel asserts
  pre-capture). The generic `page_table_1`/`_apply_cuda_graph_metadata`
  machinery exists in the source and is presumably exercised on other
  hardware, but we have zero live confirmation it works end-to-end here.
  If the indexer's own forward (torch fallback,
  `dsa_paged_mqa_logits_backend=torch`) does anything graph-unsafe
  (`.item()`, dynamic shapes) that was never hit before because capture
  never got this far, it becomes a **new** blocker discovered only once this
  fix is deployed. Test the indexer under `disable_cuda_graph=false`
  specifically, isolated if possible.
- **`kv_len_arr` correctness for the padded-tail (short-sequence) case.**
  Flagged as an open point in `dsalogitrework.md` already — the numeric
  verification there only proved plumbing (no crash/NaN), not correctness,
  for `page_table_1 < topk`. This plan's `fast_mla_decode_plan` reuse
  inherits that same unproven area; verify explicitly once graph mode is
  back on.
- **Gather+dequant buffer identity across replay (lower-confidence risk).**
  The `ckv`/`kpe` gather (`flat_kv_cache[page_table_1.reshape(-1).long()]`
  or `dequantize_k_cache_paged(...)`) still runs *inside* `run_once`/the
  graph region every call, materializing a fresh output tensor each time via
  fancy indexing (not a static-buffer in-place write). PyTorch's CUDA-graph
  private memory pool generally handles repeated same-shape allocations
  inside a captured region correctly (same pool address reused across
  captures/replays as long as the allocation pattern is identical every
  call) — this is *not* the same class of problem as `.plan()`'s host sync,
  and is not expected to be the source of the original crash (the traceback
  pointed at `wrapper.plan`, not the gather). Flag as a secondary risk to
  watch during the Step 8 GPU micro-test, not a required structural change
  up front. If it does misbehave, the fix is `torch.index_select(...,
  out=<pre-allocated static buffer>)` instead of fancy indexing.
- **`fast_mla_decode_plan` compatibility with our config unverified.** It is
  generic (any `BatchMLAPagedAttentionWrapper`), but has only ever been
  exercised with the native backend's `page_size=1, causal=False` (see
  `call_begin_forward`'s real-`.plan()` call args) vs. our
  `page_size=1, causal=True` (per the current `_forward_flashinfer_gather`
  docstring/code, `wrapper.plan(..., 1, True, ...)`). Confirm
  `fast_mla_decode_plan`'s internal `_cached_module.plan(...)` call accepts
  and correctly threads a `causal=True` config before relying on it.
- **`next_n >= 2` (MTP multi-token draft verify) is out of scope**, as in
  `dsalogitrework.md` — this plan only covers plain decode
  (`forward_mode.is_decode_or_idle()`), matching the existing
  `_forward_flashinfer_gather` scope.
- **Mixed batches** (some requests real-context < topk, others saturated)
  inside one captured `bs` shape: `kv_len_arr_cpu` naturally handles
  per-request variation (it is a `[bs]` vector already), so this should fall
  out of the design for free — but has not been tested.

## 7. Verification / test plan

1. Static: `bash -n` on the extended `sglang_launch.sh`; the extracted
   heredoc applied against the live image (`Patched`/`already applied`
   idempotency, per the established pattern); `py_compile` on the mutated
   files.
2. **GPU micro-test before touching the real patch further:** in an isolated
   debug pod, construct a tiny synthetic decode CUDA graph (small `bs`,
   `topk`) that exercises exactly the new split — real `.plan()` once,
   monkey-patch to `fast_mla_decode_plan`, `graph.capture()` a body that only
   calls `wrapper.run(...)`, then `graph.replay()` after mutating
   `kv_len_arr_cpu`/re-running the fast plan with a *different* real KV
   content — and numerically compare against the eager (non-graph) path for
   the same inputs. This isolates the CUDA-graph mechanics from the rest of
   the 25-minute model boot and would have caught the original crash in
   seconds instead of a full deploy cycle.
3. Live deploy: flip `disable_cuda_graph: false` back on in the profile,
   redeploy, confirm capture completes without the `wrapper.plan` traceback,
   confirm `Capturing batches` reaches 100%, head goes Ready.
4. Correctness: re-run the GSM8K smoke/eval already used for the eager
   (`disable_cuda_graph: true`) functional test, compare graph-mode output
   token-for-token (or at least accuracy-for-accuracy) against the eager
   run's captured baseline — a silent graph-replay staleness bug (stale
   buffer contents from a previous batch size/shape) would not crash, it
   would just produce wrong tokens.
5. Throughput: compare `gen throughput` graph-on vs. graph-off (today's
   eager baseline) and vs. the dense `flashinfer` baseline, to quantify the
   actual win before investing further (e.g. the warp-MMA path from
   `DSA_speedup.md`).
