# QUANT_HY3_GOTCHAS.md — Tencent Hy3 (hy_v3) NVFP4 on GB10/sm121

Root-cause notebook for getting **Tencent Hy3** (arch `hy_v3` / `HYV3ForCausalLM`, 295B
total / 21B active MoE) to serve **coherently** on the 4-node DGX Spark cluster
(GB10 / **sm121** / consumer Blackwell) via **SGLang**, image
`xomoxcc/dgx-spark-sglang:0.5.14-sm121`.

TL;DR (2026-07-10, **ROOT CAUSE FOUND**): the W4A4 NaN is **NOT** the FP4 kernels, the
flashinfer stack, the JIT cache, or the HYV3 model code — all exonerated. Per-layer tensor
tracing pinned the first inf/nan to **`model.layers.1.mlp.shared_mlp.down_proj`** — the
**shared-expert MLP**. Its `gate_up_proj` already outputs all-ZERO → `down_proj` FP4-quantizes
a zero input → scale `448·6/0` degenerates → NaN. **Root cause: a WEIGHT-NAME MISMATCH** — the
checkpoint names the shared expert `model.layers.N.mlp.**shared_experts**.*` (present, FP4), but
SGLang's HYV3 model module is `**shared_mlp**`, and `load_weights` has no remap for it → the
shared-expert weights are silently skipped → `shared_mlp` stays zero-init → NaN. **The routed
experts (cutlass_fused_moe) are FINITE — they work.** **Fix (VERIFIED WORKING 2026-07-10 —
CHEAP, no re-quant, no graft): a loader monkey-patch that remaps `.shared_experts.` →
`.shared_mlp.` in HYV3 `load_weights`** — wired into `sglang_launch.sh` (gated on Hy3/Hunyuan,
idempotent). After it, vroomfondel's FULL W4A4 Hy3 serves COHERENTLY on GB10/sm121 (correct
German, correct 86.67 km/h math, valid fib() code — no NaN). The whole "W4A4 NaNs on GB10" saga
was ONE weight-name gap. Full methodology + trace in the "ROOT CAUSE (layer tracing)" section
below. The W4A16 detour remains a separate dead end in SGLang (Blocker A).

Related: `TURBOQUANT.md` (kernel-crash matrix + per-model rationale), per-model
`TESTLOGS/`. The NVFP4 checkpoints referenced here were produced with a local
ModelOpt-based quantizer pipeline.

---

## ROOT CAUSE (layer tracing, 2026-07-10) — how we found it, so it's reproducible

This is the debug method that finally localised the W4A4 NaN. It is reusable for ANY
"model NaNs/garbles and we don't know which layer" case on this cluster.

### 0. Why the earlier isolation was a dead end (and the two false-negatives)

Before tracing, a long chain of isolated GPU-pod probes all came back CLEAN, which was
misleading. Recorded here so we don't repeat the mistakes:
- **Isolated the WRONG kernel.** `moe_runner_backend=cutlass` is REMOVED in 0.5.14
  (`moe_runner/` has no `cutlass.py` → NotImplementedError). The real path for
  `moe_runner_backend=flashinfer_cutlass` + modelopt_fp4 is
  `flashinfer.fused_moe.cutlass_fused_moe` (modelopt_quant.py:108/1138). My "kernels clean"
  probes used `sglang.jit_kernel.nvfp4` / SGLang's own `cutlass_moe_fp4` — a dead path.
- **The #2708 PDL/GDC race needs CONCURRENCY** — a single isolated `mm_fp4` call can't
  trigger it, so isolation can't reproduce it (it IS fixed in our flashinfer 0.6.14: GDC
  flags in `jit/gemm/core.py` AND `jit/fused_moe.py::gen_cutlass_fused_moe_sm120_module`,
  bundled CUTLASS C++ 4.5.0 ≥ the 4.4 the fix needs).
- **`.float()` on the global scale dodged #3497** (bf16 global scale corruption).
- Lesson: **when isolation keeps coming back clean, stop isolating and trace the real model.**

### 1. What we ruled out first (all confirmed present/fine in the running stack)

- Stale JIT cache — cleared `/var/lib/hf-cache/flashinfer_cache` on all 4 sparks; the fresh
  serve JIT-compiled the sm120 cutlass FP4 GEMM + grouped-MoE `.cu` fresh (verified under
  `.cache/flashinfer/0.6.14/121a/generated/…gemm_grouped/120/`). Cache is version-namespaced
  anyway (old April kernels are in dead 0.6.5/0.6.6 namespaces).
- flashinfer 0.6.14 has #2708 (GDC + CUTLASS 4.5) and #3497; #3592 is off our direct-API path.
- `fp4_quantize` saturates the E4M3 block scale at byte 126 (never 0x7F=127 NaN).
- dtype correct (checkpoint declares `dtype: bfloat16` under the transformers-5 key; SGLang
  resolves `--dtype auto` → bf16).
- HYV3 model code (`srt/models/hunyuan_v3.py`) read end-to-end — standard, no obvious inf gen.

Decisive control: a clean re-serve on the fully-patched, fresh-compiled stack **still NaN'd on
the first forward** (`sampler.py sampling_from_probs_torch → torch.multinomial` →
`probability tensor contains inf/nan`). So it's a real residual, not any known bug → trace it.

### 2. The layer trace (the tool that cracked it) — reproducible recipe

We wired SGLang's built-in per-layer tensor dump through the launch (default OFF):
- knobs `sglang_debug_tensor_dump_output_folder` / `_layers` in `defaults/main/sglang.yml`
- env `SGLANG_DEBUG_TENSOR_DUMP_*` in `tasks/sglang_instance.yml` ConfigMap
- gated `--debug-tensor-dump-output-folder` / `--debug-tensor-dump-layers` in `sglang_launch.sh`

Run it:
```bash
# 1) deploy with the dump enabled (folder must be a MOUNTED path; hf-cache is JuiceFS-shared
#    so all TP/PP ranks dump into ONE readable folder)
ansible-playbook k8s_dgx.yml --tags sglang \
  -e sglang_debug_tensor_dump_output_folder=/root/.cache/huggingface/tensor_dump
# 2) it dumps during startup warmup/autotune already — one Pass00000.pt per rank is written
#    BEFORE it dies. (NOTE: the dump hook then crashes on a SECONDARY SGLang bug —
#    add_tensor → logits_processor `next_token_logits=None` during warmup → AttributeError.
#    That is NOT our NaN; Pass00000 is complete and already contains the inf.)
# 3) the dumps land on the JuiceFS host mount:
#    /mnt/jfs/tensor_dump/TP{r}_PP{p}_Rank{n}_pid{}/Pass00000.pt   (~205 MB/rank)
```
Analyse (spin a CPU debug pod that mounts `hostPath /mnt/jfs`, run `hy3_dump_find_first_inf.py`
in the repo root, or inline): each `Pass*.pt` is `torch.load` → **ordered** dict
`{op_name: output_tensor}` in forward-execution order. The **first key with isinf/isnan is the
culprit module**; the key before it was still finite.

### 3. The result

All 4 TP ranks agreed (Pass00000, op index 102):
```
[95]  layers.1.self_attn.attn               finite (absmax 0.047)
[96]  layers.1.self_attn.o_proj             finite (0.23)
[98]  layers.1.mlp.gate (router)            finite
[100] layers.1.mlp.shared_mlp.gate_up_proj  absmax = 0     ← output already ZERO
[101] layers.1.mlp.shared_mlp.act_fn        absmax = 0
[102] layers.1.mlp.shared_mlp.down_proj     NaN            ← FIRST inf/nan
[103] layers.1.mlp.experts (routed)         finite (0.066) ← routed experts WORK
[90]  layers.0.mlp.down_proj (dense)        finite         ← plain FP4 linear is fine
```
Mechanism: the shared-expert `gate_up_proj` outputs all-zero → `act_fn`=0 → `down_proj`
FP4-quantizes a zero input → `amax=0` → global scale `448·6/0` degenerates → NaN → propagates
to layer 2 (`qkv_proj`, `q_norm` NaN).

### 3b. WHY gate_up_proj is zero — a weight-NAME mismatch (the actual root cause)

The shared-expert weights are NOT degenerate — they simply **never load**. Comparing the two
checkpoints' `model.safetensors.index.json`:
- **vroomfondel/Hy3-NVFP4-W4A4** stores the shared expert as
  `model.layers.N.mlp.**shared_experts**.{gate,up,down}_proj.{weight,weight_scale,weight_scale_2,input_scale}`
  (plural `shared_experts`, 948 tensors, FP4 — present and real). `mlp.shared_mlp.*` exists only
  for layer 80 (the MTP/NEXTN layer, BF16).
- **SGLang's HYV3 model** (`srt/models/hunyuan_v3.py`) names the module **`shared_mlp`**
  (`self.shared_mlp = HYV3FeedForward(...)`, prefix `.shared_mlp`). `load_weights` only remaps
  `router.gate` → `gate` — there is NO `shared_experts` → `shared_mlp` remap.
- ⇒ every `mlp.shared_experts.*` weight fails `if name not in params_dict: continue` and is
  **silently skipped** → `shared_mlp` stays zero-initialised → zero output → NaN. (The routed
  experts `mlp.experts.N.*` match SGLang's names and load fine — hence they work.)

### 4. The fix (cheap — loader remap, NO re-quant, NO graft)

A one-line name remap in HYV3 `load_weights`, mirroring the existing `router.gate` remap:
`name = name.replace(".shared_experts.", ".shared_mlp.")` at the TOP of the weight loop (so the
existing `gate_proj`/`up_proj` → `gate_up_proj` stacking then applies). The shared-expert weights
are real FP4, identical quant to the routed experts (which work), so once loaded they should run
finite. **Wired into `sglang_launch.sh`** as a gated (Hy3/Hunyuan), idempotent monkey-patch of
`hunyuan_v3.py` (step 3 of the hunyuan patch block). No checkpoint edit, no calibration.
(The user's graft idea — inject BF16 shared_mlp from the original/kodelow + add to `ignore` —
would also work but is unnecessary: the FP4 weights are present, only misnamed. Likely
upstream-fixable too; consider reporting the `shared_experts`→`shared_mlp` gap to sgl-project.)

**VERIFIED WORKING (2026-07-10):** after the patch, vroomfondel's full W4A4 Hy3 serves coherently
— head reaches Ready, first forward no longer NaNs, and the coherence check returns correct
German (Rayleigh scattering), correct math (390 km / 4.5 h = 86.67 km/h), and a valid iterative
`fib()`. So the shared-expert FP4 weights run finite once loaded, exactly as predicted.

**Patch-guard gotcha (cost one deploy cycle):** the original `hunyuan_v3.py` already contains the
substring `shared_experts` via the config attr `num_shared_experts` (`getattr(config,
"num_shared_experts", 0)`). An idempotency guard of `if "shared_experts" in code` therefore
false-positives on a fresh file and SKIPS the patch (log: "already present, skipping") → still
NaN. Guard on the exact remap string (`'replace(".shared_experts.", ".shared_mlp.")' in code`)
instead. General lesson for these sed/python source patches: never guard on a bare identifier
that can appear as a substring elsewhere in the file.

---

## Model / arch facts (hy_v3)

- `HYV3ForCausalLM`, 80 layers, 64 attn heads / 8 KV heads (head_dim 128), hidden 4096.
- MoE: **192 routed experts** (top-8) + 1 shared, `moe_intermediate 1536`, `rms_norm_eps 1e-5`.
- `qk_norm`, `route_norm`, **1 MTP/NEXTN layer** (layer 80), native context **262144**.
- Tokenizer carries a **`:opensource` token suffix** — parser fix backported (below).
- `trust_remote_code: true` (custom arch); parsers `tool_call_parser: hunyuan` /
  `reasoning_parser: hunyuan`.

---

## Status matrix (both NVFP4 builds)

| Build                          | Quant                                                       | Loads? | Result                  | Root cause                                                                                                            |
|--------------------------------|-------------------------------------------------------------|--------|-------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **vroomfondel/Hy3-NVFP4-W4A4** | W4A4 (weight+**activation** FP4, has `input_scale`)         | ✅      | **garbage / NaN**       | backend-independent; activation-quantize + single FP4 GEMM both EXONERATED on GB10 (see Blocker B) — cause still open |
| **kodelow/Hy3-NVFP4-W4A16**    | W4A16 (weight-only NVFP4, `tensor_group`, no `input_scale`) | ❌      | **crash at model load** | SGLang has **no weight-only-NVFP4-MoE scheme** — confirmed empirically 0/9, see below                                 |

Both profiles live in `roles/k8s_dgx/model_profiles/`
(`vroomfondel-hy3-nvfp4-w4a4.yml`, `kodelow-hy3-nvfp4-w4a16.yml`); both are listed in
the model-family selector `group_vars/all/main/sglang.yml` (selector-parity rule).
The kodelow profile now carries a `⛔ CONFIRMED DEAD-END` header + a
`⟳ RECHECK(sglang>0.5.14)` marker recording exactly when to re-test it (see Blocker A).

---

## Blocker A — kodelow W4A16 (weight-only NVFP4): SGLang has no scheme for it

**Symptom:** head crash-loops at load (restarts climb immediately, exit before any
forward). `kubectl logs --previous`:

```
[TP0] Acceleration for non-quantized schemes is not supported by Compressed Tensors.
      Falling back to UnquantizedLinearMethod
[TP0] Scheduler hit an exception: Traceback (most recent call last):
    self.model = HYV3Model(config, quant_config, prefix=...)
  File .../compressed_tensors/compressed_tensors.py, line 712, in get_moe_scheme
    elif self._is_dynamic_token_w8a8(weight_quant, input_quant):
  File .../compressed_tensors/compressed_tensors.py, line 389, in _is_dynamic_token_w8a8
    is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
AttributeError: 'NoneType' object has no attribute 'num_bits'
```

**Empirically confirmed (2026-07-10, full matrix run):** `sglang_nn4_tp4_ep1`
matrix on `kodelow/Hy3-NVFP4-W4A16`, image `0.5.14-sm121`: **0/9 passed.** All 9
cases (36 log lines = 9 cases × 4 ranks) die with the **byte-identical** crash above,
and **zero** cases reach `KV Cache is allocated` / any CUDA-graph capture — i.e. the
crash is 100% pre-backend, pre-forward, at scheme-resolution time during model load.
The matrix sweep (moe-runner / fp4-gemm / attention-backend / kv-cache-dtype
permutations) is therefore **meaningless** for this build: every cell hits the exact
same crash regardless of any backend knob, because the crash happens *before* any
backend is ever selected. Do not re-run this matrix without first fixing/working
around the underlying scheme gap.

**Why (the real root cause, not the surface `AttributeError`):**
`CompressedTensorsMoEMethod.get_moe_scheme()` walks its scheme predicates in order.
For kodelow's routed experts **none matches**, so it falls through to the w8a8 check,
which dereferences `input_quant.num_bits` on a weight-only checkpoint (`input_quant is
None`) → the crash. The crash is just the fall-through; the actual gap is that **no
predicate accepts a weight-only NVFP4 MoE**:

- **`_is_wNa16_group_channel`** (the marlin / triton weight-only path we *wanted*):
  ```python
  return is_channel_group and input_quant_none and is_static
  #      is_channel_group = weight_quant.strategy in {CHANNEL, GROUP}
  ```
  kodelow's NVFP4 weights use strategy **`tensor_group`** (group-of-16 + per-tensor
  global scale), which is **neither** `CHANNEL` nor `GROUP` → **False**. (This predicate
  is for **integer** WNA16 — int4-group marlin — not float NVFP4.)
- **`_is_fp4a4_nvfp4`** (the only NVFP4 branch): first line is
  `if weight_quant is None or input_quant is None: return False`. Weight-only has no
  `input_quant` → **False**. (This branch is **W4A4** — it *requires* an activation scale.)
- → falls through to `_is_dynamic_token_w8a8` → `None.num_bits` → **crash**.

**SGLang's compressed-tensors MoE schemes** (`.../compressed_tensors/schemes/` +
`get_moe_scheme` branches) are exactly:
- `CompressedTensorsW4A4Nvfp4MoE` — **W4A4** NVFP4 (needs activation scale)
- `CompressedTensorsWNA16{MoE,MarlinMoE,TritonMoE}` — **integer** W4A16/W8A16 group
- `CompressedTensorsW8A8{Fp8,Int8}MoE`, `W4A8Int8`, `MxInt4`, plus NPU-only variants

There is **no `CompressedTensorsW4A16Nvfp4MoE`** (float4 weight-only). kodelow's
"NVFP4 W4A16" is precisely the zwitter between the two: float4 weights (so the integer
WNA16-marlin path rejects it) with no activation scale (so the W4A4-NVFP4 path rejects
it).

**Verdict:** dead end, **independent of GB10/sm121**, and now empirically confirmed
(0/9, identical crash, zero backend-dependence). A runtime monkey-patch (our usual
`sglang_launch.sh` pattern) would **not** help here — null-guarding
`_is_dynamic_token_w8a8` only turns the crash into "unsupported scheme"; making it work
would require *writing* a weight-only-NVFP4 MoE kernel, which doesn't exist in SGLang.
The card's "serves on marlin" is a **vLLM** claim, not portable to SGLang.
If you specifically wanted weight-only, you'd need an **integer** W4A16 (int4-group)
export, not an NVFP4 one — but that abandons FP4 entirely, which defeats the purpose.
The `kodelow-hy3-nvfp4-w4a16.yml` profile header carries a
`⟳ RECHECK(sglang>0.5.14)` marker: re-test only once upstream SGLang ships a
weight-only-NVFP4 MoE scheme (watch `compressed_tensors/schemes/` for a new
`W4A16Nvfp4MoE`-shaped class, or a `get_moe_scheme` predicate that accepts
`strategy=tensor_group` with `input_quant=None`).

---

## Blocker B — vroomfondel W4A4 (weight+activation FP4): NaN, cause narrowed but still open

This is the build we actually want — **4-bit activations** exercise the GB10 FP4 tensor
cores. It **loads and runs**, but emits **garbage / NaN**:

- triton runner → single-char repetition (`!!!!`)
- flashinfer runner → NaN device-side assert / crash
- marlin runner → token salad

i.e. **backend-independent** → not a single-kernel bug in the *runner selection* sense.
Originally this pointed at the shared **activation-quantize FP4** path on sm121 as prime
suspect. That suspicion has since been **tested directly and exonerated** (see the probe
log below) — the real locus is narrower than first thought.

### Matrix run confirmation (2026-07-10)

Full `sglang_nn4_tp4_ep1` matrix on `vroomfondel/Hy3-NVFP4-W4A4`, image `0.5.14-sm121`:
**0/N passed.** Backend-availability taxonomy on sm121, read directly from head logs
(reusable fact set, independent of the NaN investigation):

- moe-runner **`cutlass`** → `NotImplementedError: Unsupported runner backend: CUTLASS`
- fp4-gemm **`trtllm`** / **`cute-dsl`** → `flashinfer.utils.BackendSupportedError:
  mm_fp4 does not support backend '...' with capability 121`
- moe-runner **`flashinfer_trtllm`** → loads, autotunes, then crashes at runtime in
  `trtllm_batched_gemm_runner.cu:305: Error occurred when running GEMM!` — the kernel is
  compiled for `sm100f` (Blackwell-DC), not `sm121` (consumer Blackwell) — a hard
  architecture mismatch, not a numerics bug.
- Only combo that reaches a **forward pass** at all: moe-runner `flashinfer_cutlass` /
  `triton` + fp4-gemm `flashinfer_cutlass`. These are the cases that produce
  `probability tensor contains inf/nan` (sampler device-assert) — i.e. the NaN is real
  compute garbage, not a load-time crash like Blocker A.

**Ruled out** (systematic isolation this session — see `vroomfondel-hy3-nvfp4-w4a4.yml`
header + `TESTLOGS/`):
- **Image regression** — `nvidia/Qwen3-235B-A22B-NVFP4` (also NVFP4) serves **coherent**
  ("Paris", reasoning_content produced) on the exact same `0.5.14-sm121` image.
- **W4A4-per-se** — Qwen NVFP4 above is also W4A4 and works. Separately: the user
  independently quantized + served `Qwen3-30B-A3B` at full W4A4 (NVFP4, ModelOpt) on a
  single GB10 Spark via a local ModelOpt quantizer — coherent output, no NaN. This proves
  W4A4-on-GB10-via-SGLang works *in general* for a plain full-attention MoE arch.
- **EP dispatch** — TP=4/**EP=1** still NaN (not the 192-expert all-to-all).
- **Config / parsing** — hunyuan parser + token-suffix backport verified good.
- **`admissible_archs` sm_121a patch** — confirmed *applied* in the new image (the
  `mma.py` sed in `sglang_launch.sh` lands, logged as `Patched cute/mma.py: added
  sm_120a + sm_121a` in every head log). So the MoE **weight**-GEMM arch gate is fine.
- **Checkpoint calibration mechanics** — calibrated cleanly on H200 (pre/post metrics
  sane) in a local ModelOpt quantizer (`qformat: nvfp4` = full W4A4). NOTE: this only proves
  the *calibration pipeline ran without error* — it does **not** prove the resulting
  checkpoint serves coherently via SGLang on *any* hardware. That question was never
  actually tested and is back in scope below.

**Blocked / untested:**
- **PP=4 / TP=1** — torch-distributed init deadlocks on this cluster (TCPStore broken
  pipe); the pure-PP (no TP-shard) hypothesis is untestable until PP=4 infra is fixed.

### Activation-FP4 dispatch — located and isolated (2026-07-10)

The activation-FP4 quantize dispatch does **not** go through the `mma.py`
`admissible_archs` patch at all — that patch only affects the CuTeDSL/cute-dsl code
path, which turns out to be **unused** on GB10. The actual dispatch:

`sglang/srt/layers/quantization/fp4_utils.py:26`
```python
_flashinfer_fp4_quantize_backend = "cute-dsl" if is_sm100_supported() else "cuda"
```

`is_sm100_supported` / `is_sm120_supported` (`sglang/srt/utils/common.py` ~277-286):
```python
is_sm120_supported = lru_cache(...)(partial(_check_cuda_device_version, device_capability_majors=[12], cuda_version=(12, 8)))
is_sm100_supported = lru_cache(...)(partial(_check_cuda_device_version, device_capability_majors=[10], cuda_version=(12, 8)))
```

GB10 is `sm_121` → CUDA capability `(12, 1)` → **major = 12** →
**`is_sm100_supported() = False`**, **`is_sm120_supported() = True`** →
activation-quantize backend on GB10 = **`"cuda"`**, not `"cute-dsl"`. This is a
genuinely separate dispatch from the MoE-weight-GEMM `admissible_archs` gate the
`mma.py` patch fixes.

### Isolated GB10 probe (debug pod `sglang-fp4-probe`, GPU, node spark4, image
`xomoxcc/dgx-spark-sglang:0.5.14-sm121`, 2026-07-10)

Live arch-gate readout on the actual hardware:
```
device_capability      = (12, 1)
is_cuda                = True
is_sm100_supported     = False
is_sm120_supported     = True
>>> ACTIVATION fp4_quantize backend = cuda
```

**1) Activation FP4 quantize, standalone — clean.** Quantized a random BF16 activation
tensor through `flashinfer.fp4_quantize`:
```
[cuda    ] xq.shape=(64, 256) dtype=uint8  sf: nan=False allzero=False min=0 max=448  xq allzero=False
[cute-dsl] EXC ModuleNotFoundError: No module named 'cutlass._mlir._mlir_libs._cutlass_ir._cute'
```
`cute-dsl` isn't even importable in this image — but it's moot, GB10 never selects it
(see dispatch above). The `cuda` backend (the one actually used) produces sane,
non-NaN, non-degenerate scale factors and packed FP4 bytes.

**2) Single CUTLASS FP4 GEMM, standalone — correct.** Used SGLang's own sm121
sgl-kernel wrapper (`sglang.jit_kernel.nvfp4.cutlass_scaled_fp4_mm`) against a BF16
matmul reference, 4 shapes chosen to resemble Hy3's MoE GEMM dimensions:

| shape (M×K×N) | tag | nan | inf | absmax(out) | absmax(ref) | mean_rel_err | corr |
|---|---|---|---|---|---|---|---|
| 128×2048×1536 | hy3 MoE-w1 | False | False | 11.125 | 11.149 | 0.135 | 0.9909 |
| 128×1536×4096 | hy3 MoE-w2 | False | False | 10.000 | 10.173 | 0.134 | 0.9910 |
| 64×4096×4096  | square 4k  | False | False | 15.125 | 15.181 | 0.134 | 0.9910 |
| 256×2048×2048 | bigger M   | False | False | 11.312 | 11.301 | 0.134 | 0.9910 |

No NaN/Inf in any shape, output magnitude tracks the BF16 reference, and
`mean_rel_err ≈ 13.5%` / `corr ≈ 0.99` is the expected lossy-but-correct signature of
4-bit quantization — **not** garbage. **The single FP4 GEMM computes correctly on
sm121.**

**3) `flashinfer.mm_fp4` backend gating on capability 121 — confirms the matrix
taxonomy independently:** `backend='trtllm'` and `backend='cute-dsl'` both raise
`BackendSupportedError: mm_fp4 does not support backend '...' with capability 121`;
only `cutlass` (and `cudnn`) are accepted. Matches Blocker B's matrix results exactly.

**Repro gotcha:** `flashinfer.fp4_quantize` returns scale factors as raw `uint8`;
`cutlass_scaled_fp4_mm` requires them typed as `float8_e4m3fn` — `.view(torch.float8_e4m3fn)`
before passing in, or it raises `RuntimeError: scale_a must be float8_e4m3fn`.

### Updated verdict

Both prior prime suspects — the **activation-FP4 quantize step** and the **single FP4
GEMM kernel** — are **exonerated** on GB10 in isolation: neither produces NaN/garbage,
and both compute numerically sane, correctly-scaled output. The backend-independent
NaN seen in the real model is therefore **not** in either of these primitives alone.

**Fused MoE probe (DONE — also clean).** `sglang.srt.layers.moe.cutlass_moe.cutlass_moe_fp4`
run on GB10 with a synthetic 8-expert / topk-2 MoE (M=16, K=512 hidden, N=256
intermediate), FP4-quantized per-expert w1/w2 + per-expert global/block scales, against a
BF16 reference MoE (silu-and-mul SwiGLU + weighted top-k combine):

```
out: nan=False inf=False   absmax(out)=2.469  ref=2.461   mean_rel_err=0.235  corr=0.9719
```

No NaN/Inf; output tracks the BF16 reference (corr 0.97). The higher rel-err vs the single
GEMM (0.235 vs 0.135) is expected — two chained FP4 GEMMs plus a re-quantize of the SwiGLU
intermediate compound the FP4 rounding. So the fused path — grouped GEMM via pointer
arrays, the fused SwiGLU, AND the per-expert global/block scale combination — computes
**correctly** on sm121. (Reused `cutlass_moe_fp4`'s own internal `prepare_moe_input` +
`scaled_fp4_experts_quant`, so the offset/permutation plumbing is the real one, not a
hand-rolled approximation.)

→ **All three FP4-specific components are now exonerated on GB10.** The backend-independent
NaN is not in the FP4 kernels at all. The single remaining suspect:

1. **SGLang's HYV3 model-impl.** The H200 "calibrated cleanly" check (the quantizer's
   before/after `generate()` in `hf_ptq`) is a **transformers + modelopt fake-quant**
   preview on the quant box — NOT SGLang, NOT the real FP4 kernels, and it runs
   **HF-transformers' HYV3 `trust_remote_code`**, whereas SGLang serves with its **own**
   HYV3 forward. So a coherent H200 preview only proves the quant MATH is sane in the HF
   framework; it says nothing about SGLang's HYV3 port. Combined with the kernel
   exoneration above, that makes SGLang's HYV3 impl (attention/qk_norm/router/shared
   expert/MTP) or the export→reload param mapping a live suspect. Upstream is actively
   patching this arch: PR #30331 (load HunyuanV3 NextN final_layernorm into the draft
   head's output norm), PR #30594 (default Hunyuan V3 to bfloat16 when the checkpoint has
   no dtype).
2. **A broader SGLang-NVFP4-on-consumer-Blackwell (SM120/121) correctness bug —
   NOT Hy3-specific.** Upstream issue **#18954** reports the SAME "NaN in logits, garbage
   output" for `Qwen3.5-397B-A17B-NVFP4` (a NON-HYV3 model) on **RTX PRO 6000 (SM120)**
   with `flashinfer_cutlass` — now known **CLOSED**, root-caused to flashinfer #2708
   (PDL/GDC race, fixed CUTLASS 4.4); see the CORRECTION block below. This partly reopens the shared
   FP4 path despite our isolated-kernel exoneration: our probes used random well-behaved
   inputs; the real failure may need real weight/activation distributions, CUDA-graph
   capture, or a shared code path the isolated test skipped. Note the split: Qwen3-235B-
   A22B-NVFP4 serves *coherently* on the same sm121 image (Blocker B), but Qwen3.5-397B and
   Hy3 (both larger/hybrid-ish) NaN — so "which NVFP4 models NaN on SM12x" is itself a clue.

**Prior art (web search 2026-07-10) — others hit this, and the working path is known:**
- **Hy3-295B-NVFP4 DOES serve coherently on GB10** — `tonyd2wild/Hy3-295B-NVFP4-MTP-2x-DGX-Spark`
  runs it on **2× DGX Spark**, "stable, verified twice", ~21.8 tok/s. BUT via **vLLM 0.23.1
  + NVFP4 W4A16 (weight-only, MARLIN kernel) + FP8-KV + MTP(1) + `--enforce-eager`**, TP=2/Ray.
  i.e. the SAME weight-only-marlin path SGLang lacks (Blocker A), on vLLM. Confirms
  model+checkpoint+hardware can produce coherent Hy3-NVFP4 — the gap is purely SGLang's
  serving path (no weight-only-NVFP4 MoE scheme; W4A4 NaNs).
- **vLLM #40252**: Qwen3-Next NVFP4 *silently produces garbage* when linear-attn weights are
  missing from `quantization_config.ignore` — the hybrid-arch "wrong layer got quantized"
  pattern. Worth checking vroomfondel/Hy3-W4A4's `quantization_config.ignore` completeness.
- The realistic conclusion: the **proven** Hy3-on-GB10 recipe today is **weight-only NVFP4 +
  marlin** (vLLM has it, SGLang doesn't). SGLang-W4A4's NaN is the #2708/#18954 PDL/GDC race —
  **fixed upstream and present in our flashinfer 0.6.14**, but its effectiveness on SM121 is
  unconfirmed (see CORRECTION block below; matrix still NaN'd on this image). The vLLM-marlin
  path collides with this repo's hard rules (no vLLM; want 4-bit activations) — flagged for a
  decision, not silently resolved.

**HYV3 model-impl inspection (DONE 2026-07-10 — clean).** Read SGLang's
`srt/models/hunyuan_v3.py` + `hunyuan_v3_nextn.py` end-to-end in the debug pod. The forward
is standard and correct: attention = qkv_proj → optional qk_norm (RMSNorm on head_dim) →
RoPE → RadixAttention → o_proj; decoder = fused input_layernorm / post_attention_layernorm
add-norm; MoE = **gate in fp32 (unquantized)**, sigmoid scoring + grouped-topk +
correction bias, shared-expert added, EP/TP all-reduce; `load_weights` uses the standard
qkv/gate_up stacking + `FusedMoE.make_expert_params_mapping` for expert weight/scale
mapping. No obvious inf/nan generator, no suspicious dtype cast, router never quantized.
One nuance: `_forward_dual_stream` (shared-expert on main stream, routed on alt stream)
runs only under cuda-graph capture — but the eager matrix cases NaN too, so it's not the
sole cause. **→ The HYV3 model code is exonerated too.**

### CORRECTION (2026-07-10, later) — it's a KNOWN, FIXED flashinfer bug cluster, not "open/unpatchable"

My earlier "open, unfixed upstream bug, not patchable" verdict was WRONG. The root cause is
identified and fixed upstream:
- **sglang #18954 is CLOSED**, root-caused to **flashinfer #2708** (CLOSED): CUTLASS FP4 GEMM
  JIT was **missing `-DCUTLASS_ENABLE_GDC_FOR_SM100=1`** compile flags → the PDL
  `wait_on_dependent_grids()` barriers compiled to **no-ops** → **NaN/garbage in 128-aligned
  tiles UNDER CONCURRENCY** on SM120/consumer-Blackwell. Fixed via CUTLASS 4.4 (flashinfer
  PR #2913). **This is exactly why my isolated single-kernel probes were CLEAN — the race
  needs concurrent overlapping dependent kernels (PDL overlap); one isolated call can't
  trigger it.** (False-negative #1.)
- **flashinfer PR #3497 (MERGED 2026-06-03):** `nvfp4_quantize(backend='cuda')` silently
  corrupts scale factors when `global_scale` is **not float32** (a bf16 `(448·6)/amax` gets
  misread byte-wise). **My activation-quantize probe passed a `.float()` global scale → dodged
  this.** (False-negative #2.)
- **flashinfer PR #3592 (still OPEN):** calibrated NVFP4 global scales in the *unified MoE API*
  (cute-dsl/trtllm) were hardcoded to 1.0 → magnitude inflation (#3548). Likely N/A to our
  SM121 cutlass path, but noted.

**BUT — our image ALREADY has the fixes** (verified in-container 2026-07-10): flashinfer
**0.6.14** (2026-07-02, > #3497 merge) + `-DCUTLASS_ENABLE_GDC_FOR_SM100=1` present in
`jit/gemm/core.py`. So bumping flashinfer is NOT the fix. Yet the matrix run — which used THIS
image — still NaN'd. So the open question is now sharp: **is the #2708 GDC fix actually
effective on SM121 (the flag is `_SM100`; GB10 is treated as sm120-class), or does a residual
remain (e.g. an AOT/JIT-cached kernel compiled without the flag, or the still-open #3592 MoE
scale path)?** The decisive test is a **concurrency reproduction on GB10 with this image's
flashinfer** — fire many overlapping `mm_fp4`/cutlass-MoE calls with PDL enabled + a bf16
global scale, and watch for the 128-aligned-block NaN that the single-shot probe cannot show.
Until that's run, the verdict is: **known bug, fix present in our flashinfer, effectiveness on
SM121 unconfirmed — reproduce under concurrency before concluding.**

### PATH CORRECTION (2026-07-10) — SGLang's own cutlass MoE is REMOVED; my isolation tested a dead path

Verified in-container: `moe_runner/` has NO `cutlass.py` — `moe_runner_backend=cutlass` →
`NotImplementedError` (runner.py:67). The FP4 MoE for a modelopt_fp4 model with
`moe_runner_backend=flashinfer_cutlass` goes through **`flashinfer.fused_moe.cutlass_fused_moe`**
(modelopt_quant.py:108/1138), NOT SGLang's `srt/layers/moe/cutlass_moe.py::cutlass_moe_fp4`.
Consequences:
- **My earlier "FP4 kernels clean" probes used `sglang.jit_kernel.nvfp4` + SGLang's
  `cutlass_moe_fp4` — the REMOVED path.** So the single-GEMM + fused-MoE exoneration tested the
  wrong code. The ACTUAL runtime kernel (`flashinfer.fused_moe.cutlass_fused_moe`) is still
  untested by isolation. Re-do the concurrency probe against THAT.
- **The `sglang-gemma4-geglu-nan-clamp.patch`** (`# SM120 FP4 bug: clamp E4M3 block scale
  uint8=127→126 to prevent NaN`, in `cutlass_moe.py::cutlass_moe_fp4`) sits on that **dead
  path** AND is **not applied in the 0.5.14-sm121 recipe** anyway. But it documents a REAL,
  distinct SM120 FP4 failure mode: the E4M3 block-scale can hit `0x7F` = the E4M3 **NaN
  encoding** → NaN. flashinfer's fp4 path has **no such clamp** (`fp4_quantization.py` only has
  UE8M0 `2^(byte-127)` math). **This is the most concrete forward-fix candidate: check whether
  `flashinfer.fused_moe.cutlass_fused_moe` on GB10 emits a 127 block-scale, and if so add the
  clamp to the flashinfer path (or find flashinfer's own fix).**
- **flashinfer PR#3592** fixes the UNIFIED MoE API's `prepare_trtllm_fp4_weights` /
  `prepare_cute_dsl_nvfp4_weights` global-scale hardcoding. SGLang calls the DIRECT
  `cutlass_fused_moe` (explicit `quant_scales`), which does not appear to route through those
  prepare functions → **#3592 is likely OFF our SM121 cutlass path; pulling it in probably
  won't fix Hy3** (verify by reading `core.cutlass_fused_moe` before investing).
- **Do our patches make it worse?** No. `arch-prune` keeps `sm_121a` and only strips dead
  gencode targets; `skip-sm90-target` disables a redundant sm90 build GB10 never loads; neither
  touches GDC/PDL/FP4-correctness, and both act on sgl-kernel, not flashinfer.

### RESOLUTION CANDIDATE (2026-07-10) — our image HAS every known fix; prime suspect is the STALE JIT CACHE

Verified in-container (flashinfer 0.6.14): the COMPLETE #2708 fix is present — bundled
**CUTLASS C++ 4.5.0** (≥ the 4.4 the fix requires) + `grid_dependency_control.h` +
`-DCUTLASS_ENABLE_GDC_FOR_SM100=1` passed by BOTH `jit/gemm/core.py` AND
`jit/fused_moe.py::gen_cutlass_fused_moe_sm120_module` (the actual MoE path). #3497 fixed.
Live block-scale probe: `fp4_quantize` **saturates the E4M3 block scale at byte 126 (=448),
NEVER emits 0x7F=127 (the NaN encoding)** even with 64× outlier blocks → the gemma4-clamp
failure mode does NOT reproduce on 0.6.14. #3592 is off-path. **So there is no known-unfixed
FP4 kernel bug left in the image** — yet the matrix (on this image) still NaN'd.

**Prime remaining suspect: the persistent, node-local flashinfer JIT cache.**
`sglang_instance.yml` sets `FLASHINFER_WORKSPACE_BASE=/root/.cache/flashinfer_persistent`,
mounted from the **hostPath `{{ hf_cache_path }}/flashinfer_cache`** (per-spark, node-local,
survives pod restarts AND image upgrades). If the cutlass fused-MoE `.so` was JIT-compiled on
an EARLIER flashinfer (pre-GDC-flag / pre-4.5 / pre-mma.py-patch) and cached, the sparks LOAD
that stale `.so` — the source has the fix but the running kernel does not. Same risk for
`triton_cache` / `fa4_cute_dsl_cache` (also hostPath).

**Decisive cheap test (needs approval — cluster write + re-serve):** on all 4 sparks clear
`{{ hf_cache_path }}/flashinfer_cache` (and `triton_cache`), then re-serve Hy3-W4A4 → forces a
fresh JIT compile against the 0.6.14 GDC-flagged source. If the NaN clears, the whole saga was
a stale cached kernel. If it persists, we're into a genuinely novel residual and the next step
is a real end-to-end serve with per-layer activation dumping.

**UPDATE (2026-07-10) — mtime check REFUTES the simple stale-cache theory; cache cleared anyway.**
Read-only inspection of `/var/lib/hf-cache/flashinfer_cache` on all 4 sparks:
- The cache is **version-namespaced** (`.cache/flashinfer/<version>/…`) with dirs for
  0.6.5…0.6.14 — including a **0.6.14** dir. So 0.6.14 would NOT load an older version's `.so`.
- The fp4/cutlass/**fused_moe** `.so` exist **only under 0.6.5 + 0.6.6 (all dated 2026-04-04)** —
  none under 0.6.10–0.6.14. And **zero** flashinfer `.so` newer than 2026-06-01, even though
  `triton_cache` has 2026-07-10 entries (so the recent matrix DID recompile triton, but NOT
  flashinfer fp4/fused_moe). → the April kernels sit in dead namespaces; 0.6.14 doesn't load
  them. **Stale-cache-shadows-the-fix is refuted for the FP4 MoE path.**
- Two live readings remain: (1) on 0.6.14 the cutlass fused-MoE kernel is likely **AOT-bundled
  in the wheel** (fixed) and never touches this JIT cache → "all fixes present, still NaN" is
  real; or (2) the matrix actually ran on an **older image** that reused the April kernels.
  Only a controlled re-serve on the confirmed-0.6.14 image separates these.
- **Action taken:** cleared `flashinfer_cache` on all 4 sparks (275M→4K each, hostname-guarded,
  no serving pod running) as hygiene — NOT expected to fix the NaN (theory refuted), but gives a
  clean slate for the decisive re-serve. `triton_cache` left intact (current, 2026-07-10).

**So the single decisive step now is the controlled re-serve** (needs approval): serve Hy3-W4A4
on the confirmed-0.6.14 image, clean caches, and observe — NaN-clears ⇒ old-image artifact;
NaN-persists ⇒ novel residual ⇒ per-layer activation dump to localise the first inf.

**Two cheap SGLang-side controls — BOTH RUN 2026-07-10:**
- **dtype → NEGATIVE (not the bug).** vroomfondel-W4A4 `config.json` DOES specify bf16 — under
  the transformers-5 key `dtype: "bfloat16"` (the old `torch_dtype` key is null, a red herring).
  SGLang's `_get_and_verify_dtype` resolves `--dtype auto` → `torch.bfloat16` correctly (verified
  live in the pod). So the model loads as bf16, PR #30594 does not apply, and forcing
  `--dtype bfloat16` would change nothing.
- **ignore-list → the W4A4 build is AGGRESSIVE.** vroomfondel-W4A4 is `quant_method: modelopt`,
  `group_0: targets=['Linear'], w(num_bits=4, type=float), act=4` (i.e. **W4A4, 4-bit
  activations**), and its `ignore` (82 entries) is ONLY `lm_head` + `embed_tokens` + all 80
  per-layer `mlp.gate` routers. It does **NOT** exclude `self_attn` (q/k/v/o), `shared_mlp`, or
  `eh_proj` (MTP) — so **attention, shared experts, and the MTP head all get 4-bit ACTIVATIONS**.
  The safe/working recipes keep exactly these in higher precision: kodelow-W4A16's ignore (14
  entries) explicitly excludes self_attn + shared_mlp + eh_proj (experts-only NVFP4), and
  tonyd2wild's working vLLM build is weight-only W4A16. So the W4A4 build applies 4-bit activation
  quant to the numerically-fragile attention/MTP layers that every working recipe avoids.
  **BUT** this alone doesn't fully explain #18954 — Qwen3.5-397B-NVFP4 is a mixed-precision
  (attention NOT 4-bit-act) modelopt build and NaNs on SM120 too. So aggressive-W4A4 is a
  plausible *aggravating* factor, not proven to be the sole root cause; the open upstream
  consumer-Blackwell NVFP4 bug remains the dominant explanation.

**Strategic fork (needs a user decision — collides with the Hard rules):** the ONLY proven
Hy3-on-GB10 recipe today is `tonyd2wild`'s **vLLM + NVFP4-W4A16-weight-only + marlin + FP8-KV
+ MTP + --enforce-eager** on 2× DGX Spark (coherent, verified) — i.e. the weight-only-marlin
path SGLang lacks (Blocker A), on the vLLM stack the Hard rules exclude. SGLang-W4A4 is
blocked on #18954. So: (a) relax "no vLLM" and ship W4A16-marlin via vLLM, (b) relax "4-bit
activations" and use integer W4A16 / FP8-W8A8 on SGLang, or (c) stay on SGLang-W4A4 and wait
for / push the upstream #18954 fix. Not resolvable by patching — flagged for the call.

**Open next steps (not yet done — need approval, no autodeploy):**
1. ~~Finish the runner sweep~~ — done, see matrix confirmation above; only
   `flashinfer_cutlass`/`triton` reach forward, both NaN.
2. ~~Trace the activation-FP4 dispatch~~ — done, see above; exonerated.
3. ~~Isolate the fused MoE path~~ — done, clean (see probe log above). All three FP4
   components exonerated.
4. **PIVOT to checkpoint/model-impl (now the leading and only remaining suspect):**
   (a) has `vroomfondel/Hy3-NVFP4-W4A4` served coherently via SGLang on *any* hardware? if
   not, sm121-specificity is unconfirmed and the checkpoint leads by elimination;
   (b) scan the checkpoint's scale tensors (`*weight_scale*`, `*input_scale*`,
   `*weight_scale_2*`) for degenerate values (0 / inf / nan / absurd magnitude) vs a
   known-good NVFP4 export (Qwen3-30B-A3B / Qwen3-235B-A22B);
   (c) if scales look sane, load the real model in the debug pod and dump per-layer
   activation stats to find WHERE the first inf/nan appears (attention out? router?
   shared expert? first MoE layer? MTP?) — that localizes model-impl vs data.
5. Fix PP=4 infra (TCPStore) to unblock the TP-shard-free control (still blocked,
   unrelated to the above).

---

## W4A8 — does SGLang have an NVFP4-weight + FP8-activation MoE path? No.

Checked directly against the CUDA MoE scheme inventory in the `0.5.14-sm121` image
(`sglang/srt/layers/quantization/compressed_tensors/schemes/` +
`sglang/srt/layers/quantization/modelslim/`). There is **no NVFP4-W4A8** (4-bit FP4
weight + FP8 activation) fused-MoE kernel anywhere in SGLang. What exists:

| Scheme | Weight | Activation | CUDA-usable on GB10? |
|---|---|---|---|
| `CompressedTensorsW4A4Nvfp4MoE` | FP4 | FP4 | yes (this is what we're debugging) |
| `CompressedTensorsW4A4MxInt4MoE` | INT4 (mx) | — | **no** — hard-asserts `flashinfer_trtllm` backend, which is dead on sm121 (see Blocker B taxonomy) |
| `NPUCompressedTensorsW4A8Int8DynamicMoE` | INT4 | INT8 | **no** — Ascend NPU only |
| ModelSlim `ModelSlimW4A8Int8MoE` (`W4A8_DYNAMIC`) | INT4 | INT8 | **no** — Ascend NPU only |
| `CompressedTensorsW8A8Fp8MoE` | FP8 | FP8 | yes |
| `CompressedTensorsW8A8Int8MoE` | INT8 | INT8 | yes |
| `CompressedTensorsWNA16{MoE,MarlinMoE,TritonMoE}` | INT4/INT8 group | 16-bit (weight-only) | yes |

The only "W4A8" schemes in the codebase are **INT8-activation and Ascend-NPU-only** —
irrelevant on a CUDA/GB10 box. So for Hy3 on GB10, the option space is:

- **4-bit activations at all** → only via **W4A4 NVFP4** (`CompressedTensorsW4A4Nvfp4MoE`
  / ModelOpt `modelopt_fp4`), i.e. exactly the build under investigation in Blocker B.
  There is no lower-effort NVFP4 variant that keeps FP4 activations.
- **"Works today" fallbacks that give up 4-bit activations:**
  - Integer weight-only **W4A16** (`wNa16` marlin/triton, **not** NVFP4 — needs a
    different, int4-group export, not the float4 `tensor_group` format kodelow used).
  - **FP8 W8A8** (`w8a8_fp8_moe`) — doubles weight memory vs FP4 but is robust on
    Blackwell and needs no exotic export.

---

## Solved / cosmetic

- **Hunyuan `:opensource` token-suffix** — backported PR **#29920**
  (`resolve_hunyuan_tokens`) into `sglang_launch.sh` as a runtime monkey-patch
  (after the GLM-5 block): computes `SGLANG_HUNYUAN_TOKEN_SUFFIX` from
  `tokenizer_config.json` `token_suffix`, patches `function_call/hunyuan_detector.py`
  + `parser/reasoning_parser.py`. Gated on `$SGLANG_MODEL` containing Hy3/Hunyuan or a
  hunyuan parser. **Verified.**
- **`Tokenizer ... is still TokenizersBackend` warning** — **cosmetic**; does not affect
  generation.
- **`top_k must be -1 or >=1, got 0`** (HTTP 400) — Hy3 `generation_config` ships
  `top_k=0`, which SGLang rejects. Pass `top_k=-1` (set in each profile's
  `recommended_sampling`).
- **MTP w13 shape mismatch** (2048 vs 4096, exit 137) — MTP head is BF16 but SGLang
  inherited `speculative_draft_model_quantization=modelopt_fp4`. Keep
  `speculative_enabled: false` for first contact (spec off).

---

## Hard rules (this workstream)

- **NO image rollback** — forward-fix only (bump / patch in `sglang_launch.sh`). The new
  image carries needed sm121 + arch fixes.
- **NO vLLM** as "the answer" — the target is SGLang on the FP4 tensor cores.
- **W4A16 is not a substitute** — the user explicitly wants **4-bit activation** working;
  W4A16 is only a diagnostic sibling, and (Blocker A) SGLang can't even load this one —
  now empirically confirmed 0/9, dead end regardless of hardware.
- **No autodeploy** — deploy/delete only on explicit approval.
