# SGLang v0.5.16: Relevant Changes Since v0.5.15.post1

Source: [Release Notes v0.5.16](https://github.com/sgl-project/sglang/releases/tag/v0.5.16) · Published 2026-07-25 · 574 PRs / 169 contributors · Diff: [`v0.5.15...v0.5.16`](https://github.com/sgl-project/sglang/compare/v0.5.15...v0.5.16)

> The release notes compare against **v0.5.15**, not against `v0.5.15.post1`. The `.post1` cherry-picks (2026-07-14) are included in that range, but some of them still show up as "new" in the list below even though we already had them in the production image.

Companion documents: [`scripts/patches/sglang-0.5.16-sm121.recipe`](./scripts/patches/sglang-0.5.16-sm121.recipe) (build side, open risks A through G), [`FLASHINFER_0.6.16rc3_TODO.local.md`](./FLASHINFER_0.6.16rc3_TODO.local.md) (flashinfer side), [`TURBOQUANT.md`](./TURBOQUANT.md) (NVFP4 backend matrix).

---

**TL;DR:** for us, v0.5.16 is **not a feature release, it is a restructuring release**. Three things dominate:

1. **PR #30448 deletes the in-tree NVFP4 path.** `cutlass_moe_fp4()`, the JIT kernels including `nvfp4_blockwise_moe.cuh`, and `--fp4-gemm-backend cutlass` are gone. NVFP4 GEMM now mandatorily goes through FlashInfer. That makes our **central SM121 kernel patch moot** (`APPLY_SGL_KERNEL_SM121=0` in the recipe), and not because it was merged, but because its file no longer exists.
2. **The runtime patch landscape has shifted.** Six of the 39 patches under `roles/k8s_dgx/files/sglang_patches/` drifted on 2026-07-28. All six are resolved by now, but in **two different ways**: two were re-targeted (fix still runs), four are self-gated to `< 0.5.16` (fix is inert because its target is gone). The acceptance gate is green, but that explicitly does not mean all six fixes are active.
3. **Behavioral defaults change under us:** UnifiedRadixTree is now the default for SWA, Mamba and DSA models, chunked input-logprob is on, FA3 sparse-mask kernels are off. That hits exactly our two active families (Qwen3.6/GDN and GLM-5.2/DSA).

The image has been on `xomoxcc/dgx-spark-sglang:0.5.16-sm121` since 2026-08-03 (commit `5371537`, default plus all affected profiles). **It is not GPU-serving-validated**, the acceptance gate is import-level (see §0).

---

## 0. Status in the Repo

| Item | State |
|---|---|
| Default image | `xomoxcc/dgx-spark-sglang:0.5.16-sm121` (`roles/k8s_dgx/defaults/main/sglang.yml:10`), bumped 2026-08-03 with `5371537` |
| Recipe | `scripts/patches/sglang-0.5.16-sm121.recipe`, renamed 2026-08-02 from `-dev-` (the `-dev` tag stays frozen, it carries flashinfer 0.6.16rc3 and is the measurement baseline for p36/p37) |
| flashinfer | 0.6.16 final (2026-07-31), two minors above SGLang's own 0.6.14 pin |
| Acceptance gate | ✅ clean 2026-08-02, `scripts/verify_sglang_image.sh`, three scenarios (bare / GLM-5.2 / Hy3), 0 anchor drift |
| GLM-5.2 serving block | lifted (p34 is v0.5.16-compatible, commit `d21f59e`) |
| Active model | `vroomfondel/Qwen3.6-35B-A3B-NVFP4-MTP-ModelOpt` (GDN/linear attention, NEXTN MTP, uniform W4A4, fp8 KV) |

⚠️ **Scope of the gate:** it checks whether a patch still finds its anchor and whether the model registry imports. It does not execute a kernel and does not serve a token. Risks A, B, E and G from the recipe stay open until a real serving run.

---

## 1. Breaking Changes That Affect Us Directly

### 1.1 ⛔ NVFP4 refactor (#30448), the core of the release for us

`--fp4-gemm-backend cutlass` is removed, together with the in-tree NVFP4 JIT kernels. `auto` now picks `flashinfer_cutedsl` on SM100 and `flashinfer_cutlass` on SM120/121. Consequences:

- **Our SM121 kernel patch is unreachable, not made-redundant-by-merge.** `python/sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh` (the `StageCount<1>` fix for the 101 KB shared-memory budget, see `CUTLASS_NVFP4_SM121_PRD.md`) no longer exists. The build would abort in the dry run with `APPLY_SGL_KERNEL_SM121=1`.
- **The entire NVFP4 MoE path on GB10 now hangs off flashinfer.** That is why the flashinfer pin (0.6.16 instead of 0.6.14) carries more weight than in any release before, and why it is the first rollback candidate on any unexplained kernel failure.
- **Profiles that do NOT pin `flashinfer_cutlass` have to be re-checked individually.** Current state in the repo (NVFP4 profiles without `flashinfer_cutlass`):

  | Profile | `moe_runner_backend` | Why it deviates |
  |---|---|---|
  | `redhatai-qwen3.6-35b-a3b-nvfp4` | `triton` | compressed-tensors NVFP4, forced by SGLang's MoE whitelist |
  | `saricles-minimax-m2.5-reap-139b-a10b-nvfp4-gb10` | `triton` | same, plus PP=4 history |
  | `nvidia-llama-4-scout-17b-16e-instruct-nvfp4` | `triton` | |
  | `nvidia-gemma-4-26b-a4b-nvfp4` | `triton` | Gemma4 allowlist |
  | `nvidia-diffusiongemma-26b-a4b-it-nvfp4` | `triton` | the dLLM handler forces triton anyway |
  | `kodelow-hy3-nvfp4-w4a16` | `marlin` | W4A16, different path |

  > Correction to the recipe header (Risk A): it still says `nvidia/Qwen3.5-397B-A17B-NVFP4 (cutlass)`. That profile has since moved to `flashinfer_cutlass` (`nvidia-qwen3.5-397b-a17b-nvfp4.yml:65`), so the point is settled for it. The triton profiles above are the actual remainder.

- **Second, separate trap (Risk B):** compressed-tensors NVFP4 MoE with `apply_router_weight_on_input=True` now **asserts** instead of running. That potentially hits exactly the compressed-tensors profiles from the table.

### 1.2 QServe / FBGEMM FP8 removed (#31109), CUTLASS FP8 blockwise deleted for SM90/SM100, SM120 moved to JIT (#30438)

We do not use QServe (QoQ W4A8) or FBGEMM-FP8, so that part is inconsequential. The second point is more relevant: **SM120 has been switched to JIT for FP8 blockwise** and got `SwapAB`. Affects our FP8 profiles (`qwen-qwen3.6-35b-a3b-fp8`, `qwen-qwen3.5-122b-a10b-fp8`, and `cyankiwi-hermes-4.3-36b-awq-4bit` only indirectly). Expected effect: first-start JIT cost instead of AOT, identical afterwards. On the first boot of an FP8 profile on the new image, keep an eye on startup time (head startup is 7 to 8 minutes anyway, see CLAUDE.md).

### 1.3 Flag renames without deprecated alias

- `--enable-deepep-waterfill` → `--enable-waterfill` (#27350)
- `--optimistic-prefill-retries` → `--optimistic-prefill-attempts` (#30951)

✅ **No consequence for us:** neither flag appears anywhere in the repo (`grep` over `roles/` and `scripts/` only returns the recipe note itself).

**Separately from that** (already due before v0.5.16, handled in `sglang_launch.sh` on 2026-07-28): `--cuda-graph-max-bs` → `--cuda-graph-max-bs-decode`, `--disable-cuda-graph` → `--cuda-graph-backend-{decode,prefill}=disabled`, `--disable-piecewise-cuda-graph` → `--cuda-graph-backend-prefill=disabled`, `--mamba-scheduler-strategy` → `--mamba-radix-cache-strategy`. The launch script now unconditionally emits the new spellings (both variants exist in 0.5.15.post1 AND 0.5.16). The **profile keys are unchanged**, this was purely a launch-script matter.

### 1.4 UnifiedRadixTree is the default for SWA, Mamba and DSA models (#30468)

A behavioral change, not a build break, but it hits **both** active families:

- **Qwen3.6-35B-A3B-NVFP4-MTP** (active model): GDN/Mamba linear attention, runs `mamba_scheduler_strategy: extra_buffer` plus `enable_spec_v2: true`. This exact combination was historically the source of the "not compatible with radix cache" boot crash. The accompanying PRs #30636 (replay SSM sync), #30626 (mamba int8 checkpoint sync), #31643 / #31648 (a cache hit now only resets the state actually used) are the precondition for this to hold.
- **GLM-5.2 (DSA)**: affected as well.

→ On the first serving run on 0.5.16 this is the thing I would watch first: boot behavior with `extra_buffer` plus spec-v2, and whether accept length stays the same.

### 1.5 Chunked input-logprob processing on by default (#31498)

Caps peak memory. In addition, the logprob results were consolidated into a unified `LogprobResult` and the chunk env vars were renamed (#31733). Relevant for anything requesting `logprobs` (`sglang-raw`, test suites). No action expected, but if logprob responses look structurally different, #31733 is the starting point.

### 1.6 `sglang.kernels` namespace (RFC #29630, #30044 / #31582)

Kernels were **moved verbatim**, only import paths change; the public wrappers stay on the AOT `sgl_kernel` backend. For code that reaches around the wrappers into internal paths this is a break, and that is exactly what our runtime patches do. This is the root cause of four of the six drifts in §2 as well as of the new `TILELANG_PATCH_VARIANT="-v0.5.16"` (the DSA TileLang kernel now lives under `python/sglang/kernels/ops/attention/dsa/`).

### 1.7 Smaller breaks unrelated to us

- FA3 sparse-mask kernels off by default (#30356). Only relevant if a DSA profile implicitly relied on them.
- `num_tokens_per_bs` → `num_tokens_per_req` in the spec-decoding runners (#30977). Internal API, our patches do not touch it.
- Legacy Sphinx `docs/` removed (#28964), Mintlify migration complete. Old doc deeplinks in our comments may 404.
- The SGLang diffusion rollout endpoint returns `application/msgpack` instead of JSON (#31565). We do not run RL rollouts.

---

## 2. Fallout in Our Runtime Patches

Measured on 2026-07-28 against the built image (spark5/podman, all 39 patches executed the way `sglang_launch.sh` executes them, three env scenarios; the same battery against `0.5.15.post1-sm121` as a control with 0 drift, so every hit really is caused by the version jump).

33 of 39 ran clean, six drifted. **All six have been resolved since 2026-08-02, but not all six are still effective:**

| Patch | What happened | Result |
|---|---|---|
| `p24_linear_nvfp4_scale` | anchor in `linear.py` gone (`weight_scale_2` / `nvfp4` no longer appear there at all) | **RE-TARGETED** (`f17cfda`), fix runs |
| `p34_dsa_trtllm_sparse_sm120` | `model_runner_kv_cache_mixin.py` deleted, contents now in `srt/model_executor/pool_configurator.py`; second hunk in `forward_mla.py` needed a rebase | **RE-TARGETED** (`d21f59e`), fix runs. This is the live-proven GLM-5.2 DSA path, which is why the GLM-5.2 serving block is lifted |
| `p26_cutlass_moe_zeroinit` | `cutlass_moe_fp4` no longer exists (#30448) | **SELF-GATED OFF** (`80c6cc3`), fix is moot |
| `p28_modelopt_cutlass_params_ep` | `CutlassMoEParams` class gone entirely | **SELF-GATED OFF** (`fd8a937`). ⚠️ Open: where are the EP expert counts computed now? |
| `p31_dsa_flashinfer_gather` | surrounding code moved (anchor rebase required) | **SELF-GATED** to `< 0.5.16` (`2d0c0e3`), fallback gather path |
| `p33_dsa_fig_graph_split` | same | **SELF-GATED** like p31 (p31 through p33 are one path) |

Correction to the earlier static analysis: **p30 and p35 are fine.** Their supposedly missing targets (`torch_paged_mqa_logits.py`, `triton_paged_mqa_logits.py`) are files these patches **create**, not edit. Both report "created".

**Possible cleanup potential:** v0.5.16 ships a vectorized, CUDA-graph-safe `fp8_paged_mqa_logits_torch_sm120` in `srt/layers/attention/dsv4/indexer.py` (verified present in the image). So upstream may have absorbed what p30 does by hand. Cross-check before the next patch refactor.

Drift is only logged as a warning and never crashes the pod. A deploy would therefore **silently run without the fixes**, which is the reason the gate exists.

---

## 3. New Features With Direct Relevance to Our Profiles

### 3.1 ⭐ GLM/DeepSeek NVFP4 + flashinfer_trtllm "!!!!" collapse fixed (#31001)

NaN routing in long context that led to token collapse. **13 profiles** set `flashinfer_trtllm` somewhere (among others both Ornith profiles, Hy3-W4A4, both Qwen3.6-NVFP4-ModelOpt profiles, DSV4-Flash, Gemma-4-26B-A4B, Minimax-M2.7, Nex-N2). Per `reference_nvfp4_gated_moe_padding`, `flashinfer_trtllm` is also the **only** backend that avoids the "intermediate size required padding" boot assert on gated-MoE NVFP4, so we are not free to choose here.

→ The open **W4A4 NaN on Hy3/HYV3** is the obvious re-test: different kernel, but the same family, and together with flashinfer #3838 (SM120 NVFP4 `qk_correction` layout, row sum, LSE) two independent NaN fixes landed in the NVFP4 attention/routing area in this window.

### 3.2 ⭐ FP4 KV cache design including SM120 (#21601)

Right on our open construction site: `p37_nvfp4_kv_spec_native_verify.py` fixes exactly the silent corruption with `--kv-cache-dtype nvfp4` plus NEXTN (the dequant workspace is never filled under speculation, GSM8K 0/10 at maximum accept length, without any error message; measured on the `0.5.16-dev` image). The upstream PR makes FP4 KV official on SM120.

→ Check whether p37 is still the same fix against the new upstream state, or whether parts of it now apply twice. This is the only point where v0.5.16 touches our active research line substantively instead of just shifting it.

### 3.3 ⭐ ReplaySSM ring spec-verify for GDN (#28695)

Replaces the per-draft SSM snapshot with a ring. Upstream figure: **11.5 GB → 1.8 GB speculation scratch per GPU (6.4x)** on Qwen3.5-35B-A3B at TP1, at equal accuracy and throughput. Opt-in via `--enable-gdn-replayssm-spec`, prerequisites: GDN with a linear draft chain, `--speculative-eagle-topk` in `{None, 1}`, ring length via `--linear-replayssm-cache-len`.

→ **Our active model meets the conditions exactly**: `vroomfondel/Qwen3.6-35B-A3B-NVFP4-MTP-ModelOpt`, `speculative_algo: NEXTN`, `speculative_eagle_topk: 1`, GDN linear attention. We run TP4 instead of TP1, so the absolute saving scales differently, but freed speculation scratch here directly means room for a higher `mem_fraction_static` (the profile has it at 0.6, and higher means more KV reserve after the weight load). **This is the single most worthwhile test of this release for us.** Off by default, so low-risk to try.

### 3.4 ⭐ DSpark, confidence-driven speculation (#30261, #31434)

New spec algorithm: drafts semi-autoregressively in blocks and sizes the verify window from the draft's confidence instead of from a fixed draft length. Upstream: 383.7 tok/s at accept length ~5 on DeepSeek-V4-Pro, TP8/B300, bs=1. Activation: `--speculative-algorithm DSPARK` plus `SGLANG_RAGGED_VERIFY_MODE=compact`, block size via `--speculative-dspark-block-size`.

→ We already have a profile for it: **`mmangkad/DeepSeek-V4-Flash-0731-NVFP4`** (created with `5371537`, `speculative_algo: DSPARK`, `speculative_num_draft_tokens: 6` = `dspark_block_size` 5 + 1, a hard requirement). The checkpoint ships the DSpark head in the same `mtp.0.*` block. **UNVALIDATED first contact**, and this profile is the only reason 0.5.16 is needed at all: DSPARK does not exist in 0.5.15. Constraints from the profile: `pp_size == 1` mandatory, the DSpark draft MoE is MXFP4 (the same SM100 trap as the preview's MTP head, hangs off the marlin branch), and the DSpark kernels under `srt/speculative/dspark_components/kernels/` are Triton with a Torch fallback.

### 3.5 GLM-5.2 DSA cache layer split under prefill CP (#29421)

KV and indexer cache layers are sharded across CP ranks, each rank owning a disjoint layer range. Upstream: **-74 % KV per rank** (0.77 → 0.20 GB) at 8192 tokens, GLM-5.2-FP8, 78 layers, `cp_size=4`. Activation: `--enable-dsa-cache-layer-split`, requires `--enable-prefill-cp --cp-strategy interleave`.

→ **Not usable right now**: we do not run context parallel anywhere (`grep` over `roles/` finds neither `cp_size` nor `enable_prefill_cp`). It is, however, the first upstream lever that makes CP attractive on our GLM-5.2-REAP profiles (504B / 526B, which sit tight on memory). Note it, do not implement it yet: CP has its own communication cost on 4×GB10, and per `reference_glm52_dsa_indexer_deepgemm_sm121` GLM-5.2's decode rate is limited by bf16 attention GEMVs, not by KV space.

Accompanying: MTP index sharing under prefill CP (#30992), stabilization of GLM-5.2 MTP IndexShare across PD and CUDA-graph replay (#30839), FlashInfer TRT-LLM MoE writes directly into the output (#28416).

### 3.6 DeepSeek V4

- **SM120 `flashinfer_mxfp4` MoE runner plus TP2 (#30272)**, the first DSV4 MoE path implemented explicitly for SM120. Relevant for both V4-Flash profiles.
- **Non-paged indexer by default for large prefill chunks (#30140)**, **BF16 instead of FP32 for indexer score computation (#30012)**, both on the indexer path, which is exactly where our p30 through p35 live.
- **Top-k-v2 emitted invalid indices on tie overflow / inf scores (#30645)**, an IMA in FA3 sparse decode. Correctness fix in the DSA path.
- **Q8KV8 FP8 sparse MLA prefill integrated into the DSA backend (#30514)**, **Wint4Abf16 / Win4Afp8 (#25763)**, **BF16 compress state for online C128 (#29609)**.
- Cherry-picks after the tag: `nvfp4 online scale with pcg` (#32246/#32259), stale flashinfer MLA fallback poisoned the spec-verify capture (#32288/#32346).

### 3.7 Model-specific fixes that hit our profiles

- **Nemotron-3 Super: `reasoning_effort=low` → `low_effort`, warning on unsupported levels (#30463).** We have `nvidia-nvidia-nemotron-3-super-120b-a12b-nvfp4` (plus Nano-Omni and Ultra). Affects the API surface that LiteLLM/Hermes talks to.
- **Ministral3: YaRN RoPE scaling aligned with the Transformers implementation (#31232).** Interesting because of `reference_mistral_native_draft_context_fallback`: on Mistral-native checkpoints SGLang derives 128000 as the context fallback offline because YaRN is unparseable, which produces the target-vs-draft mismatch. Different code path, but the same corner. Cross-check on the next Mistral profile test whether the fallback is still needed.
- **Garbage output on bare-tekken Mistral checkpoints (#30396).** Fits `reference_sglang_mistral_native_support` (name-triggered loading of `params.json` checkpoints). Affects `mistralai-mistral-small-4-119b-2603-nvfp4` and `mistralai-mistral-large-3-675b-instruct-2512-nvfp4`.
- **DeepReinforce Ornith-1.0 in the cookbook (#29404).** We run three Ornith profiles (`ressl`, `vroomfondel` 35B and 9B). Reconcile the cookbook args against our profile keys, that is the authoritative source per `reference_sglang_cookbook_authoritative`.
- **MiMo V2.5 with zigzag context parallel (#29972)**, plus a MiMo-V2 fix on Blackwell (FA3 fallback, TP-aware audio weight loading, #31343). We have `lukealonso-mimo-v2.5-nvfp4`.
- **MiniMax-M3** completes its four-part landing (#28715), but is **not yet usable end to end** (the cookbook points at a dev image, #31819). Irrelevant for our Minimax profiles (M2.5-REAP, M2.7).

### 3.8 New models

| Model | Type | Fits on 4×GB10? |
|---|---|---|
| **Inkling** (975B MoE, 1M ctx, SWA + full + Mamba2, NVFP4 MoE, native MTP) | autoregressive | No. 975B is out of reach even in NVFP4. Interesting only as an architecture reference: it mixes exactly the three attention types we keep wrestling with |
| **LongCat 2.0 FP8** | autoregressive | Worth checking, depending on the parameter count |
| **JetBrains Mellum v2** | autoregressive | Cookbook says "wip" |
| **Pi0.5** | VLA | Not our workload |
| **LongLive 2.0** | Diffusion | Not our workload |

---

## 4. Everything Else, No Immediate Action Needed

- **Attention / linear attention:** first correct KDA-MTP path on SM100 (#30113), GDN/KDA CuteDSL prefill fuses state I/O into the chunk-h kernel (#30169), auto-select for FlashInfer GDN prefill on validated SM100 configs (#29734). **All SM100, not SM120/121**: in this window the GB10 side of the linear-attention work comes from flashinfer (#3960, CuteDSL kernels as `sm_121a` on DGX Spark), not from SGLang.
- **Piecewise / breakable CUDA graph:** breakable prefill CUDA graph for DP attention (#30898). See the known issue in §5, we do not run it.
- **Scheduler:** `--default-chat-template-kwargs` (#29579), `SGLANG_MAX_NEW_TOKENS_LIMIT` as a hard per-request ceiling (#22591, could be interesting for the Hermes multi-tenant use), priority request header override (#30811), `reasoning_effort` schema aligned across chat/tokenize/responses (#31784).
- **HiCache:** FlexKV storage connector (#29701), metadata cache for the file backend (#29716). We have no KV-cache SSD tier defined, still future work.
- **MoE/EP:** elastic EP with runtime scale-up (#30164), waterfill with MegaMoE backend (#27350), EPLB diagnostics (#30646).
- **Not relevant:** AMD/ROCm, NPU/Ascend, CPU/Intel/XPU/MLX, SGLang-Diffusion, PD disaggregation (we do not run disaggregated), gRPC server.

---

## 5. Upstream Known Issues

- **Temperature-0 nondeterminism under DP attention with a breakable prefill CUDA graph.** On the DSV4-Flash-FP4 recipe, the idle-rank dummy extend introduced by #30898 disturbs the logits of real requests. The protecting determinism test was **disabled instead of fixed** as a stopgap (#31125). Avoidable by not enabling the breakable prefill CUDA graph, which we do not. Relevant as soon as someone turns it on for the V4-Flash profiles.
- **flashinfer 0.6.15 was merged and rolled back again in this cycle** (#31502, #31625), the release pins 0.6.14. We deliberately sit two minors above that at 0.6.16, see §6.
- **Mamba track boundary seqlen under the overlap scheduler:** fixed and reverted again (#31369, #31622), the underlying problem is open. Affects the Mamba/GDN family, i.e. our active model.
- CPU AMX optimizations for diffusion reverted (#28527, #30716). Irrelevant.
- **GB300 CI jobs were temporarily disabled in this cycle** (#31764). Blackwell consumer coverage therefore rests even more on manual validation than usual, which rather increases our own testing obligation on SM121.

---

## 6. What Does NOT Come From SGLang: flashinfer 0.6.16

The second driver of this recipe, and for GB10 almost the more important one. Full rationale in the recipe header, here only the key points:

- **#3897, SM121 (GB10/DGX Spark) enabled for NVFP4 attention.** The SM120 kernel compiles and runs unchanged as `sm_121a`, only the capability checks and the AOT/JIT arch gates were missing. Before this, NVFP4 attention was simply unavailable on SM121. Upstream benchmark: 2.3 to 2.4x over FA2-BF16 at `head_dim` 128. **That is exactly the path the preferred uniform-NVFP4 route (including attention) needs.** Verified at kernel level on spark5 on 2026-07-28 (cos 0.9909, rel-L2 0.135 against fp32 SDPA, 2.65x on the forward). End to end through SGLang with a real uniform-NVFP4 model is still outstanding.
- **#3838**, correctness fix in the SM120 NVFP4 attention path (qk_correction layout, row sum, LSE).
- **#3960**, SM12x CuteDSL kernels are compiled as `sm_121a` on DGX Spark (GDN/linear attention, i.e. the Qwen3.5MoE family).
- **#3922**, migration away from the APIs removed in `nvidia-cutlass-dsl` 4.6. Makes our 4.6.1 + flashinfer pairing explicitly upstream-supported for the first time instead of only empirically so.
- The rmsnorm smoke test was repeated on the final build on 2026-08-02 (needed because of #4226, which reworks the CuTe DSL arch guard) and **verified that the DSL path is actually taken** (`_use_cuda_norm = False`), so a green result is not a silent CUDA JIT fallback.

---

## 7. Recommendation

1. **Serving validation is the blocker, not the build.** The image is there, the gate is green, but import-level. The first real run should be the active Qwen3.6-35B-A3B-NVFP4-MTP profile, watching: boot with `extra_buffer` plus spec-v2 under the new UnifiedRadixTree default (§1.4), accept length against the 0.5.15.post1 reference value, GSM8K acceptance. Judge output quality by the usual rule (pattern grep plus token distribution plus tail, not `finish_reason`).
2. **Then `--enable-gdn-replayssm-spec` as an A/B** (§3.3). Conditions are met, the default is off, and the upside is scratch memory that converts directly into KV headroom.
3. **Touch the six triton/marlin NVFP4 profiles from §1.1 individually before their next use**, plus Risk B (compressed-tensors with `apply_router_weight_on_input=True` now asserts). Until they have been looked at, they are unresolved on this image, not broken.
4. **Cross-check p37 against #21601** (§3.2) and reopen the outstanding Hy3/HYV3 W4A4 NaN question against #31001 plus flashinfer #3838 (§3.1). Two independent NaN fixes in the same corner are the best occasion in weeks.
5. **DSV4-Flash-0731 with DSPARK** (§3.4) is the reason 0.5.16 was needed at all, but it is a first-contact profile. Not before item 1, otherwise two unknowns get mixed.
6. **Note CP for GLM-5.2, do not build it** (§3.5). The layer split is real, but our GLM-5.2 bottleneck is currently the decode rate, not KV space.

## Open Questions

- Where are EP expert counts computed after `CutlassMoEParams` went away (p28, §2)? As long as that is open, it is unclear whether the old fix is merely inert or actually obsolete.
- Has v0.5.16 absorbed with `fp8_paged_mqa_logits_torch_sm120` what p30 does by hand?
- Is `moe_runner_backend: triton` still a valid path at all for compressed-tensors NVFP4 after #30448, or does the whitelist now force something else? That decides four of the six profiles in §1.1.
- Is the 128000 context fallback on Mistral-native checkpoints still needed after #31232?
