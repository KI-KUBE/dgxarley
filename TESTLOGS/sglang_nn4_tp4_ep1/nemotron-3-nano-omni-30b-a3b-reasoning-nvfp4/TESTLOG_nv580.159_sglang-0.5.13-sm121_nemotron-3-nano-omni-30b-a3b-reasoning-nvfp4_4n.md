# SGLang Test Log — Nemotron-3 Nano Omni 30B-A3B-Reasoning-NVFP4 (Omni MoE/Mamba hybrid), 4 Nodes, TP=4 EP=1, v0.5.13-sm121 (first contact)

## Environment

| Component | Value                                                                       |
|-----------|-----------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell-Consumer), 128 GB unified per node             |
| Driver    | 580.159                                                                     |
| Kernel    | 6.17.0-1021-nvidia                                                          |
| OS        | Ubuntu 24.04.4 LTS (aarch64)                                                |
| K3s       | v1.36.1+k3s1                                                                |
| Nodes     | spark1 (head/rank0), spark2, spark3, spark4 (1 GB10 each)                   |
| Image     | `xomoxcc/dgx-spark-sglang:0.5.13-sm121` (PROFILE-PINNED)                    |
| Model     | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` (snapshot dc5f0b0…)   |
| Transport | **RoCE** via SR-IOV VF                                                      |
| Parallel  | tp=4, pp=1, ep=1 (ep=4 probed in case 11)                                   |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4/nv580.159_sglang-0.5.13-sm121_nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4_n4_ep1.yaml`
Profile: `roles/k8s_dgx/model_profiles/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4.yml`

**First contact for this model** — no prior baseline. Architecture sibling for A/B reference is the validated **Super-120B** NemotronH:
- `TESTLOGS/sglang_nn4_tp4_ep4/nemotron-3-super-120b-a12b-nvfp4/TESTLOG_nv580.159_sglang-0.5.13-mtp_nemotron-3-super-120b-a12b-nvfp4_4n.md` (same hybrid family; Super HAS an MTP head + EP=4 winner, this one does NOT — see Model Notes).

Why the SM121 build: the stock `scitrera/dgx-spark-sglang` image device-asserts on the triton/cutlass NVFP4 MoE path on SM121 (see `CUTLASS_NVFP4_SM121_PRD.md`); the `xomoxcc/…:0.5.13-sm121` build carries both the SM121 NVFP4-MoE fix AND the Omni arch class. ⚠️ **Audio gap:** `librosa` is NOT in this image — the Parakeet audio path would fail at runtime; add it to the recipe before any audio test. This matrix is TEXT-ONLY.

---

## Model Notes

- OMNI-MODAL wrapper `NemotronH_Nano_Omni_Reasoning_V3` around a NemotronH text core (`NemotronHForCausalLM`, `model_type=nemotron_h`). **Mamba2 + MoE + attention HYBRID.**
- Text core: 52 layers, hidden 2688, 32 attn heads, num_kv_heads 2 (GQA), 128 routed + 1 shared experts, 6 active/token, expert_intermediate 1856, native `max_position_embeddings=262144`. NoPE (Mamba2 carries order → context extension is just a cap-lift, no rope_scaling).
- NVFP4 modelopt-MIXED (~21 GB weights): routed expert FFN FP4 (E2M1, per-block FP8 E4M3 scales, group_size 16); Mamba in/out_proj + shared experts + attn o_proj FP8; vision (C-RADIOv2-H) + audio (Parakeet) encoders stay BF16.
- Reasoning post-train (`<think>`), `enable_thinking` ON by default; toggle per-request via `extra_body={"chat_template_kwargs":{"enable_thinking":false}}`.
- **NO MTP / speculative decoding.** VERIFIED 2026-06-25 three ways: (1) the served `config.json` has no `num_nextn_predict_layers`/nextn/mtp/draft key anywhere (top-level or nested `llm_config`); (2) the Nano Omni paper (arXiv 2604.24954) never mentions MTP/speculative/draft; (3) MTP is a Nemotron-3 family technique but only the **Super** ships a usable head. No native draft, no external draft → `speculative_enabled=false` everywhere. (Generic web summaries claiming "native MTP" conflate the family/Super discussion — not true for Nano/Omni.)
- Hybrid-Mamba concurrency: `max_running_requests` is clamped by the Mamba state-cache pool (`MambaRadixCache`), NOT by KV/cuda_graph — same as the Super. Without MTP there's no extra_buffer doubling, so the ratio is smaller.

## Closed axes (NOT swept — hard constraints)

- **attention = flashinfer ONLY.** triton attn is HARD-ASSERTED off on NemotronH (`apply_nemotron_h_defaults`: first layer may be Mamba, not attention). Mamba2 SSM layers use their own kernels regardless. No triton-ATTN probe.
- **quant = `modelopt_fp4`**; DeepGemm disabled (NVFP4 scale_fmt != ue8m0).
- **tp_size = 4 fixed** — this is the nn4/TP=4 topology dir. The card's TP=1 single-Spark target (~21 GB fits one 128 GB node) is a DIFFERENT topology and belongs in a separate `sglang_nn4_tp1_ep1` / single-node matrix, not here.
- **speculative / MTP = OFF everywhere** (there is none — see Model Notes).

## Open axes (each case varies ONE axis off the Block-A full-CG baseline = case 02)

A CUDA graph · B reasoning_parser · C mem_fraction_static · D cuda_graph_max_bs · E kv_cache_dtype · F fp4_gemm · G context_length · H ep_size · I moe_runner_backend · J piecewise CUDA graph.

CG variant encoding:
- **no-CG**: `disable_cuda_graph=true` (eager, safest boot)
- **full-CG**: `disable_cuda_graph=false`, `disable_piecewise_cuda_graph=true` (profile default / baseline)
- **piecewise**: `disable_cuda_graph=false`, `disable_piecewise_cuda_graph=false` (PROBE only)

## Dominant risk — Omni-wrapper MoE-defaults resolution (BOOT LITMUS)

The arch class loads, but `flashinfer_cutlass` MoE on this *wrapper* REQUIRES the `sglang_launch.sh` `_sgl_nemotronh_omni_wrapper_` patch (PR #25024). WITHOUT it the wrapper bypasses the NemotronH MoE-defaults hook → llm_config-nested MoE settings unresolved → backend falls to AUTO → the sm_100-only `cutlass_moe_fp4` path → trips the `nx2_w1` shape assert during the flashinfer NVFP4 autotune (even with `moe_runner_backend=flashinfer_cutlass` set). **Case 01 is the litmus**: if it dies at arch-registration, in a mamba kernel, or on the `nx2_w1`/`cutlass_moe_fp4` assert, ALL cases die identically — stop, confirm the launch patch is in this image build, re-run.

---

## Configuration Matrix (13 cases, Blocks A–J)

**Baseline = case 02:** `moe_runner=flashinfer_cutlass, attention=flashinfer, fp4_gemm=flashinfer_cutlass, reasoning=nemotron_3, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.60, full-CG, cuda_graph_max_bs=32, context_length=262144, ep=1, tp=4`. Every other case = baseline with the **one bold Δ** shown.

| #  | Block | axis    | Δ vs case-02 baseline                | Status   | n=1 tok/s | n=4 peak | n=8 peak | Output |
|----|-------|---------|--------------------------------------|----------|-----------|----------|----------|--------|
| 01 | A     | CG      | **no-CG (eager)** — BOOT LITMUS      | UNTESTED | —         | —        | —        | —      |
| 02 | A     | CG      | — (baseline: full-CG)                | UNTESTED | —         | —        | —        | —      |
| 03 | B     | parser  | reasoning_parser **deepseek-r1**     | UNTESTED | —         | —        | —        | —      |
| 04 | C     | mem     | mem_fraction_static **0.75**         | UNTESTED | —         | —        | —        | —      |
| 05 | C     | mem     | mem_fraction_static **0.80**         | UNTESTED | —         | —        | —        | —      |
| 06 | D     | cgbs    | cuda_graph_max_bs **64**             | UNTESTED | —         | —        | —        | —      |
| 07 | D     | cgbs    | cuda_graph_max_bs **128**            | UNTESTED | —         | —        | —        | —      |
| 08 | E     | kv      | kv_cache_dtype **auto (bf16)**       | UNTESTED | —         | —        | —        | —      |
| 09 | F     | fp4_gemm| fp4_gemm **flashinfer_cudnn** PROBE  | UNTESTED | —         | —        | —        | —      |
| 10 | G     | context | context_length **524288** (2×) PROBE | UNTESTED | —         | —        | —        | —      |
| 11 | H     | ep      | ep_size **4** PROBE                  | UNTESTED | —         | —        | —        | —      |
| 12 | I     | moe     | moe_runner **triton** PROBE          | UNTESTED | —         | —        | —        | —      |
| 13 | J     | piecewise | **piecewise CG** PROBE             | UNTESTED | —         | —        | —        | —      |

### Column legend

| Column | Description |
|--------|-------------|
| axis   | which open axis this case varies off the case-02 baseline |
| Status | `UNTESTED` / `ok` / `crash S` (startup) / `crash B` (bench) / `timeout` |
| Output | quality verdict — read the answer text in `kikube-bench-*.log`, confirm `<think>` is split out, pattern-grep + TTR + tail-eyeball |

---

## Pre-run hypotheses (per block)

- **A — CG (01 eager LITMUS / 02 full-CG):** case 01 answers the only first-order question — does the Omni wrapper resolve its MoE defaults + emit coherent tokens. ⚠️ Eager is broken on the *native* `cutlass_moe_fp4` path (CLAUDE.md), but here MoE is `flashinfer_cutlass` (FlashInfer/TRT-LLM autotune, `trtllm::fused_moe`) — likely survives eager. Case 02 (full-CG) is the production-candidate; its risk is the hybrid flashinfer-attn graph capture (`hybrid_linear_attn_backend → flashinfer_backend.init_cuda_graph_state`) — an illegal-memory-access there was seen once on a manual boot but cleared on redeploy (Preliminary Observations).
- **B — reasoning_parser (03 deepseek-r1):** CORRECTNESS axis, not throughput. HF card uses `nemotron_3`; SGLang cookbook §4.8 uses `deepseek-r1`. Verify `<think>` is separated from content (no leaked tags); pick whichever splits cleanly. Judge from the answer text in the `kikube-bench-*.log`, NOT the TESTRESULTS JSON.
- **C — mem (04 / 05):** small model (~21 GB weights), manual boot already showed `available_gpu_mem=42.49 GB` and a huge KV pool at 0.60 → 0.75 and 0.80 should be safe and only widen the (Mamba-clamped) pool. Drop back if any OOMs.
- **D — cuda_graph_max_bs (06=64 / 07=128):** capture-memory headroom on a small model; larger bs can lift batched-decode throughput IF the hybrid Mamba/attn graph still captures cleanly at the larger batch. Keep 32 if a larger bs fails to capture or OOMs on graph memory.
- **E — kv (08 auto/bf16):** fp8 KV has been broken on some arches in this fleet — confirm `fp8_e4m3` holds on the Omni text core and measure the quality/throughput Δ vs bf16. bf16 KV roughly doubles per-token KV cost → smaller pool, but is the safe correctness reference.
- **F — fp4_gemm (09 fi_cudnn PROBE):** kernel delta vs case 02. ⚠️ the 0.5.13-sm121 base may NOT ship the cuDNN-FP4 wheels (cuDNN image layer) — may fail to import. On the Qwen3.6-35B-NVFP4 sibling `fi_cudnn` was broken pre-rebuild and ~10% slower than `fi_cutlass` after — low expectation of a win.
- **G — context (10 → 524288 PROBE):** NoPE → extension is a cap-lift only (`json_model_override_args` auto-sets `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN`). Only answers "does the 2× extension boot + serve", NOT whether long-context quality holds (no published RULER curve). If it OOMs on the KV pool, drop mem_fraction_static or pin chunked_prefill_size.
- **H — ep (11 EP=4 PROBE):** 128 experts % 4 == 0 and the Super's EP=4 was its winner. Watch for the gated-padding / swizzle-pad asserts seen on other NVFP4 MoE models — does the 128-expert NVFP4 layout shard cleanly at EP=4 on this wrapper.
- **I — moe (12 triton PROBE):** on the modelopt NVFP4 path triton normally falls through to `cutlass_moe_fp4`. Single probe to confirm that behaviour on the Omni wrapper (likely no-op fallback or a crash) — keep `flashinfer_cutlass` regardless.
- **J — piecewise (13 PROBE):** profile/Super disable piecewise (the Mamba2/attn hybrid doesn't piecewise-capture cleanly). One probe to confirm that holds on this build (likely crashes / fails to capture). If it boots AND benches, piecewise could be promoted.

---

## Preliminary observations (manual boot — NOT a kikube-bench matrix run)

From running the model through the live `default` SGLang instance on 2026-06-25, BEFORE the matrix was driven. Recorded for context; they do NOT fill the matrix.

- **Profile-default shape (= case 02: full-CG, nemotron_3, mem 0.60, ctx 262144) BOOTS and SERVES.** Head `xomoxcc/dgx-spark-sglang:0.5.13-sm121` started 2026-06-25 15:25:37Z, **0 restarts**, head Ready 2/2 (the `/v1/models` readiness probe passes → it is serving). The Omni wrapper resolves its MoE defaults on this image (litmus concern did NOT materialize on the full-CG shape): NCCL init COMPLETE, weights loaded, FlashInfer autotune (`trtllm::fused_moe::gemm1/2`) completed, MoE backend = flashinfer_cutlass as set.
- Boot log facts: `Tree cache: MambaRadixCache hybrid_ssm=True`, `max_total_num_tokens=19556473`, `max_running_requests=32` (the Mamba-state-cache clamp — NOT KV/graph), `context_len=262144`, `available_gpu_mem=42.49 GB`, `Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set`.
- **Earlier transient:** a prior boot attempt crashed during CUDA-graph capture — `flashinfer_backend.py:693 init_cuda_graph_state: self.cuda_graph_kv_indices[i][0] = 0 → CUDA illegal memory access` → sigquit → head/worker restart cascade. A fresh redeploy (new head hash) cleared it with the SAME cuda-graph config, so it reads as a transient GPU/rank state, not a config defect (mem was not the cause: 44 GB free at capture). If it recurs, check per-node clocks/power FIRST before touching the profile.
- **Tokenizer warning (open):** transformers flags the NemotronH tokenizer with a Mistral-derived "incorrect regex pattern" and suggests `fix_mistral_regex=True`; tokenizer also stays `TokenizersBackend` after `--trust-remote-code` retries ("model-specific attributes may be missing"). No SGLang CLI passthrough for `fix_mistral_regex`. Impact on tokenization is UNMEASURED — encode-diff test pending before deciding whether to patch the cached tokenizer.

---

## Results

**UNTESTED — matrix not yet driven via kikube-bench.** Fill the table above + per-case sections below once the run completes.

Run with:
```
kikube-bench matrix matrixtest_matrices/sglang_nn4_tp4_ep1/nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4/nv580.159_sglang-0.5.13-sm121_nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4_n4_ep1.yaml
```
(append `--dry-run` to preview, `--start-at N` to resume; cases 04/05, 06/07, 08, 11 assume a clean boot from 01/02.)

### Crash legend (for when results land)

- **crash S** (`startup_crash`): head/worker pod restarts — never reaches inference. The kernel/axis combo doesn't compile/load on SM121 for this model.
- **crash B** (`bench_crash`): pod starts, every benchmark request fails (0/n). Inference reachable, first forward pass errors.
- **timeout**: `SGLang not ready after 900s`.

---

## Action items

- [ ] Drive the matrix (13 cases) — run **case 01 (eager litmus) FIRST**; if it dies at arch-registration / mamba kernel / `nx2_w1`/`cutlass_moe_fp4` assert, STOP and confirm the `_sgl_nemotronh_omni_wrapper_` launch patch is in this image build. Cases 03–13 carry information only after 01/02 boot clean.
- [ ] Verify output quality on every `ok` case (pattern-grep + token-distribution + tail-eyeball) — first contact, no prior quality floor.
- [ ] **B (03) correctness:** confirm `<think>` splits cleanly under `nemotron_3` vs `deepseek-r1` — read the actual answer text in `matrixtest/<date>/kikube-bench-*.log`, not the TESTRESULTS JSON. Update the profile if `deepseek-r1` wins.
- [ ] **C (04/05):** if 0.75/0.80 hold, consider lifting the profile `mem_fraction_static` from 0.60 to the best non-OOM value.
- [ ] **D (06/07):** if a larger `cuda_graph_max_bs` captures cleanly AND lifts n=8 peak, bump the profile; else keep 32.
- [ ] **E (08):** record the fp8_e4m3-vs-bf16 KV quality/throughput Δ; keep fp8 unless it regresses quality.
- [ ] **F (09):** if `fi_cudnn` fails to import, note the 0.5.13-sm121 base lacks the cuDNN-FP4 layer (needs the cuDNN-rebuilt image); else log the Δ vs case 02.
- [ ] **G (10):** if 524288 boots+serves, note it only proves boot, NOT long-context quality (no RULER curve) — keep native 262144 in the profile until a quality number exists.
- [ ] **H (11):** if EP=4 shards cleanly and helps, it's a candidate (mirrors the Super winner); watch for gated-padding/swizzle-pad asserts.
- [ ] **I (12) / J (13):** confirm the closed-axis assumptions (triton-MoE falls through / piecewise doesn't capture) hold on this build; document the failure signature.
- [ ] Record the Mamba-state-cache pool line + `max_running_requests` clamp; set `max_mamba_cache_size` explicitly if concurrency needs tuning.
- [ ] Resolve the tokenizer regex question: encode-diff `fix_mistral_regex=True`/`False` in a debug pod; patch the cached tokenizer only if token IDs actually differ.
- [ ] Once a clean boot + coherent-output winner is confirmed, drop the profile's "UNVALIDATED / FIRST-CONTACT" header caveats for the validated axes and flip the profile to the winning shape.
