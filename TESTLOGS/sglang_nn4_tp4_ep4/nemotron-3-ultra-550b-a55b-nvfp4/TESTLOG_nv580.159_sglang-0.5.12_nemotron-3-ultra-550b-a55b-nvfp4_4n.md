# SGLang Test Log — NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 (NemotronH hybrid), 4 Nodes, TP=4 **EP=4**, v0.5.12 (first contact)

## Environment

| Component | Value                                                                                   |
|-----------|-----------------------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node                                          |
| Driver    | 580.159                                                                                 |
| Kernel    | 6.17.0-1018-nvidia                                                                      |
| OS        | Ubuntu 24.04 LTS (aarch64)                                                              |
| K3s       | v1.35.3+k3s1                                                                            |
| Nodes     | spark1 (head), spark2, spark3, spark4 (workers) — 1 GPU each                            |
| Image     | `scitrera/dgx-spark-sglang:0.5.12` — **UPSTREAM base**, NOT xomoxcc-sm121               |
| Model     | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`                                        |
| Arch      | `NemotronHForCausalLM` (`model_type=nemotron_h`) — Mamba2 + MoE + attn hybrid           |
| Quant     | `modelopt_mixed` / `MIXED_PRECISION` (expert FFN FP4 g16, attn/latent/MTP/emb FP8/BF16) |
| Parallel  | **TP=4, PP=1, EP=4** (expert-parallel; `moe_a2a_backend=none`, naive all-gather)        |
| Transport | **RoCE** via SR-IOV VF                                                                  |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep4/nemotron-3-ultra-550b-a55b-nvfp4/nv580.159_sglang-0.5.12_nemotron-3-ultra-550b-a55b-nvfp4_n4_ep4.yaml`
Results: `kikube/matrixtest/2026-06-06/results/sglang_nn4_tp4_ep4/nemotron-3-ultra-550b-a55b-nvfp4/0.5.12/`

**First contact for the Ultra on this cluster at EP=4** — no prior baseline. The profile `roles/k8s_dgx/model_profiles/nvidia-nvidia-nemotron-3-ultra-550b-a55b-nvfp4.yml` carried first-contact defaults derived from the validated Super sibling; this matrix is the first Ultra validation. Backend choices inherited from Super (flashinfer_cutlass MoE, flashinfer attn, full-CG + piecewise-off) are re-confirmed here.

> **THROUGHPUT METRIC: peak = Σ per-request tok/s** (sum of each request's `tokens_per_sec`), NOT `aggregate_throughput` (total_tokens / wall_time). This distinction flips the headline result — see Finding #3. The `MATRIX_SUMMARY.json` "winner" field uses aggregate and is **misleading here**.

---

## Model Notes (observed, EP=4, mfs0.90, ctx262k — Case 03 baseline)

- 550B total / 55B active **LatentMoE** hybrid. config.json: 108 layers (48 mamba + 48 moe + 12 attention), hidden 8192, 64 attn heads, `num_key_value_heads=2` (GQA), 512 routed + 1 shared experts, 22 active/token, `ssm_state_size=128`, `mamba_num_heads=256`.
- **NoPE** — Mamba2 carries order; no RoPE/YaRN. config cap `max_position_embeddings=262144`; extend via `context_length` + `json_model_override_args` (auto-sets `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`).
- Memory footprint (TP=4 EP=4): weights **83.72 GB/GPU** (`modelopt_mixed`/MIXED_PRECISION — note: NOT the ~107 GB/GPU the profile header estimated), load ~490 s. With `--max-mamba-cache-size 48`: Mamba pool **48 slots** (ssm 4.59 GB + conv 0.06 GB), KV **774 547 tokens** (K 1.11 + V 1.11 GB, fp8), CG capture 2.73 GB, `avail_mem` **15.09 GB** after capture.
- **Concurrency:** `--max-mamba-cache-size 48` → `max_running_requests=16` (= 48 // mamba_ratio 3). Validates the dgxarley `max_mamba_cache_size` plumbing end-to-end (see Finding #6). n8 tests never queue.
- Reasoning parser `nemotron_3`, tool-call parser `qwen3_coder`. MTP out of scope (same two blockers as Super: `--disable-radix-cache` not plumbed + upstream #21138 accept-rate ≈ 0).

## Closed axes (hard constraints, not swept)

- **attention = flashinfer ONLY** — triton attn hard-asserted off on NemotronH (`apply_nemotron_h_defaults`).
- **piecewise CG = off** (card sets `--disable-piecewise-cuda-graph`).
- MoE FFN quant = `modelopt_fp4`; DeepGemm disabled (NVFP4 scale_fmt ≠ ue8m0).
- All cases: `tp=4, pp=1, ep=4, nccl=roce, attn=flashinfer, kv=fp8_e4m3, cuda_graph_max_bs=8, max_mamba_cache_size=48, max_running_requests=32` unless noted.

---

## Results — peak throughput (Σ per-request tok/s)

| #  | mfs  | ctx      | fp4_gemm     | moe        | Status                          | n1 peak | n4 peak |  n8 peak | n8 ok | n8 ttft | Output  |
|----|------|----------|--------------|------------|---------------------------------|--------:|--------:|---------:|-------|--------:|---------|
| 01 | 0.85 | 262k     | fi_cutlass   | fi_cutlass | **CRASH** — "Not enough memory" |       — |       — |        — | —     |       — | —       |
| 02 | 0.88 | 262k     | fi_cutlass   | fi_cutlass | ok                              |    10.2 |    29.3 | **43.4** | 8/8   |    1.77 | clean ✓ |
| 03 | 0.90 | 262k     | fi_cutlass   | fi_cutlass | ok ← **profile shape**          |    10.1 |    27.9 |     42.0 | 8/8   |    1.95 | clean ✓ |
| 04 | 0.92 | 262k     | fi_cutlass   | fi_cutlass | ok                              |    10.1 |    29.6 |     42.8 | 8/8   |    1.63 | clean ✓ |
| 05 | 0.90 | **524k** | fi_cutlass   | fi_cutlass | ok                              |    10.0 |    29.5 |     43.2 | 8/8   |    2.37 | clean ✓ |
| 06 | 0.90 | 262k     | **fi_cudnn** | fi_cutlass | ok                              |    10.1 |    28.9 |     42.0 | 8/8   |    1.71 | clean ✓ |
| 07 | 0.90 | 262k     | fi_cutlass   | **triton** | **CRASH** — cutlass_moe assert  |       — |       — |        — | —     |       — | —       |

Per-request decode is a flat **~5.3 tok/s** across every working case (it's a 550B/55B-active model on TP=4). The n8-peak spread (42.0–43.4) is within run-to-run noise.

---

## Crash details

**Case 01 — `mem_fraction_static: 0.85` (startup_crash).** During KV profiling:
```
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```
This is the **KV-pool-goes-negative** crash, not a weight-load OOM. Under EP=4 the expert-dispatch buffers shrink `rest_memory`; at mfs0.85 the held-back reserve `(1−0.85)·pre` exceeds the post-weight free memory, so `KV = post_weight_free − reserve` goes negative. 0.88 is the EP=4 floor; 0.90 is the profile default. (Note: mem_fraction_static is a *post-weight reserve* knob — higher = smaller reserve = more KV — NOT a vLLM-style fraction-of-total.)

**Case 07 — `moe_runner_backend: triton` PROBE (startup_crash).** Despite `--moe-runner-backend triton`, the launch crashes in the **cutlass** FP4 MoE path during CG capture:
```
sglang/srt/layers/moe/cutlass_moe.py:427  nx2_w1 == params.intermediate_size_per_partition * 2
AssertionError: mismatch in expected `n`
```
Identical signature to Super's Case 06. `ModelOptFp4` always routes the FFN through `cutlass_moe_fp4`; the triton runner flag is effectively ignored, and the LatentMoE/512-expert shape trips the hard cutlass assertion. **triton MoE is not viable on Nemotron-3-NVFP4** (EP=4 confirms the Super finding) — `flashinfer_cutlass` is the only MoE runner.

---

## Findings

1. **EP=4 boots and serves on the mainline `scitrera:0.5.12`.** 5/7 cases serve cleanly at n8 8/8, output coherent (spot-checked: no `!`-collapse, no word-salad; the only char-runs are markdown table separators). Per-request ~5.3 tok/s, n8 peak ~42–43 tok/s.

2. **`mem_fraction_static` floor under EP=4 is 0.88, not 0.85.** Case 01 (0.85) startup-crashes with the KV-negative "increase --mem-fraction-static" error; 0.88/0.90/0.92 all serve with flat peak throughput (43.4/42.0/42.8). → Profile header corrected ("0.85 is the floor" was wrong for EP=4).

3. **🔑 `flashinfer_cudnn` does NOT beat `flashinfer_cutlass` — they TIE on peak (42.0 = 42.0 tok/s n8).** The `MATRIX_SUMMARY` "winner = cudnn (39.39 vs 35.66)" is an **aggregate** artifact (total_tokens / wall_time, sensitive to the finish-reason/wall-time mix), not kernel speed. On the correct peak metric there is zero difference. → fp4_gemm stays **flashinfer_cutlass** (Super-validated, safe). The initial profile edit to cudnn was reverted once peak was computed.

4. **⚠ Unresolved cuDNN discrepancy vs Super.** cuDNN-FP4 *booted and served* here (Case 06), but the Super sibling's Case 05 (2026-06-04, **same `scitrera:0.5.12` tag**) *startup-crashed* in flashinfer's `_check_cudnn_availability` ("install nvidia-cudnn-cu12 … FP8 GEMM functions"). The current image carries **native cuDNN** (`libcudnn.so.9`, `torch.backends.cudnn` v9.2.0 available) but **not** the `nvidia-cudnn-cu12` Python wheel that flashinfer's check wants. Same tag, opposite outcome — not root-caused (FP8-path-specific check? image delta between dates?). Since cuDNN gives no peak gain anyway (#3), this stays a documented curiosity, not a production dependency.

5. **512K context is essentially free.** Case 05 (524k, mfs0.90) serves at n8 peak 43.2 vs 42.0 (262k) — within noise, 8/8 clean. NoPE + 96/108 non-attention layers mean KV barely grows. 512K is viable on EP=4; only RULER quality (not throughput) argues for a shorter cap. 1M still untested.

6. **`max_mamba_cache_size: 48` plumbing validated end-to-end.** Boot logged `max_mamba_cache_size: 48` → `max_running_requests=16` (= 48 // ratio 3, exactly as predicted), ssm_state 4.59 GB, clean serve. The dgxarley `mamba_full_memory_ratio` / `max_mamba_cache_size` keys (added 2026-06-08) work. On hybrid NemotronH the mamba pool — not KV or cuda_graph — is the concurrency ceiling.

7. **EP=4 vs EP=1 trade (context).** EP shards the 512 experts (128/GPU) rather than TP-splitting them; weights are unchanged (~83.7 GB/GPU either way). EP adds dispatch buffers (here `moe_a2a_backend=none` → naive all-gather) that shrink `rest_memory`, which is why the mfs floor rises to 0.88. No EP-1 Ultra peak baseline exists yet for a clean A/B.

---

## Production recommendation

Profile shape already matches a clean winner (Case 03 / Case 05):

```yaml
# roles/k8s_dgx/model_profiles/nvidia-nvidia-nemotron-3-ultra-550b-a55b-nvfp4.yml
moe_runner_backend: "flashinfer_cutlass"   # ONLY viable runner (triton crashes — Finding #2/Case 07)
attention_backend: "flashinfer"            # triton hard-asserted off on NemotronH
fp4_gemm_backend: "flashinfer_cutlass"     # TIES cudnn on peak; cudnn has the Super-crash discrepancy (Finding #3/#4)
disable_cuda_graph: false                  # full-CG
disable_piecewise_cuda_graph: true
mem_fraction_static: "0.9"                 # EP=4 floor is 0.88; 0.90 validated middle (Finding #2)
max_mamba_cache_size: 48                   # → 16 parallel (Finding #6)
context_length: 262144                     # 524288 also free if long context wanted (Finding #5)
ep_size: 4 / tp_size: 4
```

Profile-header edits made after this run: fp4_gemm rationale (peak-tie + cuDNN discrepancy), mfs floor 0.88, 512K validated, max_mamba_cache_size 48 validated.

## Action items / follow-ups

- [x] **`cuda_graph_max_bs` aligned to validated value:** profile set to `8` (was 16) to match the matrix. NOTE: max_mamba 48 allows 16 parallel but the matrix only drove n8 (≤8 concurrent), so bs:8 covered every tested batch. To get CUDA-graph coverage for decode batches 9–16, run an **n16 matrix** at `cuda_graph_max_bs:16` first and check the boot log for CG-capture OOM headroom — UNVALIDATED until then.
- [ ] Run an **EP=1 Ultra** matrix for a clean peak A/B vs EP=4 (the EP=4 mfs floor + dispatch-buffer cost may not be worth it if EP=1 serves equal peak at lower memory pressure).
- [ ] Root-cause the cuDNN Super/Ultra discrepancy (Finding #4) only if fi_cudnn is ever actually wanted — low priority (no peak gain).
- [ ] Revisit MTP once `--disable-radix-cache` is exposed AND upstream #21138 closes.
