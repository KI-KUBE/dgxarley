# SGLang Test Log

Chronological record of configuration changes, test results, and crashes for the SGLang deployment.

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (SM121/Blackwell), 128 GB per node |
| Driver | 580.142 |
| CUDA | 13.0 |
| Kernel | 6.17.0-1014-nvidia |
| OS | Ubuntu 24.04.4 LTS (aarch64) |
| K3s | v1.35.3+k3s1 |
| Nodes | spark1, spark2, spark3 (1 GPU each) |

Previous 2-node config (TP=2, EP=2, PP=1) was stable — see `TESTLOG_nv580.142_sglang-0.5.9-dev2_minimax-m2.5-nvfp4_2n.md`.

---

## 2026-03-31: MiniMax M2.5 on 3 nodes — initial bringup

spark3 added as third DGX Spark. `sglang_nnodes` changed from 2 to 3.

### Test 1: TP=3 (auto-derived from nnodes=3)

- **Image:** `scitrera/dgx-spark-sglang:0.5.9-dev2-acab24a7-t5`
- **Model:** `nvidia/MiniMax-M2.5-NVFP4`
- **Config:** `tp_size=3, pp_size=1, ep_size=1, quantization=modelopt_fp4, moe_runner_backend=flashinfer_cutlass, attention_backend=flashinfer, fp4_gemm_backend=auto`
- **Result:** **CRASH** — `AssertionError` in `minimax_m2.py:553`: `assert self.total_num_kv_heads % tp_size == 0`. MiniMax M2.5 has `num_key_value_heads=8`, `8 % 3 != 0`. TP=3 is architecturally incompatible. (Previously worked with TP=2 because `8 % 2 == 0`.)
- **Root cause:** 8 KV heads not divisible by 3. No EP/moe_dense_tp_size workaround — custom model code (`trust_remote_code`) always reads global `tp_size`.
- **Fix:** Switch to Pipeline Parallelism: `tp_size=1, pp_size=3`.

### Test 2: PP=3, TP=1 (first PP attempt)

- **Config change:** `tp_size=1` in profile → `sglang_tp=1, sglang_pp=3` (auto-derived). Added `--pp-size` to launch script. Added `PP` env var to K8s manifests.
- **Config:** `tp_size=1, pp_size=3, ep_size=1, quantization=modelopt_fp4, moe_runner_backend=flashinfer_cutlass, attention_backend=flashinfer, fp4_gemm_backend=auto (→ flashinfer_cudnn on SM121), disable_cuda_graph=false, disable_piecewise_cuda_graph=true`
- **Result:** Server starts successfully! NCCL init with 3 ranks, weight loading completes, warmup prefill passes, API responds on `/v1/models`. `[PP0] Prefill batch ... cuda graph: False` — server fires up.
- **Secondary issue:** Workers crash after ~15 min with `livenessProbe` failure — the `httpGet` probe on port 8000 fails because PP workers have no HTTP server (only PP0/head runs uvicorn).
- **Fix:** Changed worker livenessProbe from `httpGet /health:8000` to `exec pgrep -f sglang.launch_server`.

### Test 3: PP=3, fixed worker livenessProbe

- **Config change:** Worker livenessProbe → exec `pgrep`. Redeployed, clean restart all pods.
- **Result:** Server starts, workers stable (0 restarts). Sent requests — responses successful. But after **~4 minutes** of serving requests:
- **CRASH:** `CUDA error: an illegal instruction was encountered` on spark1 (head/PP0).
- **dmesg (spark1):** `Xid 13: Graphics SM Warp Exception — Illegal Instruction Parameter` on ALL GPCs/TPCs/SMs. Address `0x1c81fb60`, offset `0x1174`. Followed by `Xid 43` (GPU channel exception).
- **Root cause hypothesis:** JIT-compiled kernel (FlashInfer cuDNN FP4 GEMM) produces invalid instructions for SM121.

### Test 4: PP=3, disable_cuda_graph=true

- **Config change:** `disable_cuda_graph: true` (hypothesis: CUDA graph capture triggers bad kernel).
- **Result:** **CRASH** — same Xid 13 at `0x1c81fb60:0x1174` after ~2-4 min. CUDA graphs are not the cause.

### Test 5: PP=3, fp4_gemm_backend=flashinfer_cutlass

- **Config change:** `fp4_gemm_backend: "flashinfer_cutlass"` (bypass cuDNN JIT path), `disable_cuda_graph: true`.
- **Result:** **STABLE.** Server runs 12+ minutes, 3 successful requests, 0 restarts. No Xid errors. `cuda graph: False` confirmed in logs.
- **Conclusion:** The `flashinfer_cudnn` FP4 GEMM backend produces an illegal instruction kernel on SM121. `flashinfer_cutlass` avoids it.

### Test 6: PP=3, flashinfer_cutlass, re-enable CUDA graphs

- **Config change:** `disable_cuda_graph: false` (re-enable, since Xid 13 was cuDNN not CUDA graph related).
- **Result:** **STABLE.** Server runs 9+ minutes, 2 successful requests, 0 restarts. `cuda graph: True` in prefill logs.

### Test 7: PP=3, pp_async_batch_depth=2

- **Config change:** `pp_async_batch_depth: 2` (async micro-batching for PP throughput).
- **Result:** **CRASH** after ~13 min. Worker-1 hits `Connection closed by peer` from head. Head's NCCL watchdog terminated. No Xid in dmesg — pure software crash in PP async scheduling.
- **Fix:** Reverted to `pp_async_batch_depth: 0`.

### Test 8: PP=3, disable_piecewise_cuda_graph=false (re-enable)

- **Config change:** `disable_piecewise_cuda_graph: false` (re-enable piecewise CUDA graphs for variable token lengths).
- **Result:** **CRASH** — `NV_ERR_NO_MEMORY` in `_memdescAllocInternal` (dmesg). Piecewise capture needs 58 chunks × separate graph buffers — exceeds remaining GPU memory after model weights + KV cache reservation.
- **Fix:** Reverted to `disable_piecewise_cuda_graph: true`.

### Test 9: PP=3, disable_cuda_graph=false (regular graphs only)

- **Config:** `disable_cuda_graph: false, disable_piecewise_cuda_graph: true, cuda_graph_max_bs: 16, cuda_graph_bs: [1,2,4,8,12,16]`.
- **Result:** **CRASH** — same `NV_ERR_NO_MEMORY` in `_memdescAllocInternal`. Even 6 regular CUDA graphs exceed memory with 256 experts per stage × 21 layers.
- **Fix:** `disable_cuda_graph: true` — CUDA graphs are not viable for MiniMax M2.5 with PP.

---

## 2026-04-01: MiniMax M2.5 PP — continued stability testing

### Test 10: PP=3, flashinfer_cutlass, no CUDA graphs (known-stable config overnight)

- **Config:** `tp_size=1, pp_size=3, fp4_gemm_backend=flashinfer_cutlass, attention_backend=flashinfer, disable_cuda_graph=true, disable_piecewise_cuda_graph=true, pp_async_batch_depth=0`
- **Result:** **CRASH** after ~10 min. Worker-2 (spark3) hit `Xid 13: Illegal Instruction` at `0x1c81fb60:0x1174` — same address as before.
- **Conclusion:** `flashinfer_cutlass` fp4_gemm only delayed the crash, didn't prevent it. The offending kernel at `0x1c81fb60` is NOT the attention kernel or the FP4 GEMM kernel — it fires regardless of which backend handles those. It's in a code path shared by all configurations.

### Test 11: PP=3, attention_backend=triton

- **Config change:** `attention_backend: "triton"` (bypass FlashInfer attention entirely).
- **Config:** `tp_size=1, pp_size=3, moe_runner_backend=flashinfer_cutlass, fp4_gemm_backend=flashinfer_cutlass, attention_backend=triton, disable_cuda_graph=true, disable_piecewise_cuda_graph=true, pp_async_batch_depth=0`
- **Result:** **CRASH** after ~8 min. Spark3 (PP2): `Xid 13` at `0x1c81fb60:0x1174` — identical to all previous Xid 13 crashes. Spark1 (PP0): `NV_ERR_NO_MEMORY` (cascade from spark3 crash).
- **Conclusion:** Triton attention does NOT help. The illegal instruction is NOT in the attention kernel. The `moe_runner_backend=flashinfer_cutlass` is the remaining FlashInfer component — its MoE dispatch kernel is the likely culprit (runs on every forward pass regardless of attention/fp4_gemm backend choice).

### Test 12: PP=3, moe_runner_backend=triton, attention_backend=flashinfer

- **Config change:** `moe_runner_backend: "triton"` (bypass FlashInfer MoE dispatch), `attention_backend: "flashinfer"` (restored — attention was never the problem).
- **Config:** `tp_size=1, pp_size=3, moe_runner_backend=triton, fp4_gemm_backend=flashinfer_cutlass, attention_backend=flashinfer, disable_cuda_graph=true, disable_piecewise_cuda_graph=true, pp_async_batch_depth=0`
- **Note:** With NVFP4, `moe_runner_backend=triton` falls back internally to `cutlass_moe_fp4` (Triton has no native FP4 MoE kernel). If this uses the same CUTLASS FP4 kernel as `flashinfer_cutlass`, the crash may persist. EP=1 (PP mode), so the `CutlassMoEParams` EP bug is not triggered.
- **Result:** **STABLE 32+ min** (longest run so far). No Xid errors.
- **Throughput:**

  | Metric | 1 request | 4 parallel |
  |--------|-----------|------------|
  | Wall time | 99.8s | 685.5s |
  | Successful / failed | 1 / 0 | 3 / 1 |
  | Total output tokens | 1508 | 12974 |
  | Aggregate throughput | 15.1 tok/s | 18.9 tok/s |
  | Avg TTFT | 0.93s | 1.58s |
  | Avg per-request tok/s | 15.1 | 10.7 |
  | P50 per-request tok/s | 15.1 | 11.1 |

- **Conclusion:** The FlashInfer CUTLASS **MoE dispatch kernel** (`moe_runner_backend=flashinfer_cutlass`) was the source of Xid 13 all along. `moe_runner_backend=triton` (which falls back to `cutlass_moe_fp4` for NVFP4 — a different code path) avoids the illegal instruction. FlashInfer attention is fine.

### Test 13: PP=3, triton MoE, re-enable CUDA graphs (max_bs=8)

- **Config change:** `disable_cuda_graph: false`, `cuda_graph_max_bs: 8` (reduced from 16 to lower capture memory pressure). Tests whether CUDA graph OOM (Tests 8+9) was caused by `flashinfer_cutlass` MoE graphs or by graph capture in general.
- **Config:** `tp_size=1, pp_size=3, moe_runner_backend=triton, fp4_gemm_backend=flashinfer_cutlass, attention_backend=flashinfer, disable_cuda_graph=false, disable_piecewise_cuda_graph=true, pp_async_batch_depth=0, cuda_graph_max_bs=8`
- **Result:** *pending*

---

## Configuration Matrix (MiniMax M2.5, 3× DGX Spark)

All tests use: `tp=1, pp=3, ep=1, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.80, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

| # | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | Throughput (4∥) |
|---|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|-----------------|
| 1 | — | — | — | — | — | — | — | TP=3 AssertionError | — |
| 2 | fi_cutlass | flashinfer | auto→cudnn | false | true | 0 | 16 | OK startup, livenessProbe kill | — |
| 3 | fi_cutlass | flashinfer | auto→cudnn | false | true | 0 | 16 | Xid 13 ~4min | — |
| 4 | fi_cutlass | flashinfer | auto→cudnn | true | true | 0 | — | Xid 13 ~2-4min | — |
| 5 | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | **STABLE 12+ min** | — |
| 6 | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 16 | **STABLE 9+ min** | — |
| 7 | fi_cutlass | flashinfer | fi_cutlass | false | true | 2 | 16 | NCCL crash ~13min | — |
| 8 | fi_cutlass | flashinfer | fi_cutlass | false | false | 0 | 16 | OOM piecewise capture | — |
| 9 | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 16 | OOM graph capture | — |
| 10 | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | Xid 13 ~10min (spark3) | — |
| 11 | fi_cutlass | triton | fi_cutlass | true | true | 0 | — | Xid 13 ~8min (spark3) | — |
| 12 | triton | flashinfer | fi_cutlass | true | true | 0 | — | **STABLE 32+ min** | 15.1 / 18.9 tok/s |
| 13 | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | *pending* | — |

### Column Legend

| Column | Description |
|--------|-------------|
| moe_runner | `moe_runner_backend` — MoE expert dispatch kernel (`fi_cutlass` = flashinfer_cutlass, `triton` = triton→cutlass_moe_fp4 fallback for NVFP4) |
| attention | `attention_backend` — attention kernel |
| fp4_gemm | `fp4_gemm_backend` — FP4 dense GEMM kernel (`auto→cudnn` = auto-selected flashinfer_cudnn on SM121) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs for batch sizes in `cuda_graph_bs` |
| dis_piecewise | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs (58 chunks) |
| pp_async | `pp_async_batch_depth` — async micro-batches in PP pipeline (0 = synchronous) |
| cuda_graph_max_bs | `cuda_graph_max_bs` — largest batch size to capture (— = N/A when graphs disabled) |
| Throughput (1∥ / 4∥) | Aggregate tok/s: single request / 4 parallel requests |
