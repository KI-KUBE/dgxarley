# scitrera `dgx-spark-pytorch-dev:2.10.0-v2-cu131`: SDPA `EFFICIENT_ATTENTION` Silently Returns Wrong Output on sm121 (Blackwell GB10)

**Filing target:** [scitrera/cuda-containers](https://github.com/scitrera/cuda-containers) — build-tooling for `scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131`.
**Suggested issue title:** *PyTorch SDPA `EFFICIENT_ATTENTION` backend silently returns numerically incorrect output on sm121 in `dgx-spark-pytorch-dev:2.10.0-v2-cu131` (works correctly in NGC PyTorch 25.12 with same major version)*
**Severity:** silent correctness — no NaN, no Inf, no exception, no warning. Output magnitude is 12–27× off from a CPU reference.
**Status:** observed and isolated to scitrera's `dgx-spark-pytorch-dev:2.10.0-v2-cu131` build artefact; locally worked around (see `COMFYUI_PROMPT_FAIL.md`); scitrera issue not yet filed.

---

## Verified scope

Same Blackwell GB10 hardware (NVIDIA DGX Spark, sm121, ARM64,
driver 580.95.05), same reproducer (below), four different PyTorch
builds:

| PyTorch build | torch version | CUDA | sm121 EFFICIENT correct? |
|---|---|---|---|
| `xomoxcc/comfyui:sm121` (our overlay over scitrera-2.10 base) | 2.10.0 | 13.1 | **NO — broken** |
| `scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131` (bare) | 2.10.0 | 13.1 | **NO — broken (byte-identical to overlay)** |
| `xomoxcc/dgx-spark-pytorch-dev:2.11.0-v1-cu132` (our build of scitrera's 2.11 recipe) | 2.11.0 | 13.2 | **NO — broken (byte-identical to scitrera 2.10)** |
| `nvcr.io/nvidia/pytorch:25.12-py3` | 2.10.0a0+nv25.12 | 13.1 | **YES — correct** |
| `nvcr.io/nvidia/pytorch:26.02-py3` | 2.11.0a0+nv26.02 | 13.1 | **YES — correct** |

The load-bearing comparison is the trio **scitrera-2.10/cu131** vs
**xomoxcc-2.11/cu132 (built from scitrera's recipe)** vs
**NGC-25.12/26.02**: scitrera-pipeline outputs are byte-identical
broken across a major-version PyTorch bump (2.10 → 2.11) **and** a
CUDA major bump (13.1 → 13.2); NVIDIA's own builds at the same major
versions are correct on the same hardware.

That bit-identical persistence across torch/CUDA major bumps rules
out:

- **PyTorch 2.10 source** as the culprit (the bug is also present on
  2.11, so the source line itself isn't where the regression lives).
- **CUDA 13.1** as the culprit (same numbers under CUDA 13.2).
- **Our xformers/sage/ComfyUI overlays** as the culprit (Run-A and
  Run-B byte-identical, with overlay vs without).

What stays constant across both broken images is the scitrera build
pipeline itself: same `Dockerfile.base`, same `pytorch_builder` target,
same `TORCH_CUDA_ARCH_LIST=12.1a`, same `NVCC_GENCODE` for sm_121,
same build host. The defect is therefore in the
`scitrera/cuda-containers` build pipeline (most likely a third-party
submodule pin, a local patch applied during the recipe, a compiler
flag, or a stale binary cache that survives image bumps).

## TL;DR

`torch.nn.functional.scaled_dot_product_attention(q, k, v)` running
on `scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131` against an sm121
GPU dispatches to the `EFFICIENT_ATTENTION` backend (CUTLASS-family
memory-efficient FMHA), which executes successfully but returns
output that is unrelated to the inputs:

- **Norm** of the output tensor is 12×–27× off from a CPU `MATH`
  reference for the same `q,k,v`.
- **Component-wise drift** (`max|Δ|`) is 5×–250× the natural standard
  deviation of the output, far above any plausible fp16/bf16
  rounding-noise threshold.
- **No NaN, no Inf, no exception, no warning.** Every downstream
  numerical guard sees plausible finite tensors and continues running.

The same code path on `MATH` and (where dispatchable) `FLASH_ATTENTION`
backends produces bit-near-correct output on the same hardware in the
same PyTorch process — only `EFFICIENT_ATTENTION` is broken.

This was discovered in the wild via ComfyUI text-to-image workflows
(SDXL + Flux): every transformer text encoder
(`comfy.sd1_clip.SDClipModel`-derived) calls SDPA with `small_input=True`
and lands on `attention_pytorch` →
`torch.nn.functional.scaled_dot_product_attention` → `EFFICIENT`.
Resulting embeddings are random noise that still passes NaN-checks,
so SDXL and Flux render visually-plausible but prompt-unrelated
images. See [`COMFYUI_PROMPT_FAIL.md`](COMFYUI_PROMPT_FAIL.md) for
the discovery trail.

While the EFFICIENT path is exercised, PyTorch's CUTLASS dispatcher
emits the diagnostic line

```
FATAL: kernel `fmha_cutlassF_*_sm80` is for sm80-sm100, but was built for sm121
```

dozens of times per call (the kernel-table-probe loop in
`aten::_efficient_attention_forward`). The kernel that ultimately
runs after the probe loop is the one returning garbage.

## Reproducer (pure PyTorch, no third-party deps)

Saves to `/tmp/sdpa_sm121_repro.py`. Compares MATH /
EFFICIENT_ATTENTION / FLASH_ATTENTION against a CPU fp32 reference
for shapes that mirror common transformer use (CLIP-L 77×12×64,
SDXL self-attention bottleneck 4096×8×64).

```python
"""sdpa_sm121_repro.py — minimal SDPA correctness check on sm121."""
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.manual_seed(0)
print(f"torch={torch.__version__}  cuda={torch.version.cuda}")
print(f"device={torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")
print()

shapes = [
    (1, 12,   77, 64, torch.float32),   # CLIP-L-ish, fp32
    (1, 12,   77, 64, torch.bfloat16),  # CLIP-L-ish, bf16
    (1, 12,   77, 64, torch.float16),   # CLIP-L-ish, fp16
    (1,  8, 4096, 64, torch.bfloat16),  # SDXL self-attention bottleneck
]

for B, H, N, D, dt in shapes:
    print(f"=== B={B} H={H} N={N} D={D} dtype={str(dt).split('.')[-1]} ===")
    q = torch.randn(B, H, N, D, device="cuda", dtype=dt)
    k = torch.randn(B, H, N, D, device="cuda", dtype=dt)
    v = torch.randn(B, H, N, D, device="cuda", dtype=dt)

    ref = torch.nn.functional.scaled_dot_product_attention(
        q.cpu().float(), k.cpu().float(), v.cpu().float()
    )

    print(f"  reference (CPU fp32): norm={ref.norm():.3f}  std={ref.std():.4f}")
    print(f"  {'backend':<22} {'norm':>10} {'max|Δ|':>10} {'mean|Δ|':>10} {'has_nan':>8}")
    for backend in [SDPBackend.MATH,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.FLASH_ATTENTION]:
        try:
            with sdpa_kernel([backend]):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            of = out.cpu().float()
            d = (of - ref).abs()
            has_nan = bool(torch.isnan(out).any().item())
            print(f"  {backend.name:<22} {of.norm().item():>10.3f}"
                  f" {d.max().item():>10.4f} {d.mean().item():>10.4f}"
                  f" {str(has_nan):>8}")
        except Exception as e:
            print(f"  {backend.name:<22} ERROR: {type(e).__name__}: {str(e)[:80]}")
    print()
```

Run inside a one-off pod against scitrera's image:

```bash
kubectl run sdpa-test --restart=Never \
    --image=scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131 \
    --overrides='{"spec":{"nodeSelector":{"<sm121-node-selector>":"true"},
                  "containers":[{"name":"sdpa-test",
                  "image":"scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131",
                  "command":["tail","-f","/dev/null"],
                  "resources":{"limits":{"nvidia.com/gpu":"1"}}}]}}'
kubectl cp /tmp/sdpa_sm121_repro.py sdpa-test:/tmp/
kubectl exec sdpa-test -- python3 -u /tmp/sdpa_sm121_repro.py
kubectl delete pod sdpa-test
```

## Observed output on `scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131`

```
torch=2.10.0  cuda=13.1
device=NVIDIA GB10  cap=(12, 1)

=== B=1 H=12 N=77 D=64 dtype=float32 ===
  reference (CPU fp32): norm=43.900  std=0.1805
  backend                      norm     max|Δ|    mean|Δ|  has_nan
  MATH                       43.900     0.0000     0.0000    False
  EFFICIENT_ATTENTION        85.923     1.8189     0.3161    False
  FLASH_ATTENTION    ERROR: RuntimeError: No available kernel. Aborting execution.

=== B=1 H=12 N=77 D=64 dtype=bfloat16 ===
  reference (CPU fp32): norm=43.584  std=0.1792
  backend                      norm     max|Δ|    mean|Δ|  has_nan
  MATH                       43.584     0.0033     0.0002    False
  EFFICIENT_ATTENTION         3.506     1.1487     0.1418    False
  FLASH_ATTENTION            43.584     0.0033     0.0003    False

=== B=1 H=12 N=77 D=64 dtype=float16 ===
  reference (CPU fp32): norm=43.676  std=0.1796
  backend                      norm     max|Δ|    mean|Δ|  has_nan
  MATH                       43.676     0.0005     0.0000    False
  EFFICIENT_ATTENTION       250.800     2.5419     0.7931    False
  FLASH_ATTENTION            43.676     0.0005     0.0000    False

=== B=1 H=8 N=4096 D=64 dtype=bfloat16 ===
  reference (CPU fp32): norm=37.586  std=0.0260
  backend                      norm     max|Δ|    mean|Δ|  has_nan
  MATH                       37.592     0.0005     0.0000    False
  EFFICIENT_ATTENTION      1023.564     4.8858     0.4092    False
  FLASH_ATTENTION            37.591     0.0005     0.0000    False
```

Magnitude of the breakage:

| Shape × dtype | Reference norm | EFFICIENT norm | Ratio |
|---|---|---|---|
| 77 × fp32  | 43.900 | 85.923   | **1.96×** |
| 77 × bf16  | 43.584 |  3.506   | **0.080×** (12× too small) |
| 77 × fp16  | 43.676 | 250.800  | **5.74×** |
| 4096 × bf16 | 37.586 | 1023.564 | **27.2×** |

`max|Δ|` of 1.1–4.9 against an output `std` of 0.02–0.18 is 5×–250×
the natural variation — well above anything attributable to fp16/bf16
rounding. `has_nan=False` everywhere is the worrying part: the
output is finite, so every downstream check passes, while the values
are unrelated to the inputs.

## Expected output (from NGC PyTorch 25.12 on the same hardware)

```
torch=2.10.0a0+b4e4ee81d3.nv25.12  cuda=13.1
device=NVIDIA GB10  cap=(12, 1)

=== B=1 H=12 N=77 D=64 dtype=float32 ===
  reference (CPU fp32): norm=43.900  std=0.1805
  backend                      norm     max|Δ|    mean|Δ|  has_nan
  MATH                       43.900     0.0000     0.0000    False
  EFFICIENT_ATTENTION        43.900     0.0000     0.0000    False
  FLASH_ATTENTION    ERROR: RuntimeError: No available kernel. Aborting execution.

=== B=1 H=12 N=77 D=64 dtype=bfloat16 ===
  reference (CPU fp32): norm=43.584  std=0.1792
  backend                      norm     max|Δ|    mean|Δ|  has_nan
  MATH                       43.584     0.0033     0.0002    False
  EFFICIENT_ATTENTION        43.584     0.0033     0.0003    False
  FLASH_ATTENTION            43.584     0.0033     0.0003    False

=== B=1 H=12 N=77 D=64 dtype=float16 ===
  reference (CPU fp32): norm=43.676  std=0.1796
  backend                      norm     max|Δ|    mean|Δ|  has_nan
  MATH                       43.676     0.0005     0.0000    False
  EFFICIENT_ATTENTION        43.676     0.0005     0.0000    False
  FLASH_ATTENTION            43.676     0.0005     0.0000    False

=== B=1 H=8 N=4096 D=64 dtype=bfloat16 ===
  reference (CPU fp32): norm=37.586  std=0.0260
  backend                      norm     max|Δ|    mean|Δ|  has_nan
  MATH                       37.591     0.0005     0.0000    False
  EFFICIENT_ATTENTION        37.592     0.0005     0.0000    False
  FLASH_ATTENTION            37.591     0.0005     0.0000    False
```

EFFICIENT, MATH, and FLASH all agree with each other and with the CPU
reference within fp16/bf16 rounding tolerance — i.e. the SDPA kernels
on sm121 *can* run correctly when built by NVIDIA's pipeline, on the
same major PyTorch version.

## Cross-validation summary table

| Shape × dtype | CPU ref | scitrera 2.10/cu131 | xomoxcc 2.11/cu132 (scitrera-pipeline) | NGC 25.12 (torch 2.10a) | NGC 26.02 (torch 2.11a) |
|---|---|---|---|---|---|
| 77 × fp32   | 43.900 | **85.923 ✗**   | **85.923 ✗**   | 43.900 ✓ | 43.900 ✓ |
| 77 × bf16   | 43.584 | **3.506 ✗**    | **3.506 ✗**    | 43.584 ✓ | 43.584 ✓ |
| 77 × fp16   | 43.676 | **250.800 ✗**  | **250.800 ✗**  | 43.676 ✓ | 43.676 ✓ |
| 4096 × bf16 | 37.586 | **1023.564 ✗** | **1023.564 ✗** | 37.592 ✓ | 37.592 ✓ |

(EFFICIENT_ATTENTION norms only; full per-backend matrices in the
output blocks above.) The two scitrera-pipeline columns are byte-
identical despite a PyTorch major bump (2.10 → 2.11) and a CUDA major
bump (13.1 → 13.2) — confirming the defect persists across the
version axis and lives in the build pipeline rather than in
PyTorch/CUDA source.

## Environment

```
hardware:        NVIDIA GB10 (Blackwell, sm121, GB10 SoC on ASUS Ascent GX10 / DGX Spark)
host kernel:     Linux 6.19.x ARM64
host distro:     Ubuntu 24.04 ARM64
nvidia driver:   580.95.05
container CUDA:  13.1 (matches NGC 25.12 / 26.02 — both correct)
torch:           2.10.0  (scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131)
torch.git_version: 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0
                   (note: identical between scitrera bare base and our
                   xomoxcc/comfyui:sm121 overlay, confirming the wheel
                   inside our overlay is unmodified scitrera output)
```

## Suggested investigation for scitrera maintainers

The bug is in **what scitrera's build pipeline produces**, not in
the PyTorch 2.10 source it builds from (Run-C / NGC 25.12 establishes
that the same major version source line works on the same CUDA major
on the same hardware when built by NVIDIA's pipeline). Worth checking:

1. **Bisect existing tags.** Test `dgx-spark-pytorch-dev` tags
   prior to `2.10.0-v2-cu131` and any newer ones (e.g.
   `2.11.0-v1-cu132`) on the same reproducer. The first failing tag
   identifies which build-recipe change introduced the regression.
2. **Diff the build recipe** at the failing tag against the prior
   working one. Areas most likely to matter:
   - `TORCH_CUDA_ARCH_LIST` / `CMAKE_CUDA_ARCHITECTURES` for sm121
     specifically.
   - PyTorch submodule pins (`third_party/cutlass`,
     `third_party/flash-attention`, `third_party/composable_kernel`).
   - Any local patches applied between `git clone pytorch` and `pip
     install`.
   - CUDA toolkit minor version (cu131 vs cu13.x flavours).
   - Compile-time defines that affect FMHA dispatch
     (`USE_FLASH_ATTENTION`, `USE_MEM_EFF_ATTENTION`,
     `XFORMERS_*`-style flags if any leaked in).
3. **Compare to NGC 25.12's build setup.** NVIDIA publishes the
   recipe-equivalent in the [NGC PyTorch container release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/).
   Any sm121-specific cherry-pick they apply that scitrera doesn't is
   a strong candidate.
4. **Check binary cache / wheel reuse.** If scitrera's build uses a
   pip wheel cache or a pre-built CUTLASS object cache, a stale
   binary from an earlier (mis-built) state could persist across
   tag bumps even when the source-level recipe is correct. A
   no-cache rebuild from clean state may resolve it without any
   recipe change.

The user-side workaround (next section) is in production for our
ComfyUI deployment but is an unsatisfactory long-term answer for a
general PyTorch image.

## User-side workaround

For users who cannot wait for a fixed scitrera image, scope the SDPA
backend to MATH for the affected forward(s):

```python
from torch.nn.attention import sdpa_kernel, SDPBackend

with sdpa_kernel([SDPBackend.MATH]):
    out = model(...)
```

For text encoders (77-token-class sequences) MATH's O(n²) cost is
single-digit milliseconds and effectively free. For longer
diffusion-style attention prefer `[SDPBackend.FLASH_ATTENTION,
SDPBackend.MATH]`, since FLASH is correct on this build for
fp16/bf16; the dispatcher will pick FLASH where dtype/shape allows
and fall back to MATH otherwise.

We use this pattern in ComfyUI by wrapping
`comfy.sd1_clip.SDClipModel.forward` (the base class for every
ComfyUI text encoder) via a `sitecustomize.py` shim — see
[`roles/k8s_dgx/templates/comfyui_launch.sh.j2`](roles/k8s_dgx/templates/comfyui_launch.sh.j2)
§4c. Verified to fully restore SDXL and Flux text-to-image
correctness on this image with no measurable performance penalty.

## Discovery context

We discovered the bug while debugging ComfyUI text-to-image: outputs
on our sm121 cluster differed reproducibly per prompt but bore no
semantic relation to the prompt text (a "red apple" prompt rendered
an interior loft scene; "blue cube" rendered a tiki statue; Flux
rendered hand-written-card-with-pseudo-English images). Bisection
ruled out, in order:

- SageAttention as the cause (removing `--use-sage-attention` only
  changed *which* garbage came back).
- A bad RealVisXL checkpoint (sha256 matched a known-good local
  copy).
- An SDXL- or CLIP-specific bug (Flux-schnell with T5-XXL +
  SentencePiece tokeniser fails identically — different model
  family, different encoder, same failure mode).
- A NaN-or-Inf-eats-everything bug (no NaN at any layer).
- An xformers / sage / ComfyUI-overlay artefact (Run-A vs Run-B
  byte-identical numbers).
- An "all of PyTorch on sm121" bug (NGC 25.12 with the same major
  version is correct).

What's left is the scitrera-specific build artefact reported here.

## Cross-references inside this repository

| Path | Role |
|---|---|
| `COMFYUI_PROMPT_FAIL.md` | End-to-end discovery, per-phase diagnosis, fix verification |
| `COMFYUI_SM121_PATCHES.md` | Earlier xformers-side sm121 fixes (cutlass self-reject + FA3 disable) — separate concern, complementary |
| `roles/k8s_dgx/templates/comfyui_launch.sh.j2` §4c | Production workaround: sitecustomize shim wrapping `SDClipModel.forward` with `sdpa_kernel([MATH])` |

## Once filed

Add the scitrera/cuda-containers issue URL here, and link from
`COMFYUI_PROMPT_FAIL.md` § Future Maintenance. Once scitrera ships a
fixed image, re-run the reproducer above; if EFFICIENT now matches
the reference, the §4c workaround in `comfyui_launch.sh.j2` can be
removed.
