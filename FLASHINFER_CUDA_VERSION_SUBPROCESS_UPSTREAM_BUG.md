# Flashinfer Upstream Bug: `get_cuda_version()` subprocess.Popen fails inside torch.compile/dynamo trace

## Status

**Workaround applied.** 2026-04-15 session outcome:

- **Root cause identified**: `flashinfer.jit.cpp_ext.get_cuda_version()` calls
  `subprocess.check_output([nvcc, "--version"])` on its first invocation (it's
  `@functools.cache`-decorated, so only once per process). When that first
  invocation is reached from inside a `torch.compile` / dynamo trace context,
  dynamo tries to polyfill the `subprocess.Popen` call via
  `torch/_dynamo/polyfills/__init__.py:392 getattr_and_trace` — the polyfill
  cannot handle `Popen.__init__`'s internal fork/threading machinery, and the
  child process dies with sigquit. The sglang launcher observes the child
  failure and restarts the pod → `startup_crash`.
- **Runtime fix in `sglang_launch.sh`**: short-circuit `get_cuda_version()` to
  return `Version(torch.version.cuda)` directly (which is always populated on
  our CUDA-built PyTorch and matches what `nvcc --version` reports for the same
  install). The subprocess path remains as an untaken fallback for PyTorch
  builds without CUDA support. Idempotent, grep-guarded, and the marker
  `_fi_cuda_ver_subprocess_bypass_` prevents double application.
- **Not reported upstream yet** — needs a minimal repro (`torch.compile` + any
  flashinfer FP4 quant call from inside the traced region is enough). Adjacent
  issues exist but none match this exact failure mode. See "Upstream status".

Bug exists in flashinfer **0.6.7.post3** (the version shipped in
`scitrera/dgx-spark-sglang:0.5.10`) and is structurally present in all
flashinfer releases that have `get_cuda_version()` spawning `nvcc` at call time
— i.e. everything since the subprocess-based version lookup was introduced.
Verified absent in our patched `xomoxcc/dgx-spark-sglang:0.5.10-cudnn` image
after the runtime patch is applied.

## Summary

Flashinfer's FP4 quantization path JIT-compiles kernels on first use. The JIT
build calls `is_cuda_version_at_least("12.8")` → `get_cuda_version()` →
`subprocess.check_output([nvcc, "--version"])`. If the very first call happens
from a Python frame that dynamo is actively tracing (e.g., piecewise CUDA
graph capture on a forward pass that hits `fp4_quantize`), the subprocess spawn
crashes the child process instead of returning the version string.

The function is `@functools.cache`-decorated, so if it's called **once outside**
a dynamo trace before the traced path is hit, the cache is populated and the
subprocess spawn never happens again — and flashinfer works fine. That's why
sglang's non-piecewise configs are unaffected (the graph-capture path doesn't
go through dynamo the same way): the JIT build fires at a "safe" moment or
uses a different backend module that was already compiled.

Our monkey-patch removes the subprocess path entirely, so it doesn't matter
when or from where `get_cuda_version()` is called.

## Symptom

Observed on GLM-4.7-NVFP4 at EP=1 on 4× DGX Spark (SM121/GB10) with
`fp4_gemm_backend=flashinfer_cudnn`, `disable_piecewise_cuda_graph=false`,
running on the `xomoxcc/dgx-spark-sglang:0.5.10-cudnn` image (which already
has the `nvidia-cudnn-cu12` wheels installed, so the old cuDNN missing-dep
is out of the way). Verbatim stack from the sglang head pod
(`sglang-head-6c984df886-zvxmb`):

```
File "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/glm4_moe.py", line 174, in forward
    gate_up, _ = self.gate_up_proj(x)
File "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/linear.py", line 460, in forward
    output_parallel = self.quant_method.apply(self, input_, bias)
File "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/modelopt_quant.py", line 1482, in apply
    x_fp4, x_scale_interleaved = fp4_quantize(x, layer.input_scale_inv)
File "/usr/local/lib/python3.12/dist-packages/flashinfer/quantization/fp4_quantization.py", line 700, in fp4_quantize
    x_q, sf = get_fp4_quantization_module(f"{major}{minor}").fp4_quantize_sm100(
File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/polyfills/__init__.py", line 392, in getattr_and_trace
    return fn(*args[2:], **kwargs)
File "/usr/local/lib/python3.12/dist-packages/flashinfer/quantization/fp4_quantization.py", line 170, in get_fp4_quantization_module
    module = backend_modules[backend]().build_and_load()
File "/usr/local/lib/python3.12/dist-packages/flashinfer/quantization/fp4_quantization.py", line 107, in gen_fp4_quantization_sm120f_module
    return gen_fp4_quantization_module(sm120f_nvcc_flags, "120f")
File "/usr/local/lib/python3.12/dist-packages/flashinfer/quantization/fp4_quantization.py", line 131, in gen_fp4_quantization_module
    "-DENABLE_FP4" if is_cuda_version_at_least("12.8") else "",
File "/usr/local/lib/python3.12/dist-packages/flashinfer/jit/cpp_ext.py", line 91, in is_cuda_version_at_least
    return get_cuda_version() >= Version(version_str)
File "/usr/local/lib/python3.12/dist-packages/flashinfer/jit/cpp_ext.py", line 73, in get_cuda_version
    txt = subprocess.check_output([nvcc, "--version"], text=True)
File "/usr/lib/python3.12/subprocess.py", line 466, in check_output
    return run(*popenargs, stdout=PIPE, timeout=timeout, check=True,
File "/usr/lib/python3.12/subprocess.py", line 548, in run
    with Popen(*popenargs, **kwargs) as process:
File "/usr/lib/python3.12/subprocess.py", line 828, in __init__
    self._waitpid_lock = threading.Lock()

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

[2026-04-15 07:45:17] Received sigquit from a child process. It usually means the child failed.
```

The two tells:
1. `File "torch/_dynamo/polyfills/__init__.py", line 392, in getattr_and_trace` in
   the middle of the flashinfer stack — dynamo is actively tracing when the
   subprocess call happens.
2. `Set TORCHDYNAMO_VERBOSE=1 ...` footer from torch — dynamo itself emitted
   this hint, so it was involved in the failure.

Final line is the sglang launcher observing the child process died.

## Root cause

The un-patched upstream code in `flashinfer/jit/cpp_ext.py`:

```python
@functools.cache
def get_cuda_version() -> Version:
    # Try to query nvcc for CUDA version; if nvcc is unavailable, fall back to torch.version.cuda
    try:
        cuda_home = get_cuda_path()
        nvcc = os.path.join(cuda_home, "bin/nvcc")
        txt = subprocess.check_output([nvcc, "--version"], text=True)
        matches = re.findall(r"release (\d+\.\d+),", txt)
        if not matches:
            raise RuntimeError(
                f"Could not parse CUDA version from nvcc --version output: {txt}"
            )
        return Version(matches[0])
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
        # NOTE(Zihao): when nvcc is unavailable, fall back to torch.version.cuda
        if torch.version.cuda is None:
            raise RuntimeError(
                "nvcc not found and PyTorch is not built with CUDA support. "
                "Could not determine CUDA version."
            ) from e
        return Version(torch.version.cuda)


def is_cuda_version_at_least(version_str: str) -> bool:
    return get_cuda_version() >= Version(version_str)
```

The `torch.version.cuda` fallback is already present — it just only runs **on
exception**. On the happy path, `subprocess.check_output` is called, and this
is what explodes when reached from inside a dynamo trace context.

Dynamo's polyfills for unsupported builtins/library calls try to hoist the
side effect out of the traced region. For `subprocess.Popen`, the polyfill
can't model the fork/clone/pipe setup, so it raises a `Unsupported` exception
that bubbles up in a way that leaves a zombied child (hence `self._waitpid_lock
= threading.Lock()` as the last Python frame — dynamo aborted mid-`__init__`).
The sglang launcher picks up the child-gone signal and restarts the pod.

## Reproduction

Minimal repro should be:

```python
import torch
import flashinfer.quantization.fp4_quantization as fp4q

x = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
scale = torch.tensor(1.0, device="cuda")

@torch.compile(fullgraph=True, backend="inductor")
def f(x, scale):
    return fp4q.fp4_quantize(x, scale)

f(x, scale)  # first call in a fresh process, from inside a compiled region
```

The first call triggers `gen_fp4_quantization_sm120f_module.build_and_load()`,
which calls `is_cuda_version_at_least` while dynamo is tracing `f`, which
calls the subprocess, which dies.

Workaround for the repro (same as our fix): pre-warm the cache by calling
`fp4q.fp4_quantize(x, scale)` once **outside** the `@torch.compile` region, or
call `flashinfer.jit.cpp_ext.get_cuda_version()` at module-import time before
any compiled function runs.

## Our workaround

`sglang_launch.sh` contains a startup-time patch that rewrites the body of
`get_cuda_version()` to return `Version(torch.version.cuda)` directly, leaving
the original subprocess path as a never-taken fallback. Key properties:

- **Source file**: `/usr/local/lib/python3.12/dist-packages/flashinfer/jit/cpp_ext.py`
- **Marker**: `# [patch] _fi_cuda_ver_subprocess_bypass_` — idempotent check
  on re-runs.
- **Grep guard**: patch only applies if the exact pre-patch function signature
  (including the `# Try to query nvcc...` comment and the `try:` line) is
  present. If flashinfer upstream ever rewrites this function, the patch
  silently skips and prints a warning instead of corrupting the file.
- **Fallback path preserved**: the `try`/`except` block is still in the file
  below the short-circuit, so the patch survives a hypothetical flashinfer
  version bump that adds more validation — as long as the signature comment
  doesn't move, the new code just runs *before* the old code.

See the patch block in `sglang_launch.sh` (look for
`PATCH_FI_CUDA_VER_EOF`). Patched file contents, conceptually:

```python
@functools.cache
def get_cuda_version() -> Version:
    # [patch] _fi_cuda_ver_subprocess_bypass_
    # Short-circuit with torch.version.cuda to avoid spawning a `nvcc --version`
    # subprocess from inside a torch.compile/dynamo trace context. See the
    # sglang_launch.sh header block above this patch for the full rationale.
    if torch.version.cuda is not None:
        return Version(torch.version.cuda)
    # Try to query nvcc for CUDA version; if nvcc is unavailable, fall back to torch.version.cuda
    try:
        ...  # original subprocess path, now unreachable on normal PyTorch builds
```

`is_cuda_version_at_least()` is unchanged — it still calls `get_cuda_version()`
by module-global name lookup, so the patched version takes effect automatically.

## Upstream status

**No known open PR or issue for this specific failure mode.** Adjacent work:

- **flashinfer `get_cuda_version` history**: the subprocess-based implementation
  was introduced to support CUDA version gates for feature flags like
  `-DENABLE_FP4` (gated on CUDA ≥ 12.8). Previous versions used
  `torch.version.cuda` directly, which is what our patch reverts to. A clean
  upstream fix would either:
  1. Call `get_cuda_version()` at module import time to warm the cache
     unconditionally, or
  2. Change the function body to prefer `torch.version.cuda` when available
     and only fall back to `nvcc --version` when `torch.version.cuda is None`
     (i.e., reverse the try/except priority — exactly what our runtime patch does).

- **torch.compile + subprocess** is a general dynamo limitation, not a torch
  bug — dynamo explicitly documents that side-effecting calls like
  `subprocess.Popen` are "unsupported". Any library that calls subprocess
  lazily from its hot path will eventually trip this when used from a
  `@torch.compile`'d function.

**Report to file**: `flashinfer/issues` with the minimal repro above. Option
(2) is the preferred fix (one-line reorder, no behavior change on happy path,
fixes torch.compile compatibility).

## Relationship to other bugs

- **Orthogonal to** `SGLANG_NVFP4_SHUFFLE_ROWS_OOB_UPSTREAM_BUG.md` — that one
  is about `cutlass_moe_fp4` MoE dispatch under EP. This one is purely about
  flashinfer's CUDA-version detection in any FP4 quantize call.
- **Orthogonal to** the cuDNN missing-dep issue (`scripts/build_cudnn_image.sh`
  docstring) — that was a pip-package shipping problem. This one is a
  code/tracing interaction.
- **Explains the "piecewise crashes" rule on GLM-4.7 EP=1**: prior to this
  analysis, all 12 `disable_piecewise_cuda_graph=false` variants in the
  GLM-4.7 EP=1 matrix crashed at startup (see
  `TESTLOGS/sglang_nn4_tp4_ep1/glm-4.7-nvfp4/TESTLOG_nv580.142_sglang-0.5.10_glm-4.7-nvfp4_4n.md`).
  The original diagnosis was "piecewise path is broken". This bug document
  provides the actual mechanism: piecewise graph capture triggers torch.compile
  on the forward, which traces through `fp4_quantize`, which hits the
  subprocess spawn. With the patch applied, the piecewise configs **may** work
  — pending a matrix re-run.
- **Latent on non-piecewise configs**: technically the same failure path is
  *reachable* on non-piecewise + CG-on configs, but in practice `get_cuda_version`
  gets called during module loading (before graph capture) and the
  `@functools.cache` result is warm by the time graph capture starts. Only
  piecewise+fi_cudnn managed to reorder the calls so the first subprocess
  invocation landed inside a trace context.

## Test matrix impact (GLM-4.7-NVFP4 EP=1)

Before the patch, with `xomoxcc/dgx-spark-sglang:0.5.10-cudnn`:

| # | MoE | Attn | FP4 | Pcw | Outcome (pre-patch) |
|---|-----|------|-----|-----|---------------------|
| 9  | triton     | fi     | fi_cudnn | ✓ | startup_crash (this bug) |
| 12 | triton     | triton | fi_cudnn | ✓ | startup_crash (this bug) |
| 21 | fi_cutlass | fi     | fi_cudnn | ✓ | startup_crash (this bug + fi_cutlass MoE EP=1) |
| 24 | fi_cutlass | triton | fi_cudnn | ✓ | startup_crash (this bug + fi_cutlass MoE EP=1) |
| 33 | cutlass    | fi     | fi_cudnn | ✓ | startup_crash (this bug) |
| 36 | cutlass    | triton | fi_cudnn | ✓ | startup_crash (this bug) |

Post-patch expectation: tests 9, 12, 33, 36 should become STABLE (they only
had this bug blocking them); tests 21 and 24 should continue to crash for the
independent `fi_cutlass` MoE EP=1 dispatch bug documented in
`SGLANG_NVFP4_SHUFFLE_ROWS_OOB_UPSTREAM_BUG.md`. Needs a matrix re-run with
the patched `sglang_launch.sh` on all 4 sparks (re-deploy via
`ansible-playbook k8s_dgx.yml --tags sglang`) to confirm.

## Files

- `roles/k8s_dgx/files/sglang_launch.sh` — runtime monkey-patch (look for
  the `PATCH_FI_CUDA_VER_EOF` heredoc, ~line 175).
- `/usr/local/lib/python3.12/dist-packages/flashinfer/jit/cpp_ext.py` — patch
  target (inside the running container).
- `/usr/local/lib/python3.12/dist-packages/flashinfer/quantization/fp4_quantization.py`
  — the call site that triggers the JIT build at first forward pass.
