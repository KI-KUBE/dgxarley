"""[dgxarley] flashinfer/jit/cpp_ext.py: get_cuda_version() avoids subprocess from
inside a torch.compile/dynamo trace.

Symptom (GLM-4.7-NVFP4 EP=1, piecewise CUDA graphs + fi_cudnn FP4):
  flashinfer/quantization/fp4_quantization.py:170 build_and_load()
  -> gen_fp4_quantization_sm120f_module
  -> flashinfer/jit/cpp_ext.py:91 is_cuda_version_at_least("12.8")
  -> cpp_ext.py:73 subprocess.check_output([nvcc, "--version"])
  -> subprocess/threading.Lock() under torch/_dynamo/polyfills:392 getattr_and_trace
  -> child process sigquit -> pod restart -> startup_crash

Why: the JIT build is triggered on the first forward pass, which for piecewise
CUDA graphs happens inside a torch.compile trace. Dynamo can't polyfill
subprocess.Popen (it does fork/threading internals), so the call blows up even
though nvcc is present and works fine from a normal shell.

Fix: short-circuit get_cuda_version() with torch.version.cuda, which is always
available at import time and matches what nvcc reports for the same install.
The original function already had this as a fallback path on exception -- we
just promote it to run first. This keeps the subprocess path as the fallback
for pytorch builds without a CUDA version (none of ours).

This patches flashinfer, NOT sglang -- target is `flashinfer/jit/cpp_ext.py`
under DIST_PACKAGES (flashinfer is a separate top-level package in the same
dist-packages tree, not a subpackage of sglang).

No model gate, no env gate: unconditional, same as the original heredoc.
"""

from _patchlib import Patch

patch = Patch(
    name="get_cuda_version(): short-circuit via torch.version.cuda (no subprocess in dynamo trace)",
    target="flashinfer/jit/cpp_ext.py",
)

MARKER = "# [patch] _fi_cuda_ver_subprocess_bypass_"

OLD = (
    "@functools.cache\n"
    "def get_cuda_version() -> Version:\n"
    "    # Try to query nvcc for CUDA version; if nvcc is unavailable, "
    "fall back to torch.version.cuda\n"
    "    try:"
)
NEW = (
    "@functools.cache\n"
    "def get_cuda_version() -> Version:\n"
    "    " + MARKER + "\n"
    "    # Short-circuit with torch.version.cuda to avoid spawning a `nvcc --version`\n"
    "    # subprocess from inside a torch.compile/dynamo trace context. See the\n"
    "    # sglang_launch.sh header block above this patch for the full rationale.\n"
    "    if torch.version.cuda is not None:\n"
    "        return Version(torch.version.cuda)\n"
    "    # Try to query nvcc for CUDA version; if nvcc is unavailable, "
    "fall back to torch.version.cuda\n"
    "    try:"
)


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD, NEW, marker=MARKER, what="get_cuda_version subprocess bypass")
