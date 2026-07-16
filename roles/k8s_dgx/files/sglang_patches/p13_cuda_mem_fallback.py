"""[dgxarley] utils/common.py: _cuda_mem_fallback proc-meminfo tier (GLM-5 only).

GLM-5 specific: transformers upgrade + mem_get_info patch.
Only needed for glm_moe_dsa models -- skip for MiniMax, Qwen, etc. (the
transformers>=5.3.0 / huggingface_hub>=1.3.0 pip upgrade this patch depends on
stays inline in sglang_launch.sh -- it is a dependency bump, not a source
patch, and is gated on the same `SGLANG_MODEL == *GLM-5*` condition via
gate_model("GLM-5") below.)

Patch _cuda_mem_fallback: transformers 5.x + huggingface_hub >=1.3.0
triggers a CUDA context init during import that breaks torch.cuda.mem_get_info()
on GB10 (cudaErrorMemoryAllocation). nvidia-smi also can't report memory on GB10.
Fix: fall back to /proc/meminfo (GB10 unified memory = system RAM).

RE-ANCHORED 2026-07-16: upstream refactored the old inline "torch.cuda.mem_get_info()
failed -> raise RuntimeError" branch of get_nvgpu_memory_capacity() into a standalone
helper function ALSO named _cuda_mem_fallback() (name collision with our marker, not
our patch -- it's upstream's own tier-1 nvidia-smi->mem_get_info() fallback, called from
3 sites). Our tier-2 (mem_get_info() ALSO fails -> /proc/meminfo) now anchors inside
that function's except block. NOTE: in practice this tier is not exercised on this
cluster (mem_get_info() succeeds -- see live logs: "Falling back to
torch.cuda.mem_get_info(). Reported total GPU memory per device (MiB): [124546]"), but
kept as defense-in-depth for the driver-stack edge case the comment above describes.
Also fixed here: the old anchor had no dedicated "already applied" marker in the
outer bash gate (it grepped the function NAME, which now collides with upstream's own
function) -- re-running against an already-patched file would silently re-match part of
its own injected code and could double-patch on every pod restart. Now gated on the
marker string itself.

[moved 2026-07-16] Was an inline `python3 << 'PATCH_MEM_FALLBACK_EOF'` heredoc wrapped
in bash-native file-exists / already-patched / anchor grep checks; those bash checks
are now redundant with what `Patch.run` / `Patch.replace` already do (file-missing,
already-applied marker, anchor-drift), so only the GLM-5 model gate survives, as
`when=gate_model("GLM-5")` below.
"""

from _patchlib import Patch, gate_model

patch = Patch(
    name="_cuda_mem_fallback proc-meminfo tier",
    target="sglang/srt/utils/common.py",
    when=gate_model("GLM-5"),
)

MARKER = "# [patch] _sgl_cuda_mem_fallback_proc_meminfo_"

OLD = """    except (RuntimeError, ValueError, OSError) as e:
        raise RuntimeError(
            f"{reason} torch.cuda.mem_get_info() fallback also failed: {e}"
        ) from e"""

NEW = (
    """    except (RuntimeError, ValueError, OSError) as e:
        """
    + MARKER
    + """ -- GB10 unified memory: try /proc/meminfo as a last resort
        # before giving up (rare case: transformers/huggingface_hub break
        # torch.cuda.mem_get_info() during import, cudaErrorMemoryAllocation).
        try:
            with open("/proc/meminfo") as _mf:
                _mem_mib = None
                for _line in _mf:
                    if _line.startswith("MemTotal:"):
                        _mem_mib = int(_line.split()[1]) // 1024  # kB -> MiB
                        break
            if _mem_mib is not None:
                logger.warning(
                    f"{reason} torch.cuda.mem_get_info() fallback also failed: {e}. "
                    f"Falling back to /proc/meminfo MemTotal: {_mem_mib} MiB."
                )
                return _mem_mib
        except OSError:
            pass
        raise RuntimeError(
            f"{reason} torch.cuda.mem_get_info() fallback also failed: {e}"
        ) from e"""
)


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD, NEW, marker=MARKER, what="_cuda_mem_fallback proc-meminfo tier")
