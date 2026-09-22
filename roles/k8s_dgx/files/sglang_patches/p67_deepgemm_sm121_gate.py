"""[dgxarley] Backport of upstream sgl-project/sglang#39482 ("[Bugfix] Include
SM121 in DeepGEMM packed-scale selection", merged 3d1b9e7549a3): widen the two
places `deep_gemm_wrapper/configurer.py` special-cases `sm_version == 120` to
`sm_version in (120, 121)`, so GB10 (SM121) gets the same DeepGEMM
capability-probe and UE8M0 packed-scale selection as SM120 instead of silently
falling through unchecked / staying on the non-UE8M0 scale path.

WHAT THE UNPATCHED CODE DOES ON SM121 (v0.5.20)
  `_compute_enable_deep_gemm()`'s `if sm_version == 120:` probe block is
  skipped entirely for sm_version 121 (121 != 120), so `ENABLE_JIT_DEEPGEMM`
  can end up True on GB10 without ever calling the SM120/SM121 capability
  probe (`m_grouped_fp8_fp4_gemm_nt_contiguous`) that the surrounding comment
  says exists precisely because installed DeepGEMM builds may predate it.
  Independently, `DEEPGEMM_SCALE_UE8M0 = ENABLE_JIT_DEEPGEMM and
  (get_platform().is_sm100 or get_device_sm() == 120)` also excludes 121, so
  even when DeepGEMM is enabled on SM121 it never selects the UE8M0 packed
  scale format the SM120/SM121 DeepGEMM kernels need.

WHY GATED OFF BY DEFAULT
  Both edits change DeepGEMM kernel selection on GB10, which this cluster has
  a history of needing careful, measured validation around (see
  reference_glm52_dsa_indexer_deepgemm_sm121.md / reference_triton_moe_config_
  cache.md in project memory). Opt in per instance via env
  SGLANG_OPT_DEEPGEMM_SM121_PACKED_SCALE=1 once validated; unset/"0" leaves
  SGLang exactly as upstream v0.5.20 ships it.

ANCHOR SCOPE -- v0.5.20-era shape only (upstream #29927), verified 2026-09-22
against git objects (no working-tree checkout):
  * v0.5.19: `_compute_enable_deep_gemm()`'s SM120 block reads
    `# DeepGEMM requires TMEM/tcgen05 (SM100+datacenter), not available on
    SM120\n    if sm_version == 120:\n        return False` (a hard
    blanket-disable, no probe/try-except) and `DEEPGEMM_SCALE_UE8M0 =
    DEEPGEMM_BLACKWELL` (a plain alias, not the `ENABLE_JIT_DEEPGEMM and
    (...)` expression at all). Neither v0.5.20 anchor exists in that shape.
    Widening the v0.5.19 blanket disable would only turn "DeepGEMM off on
    SM120" into "DeepGEMM off on SM120 and SM121" -- pointless, since the
    goal is to ENABLE the probe/UE8M0 path, not extend a hard veto. So this
    patch deliberately does NOT special-case v0.5.19: both edits report
    ANCHOR-DRIFT there (a correct "not applicable, re-check on rebase"
    signal under the _patchlib contract, not a crash) and the image runs
    unpatched, same as if this file didn't exist.
  * v0.5.20: both OLD anchors below match byte-exactly (diffed against
    `git show v0.5.20:...configurer.py`).
  * upstream/main (as of 2026-09-22): a LATER refactor beyond #39482 already
    landed -- `_compute_enable_deep_gemm()` was rewritten around a new
    `_sm120_deep_gemm_apis_available()` helper that probes three DeepGEMM
    APIs (`fp8_einsum`, `m_grouped_fp8_fp4_gemm_nt_contiguous`,
    `transform_sf_into_required_layout`) and reads
    `if sm_version in (120, 121) and not _sm120_deep_gemm_apis_available():`,
    so EDIT 1's exact anchor (`if sm_version == 120:` with the #39482
    comment) no longer exists in ANY form there (main has moved past this
    patch's scope, not regressed -- re-check `_sm120_deep_gemm_apis_available()`
    before rebasing onto a main-tracking image). EDIT 2's own anchor
    (`DEEPGEMM_SCALE_UE8M0 = ENABLE_JIT_DEEPGEMM and (get_platform().is_sm100
    or get_device_sm() == 120)`) IS unchanged on main and would self-report
    "already applied" in isolation -- but both edits run in ONE Patch.run()
    call (deliberately, per _patchlib's all-or-nothing-per-file contract), so
    EDIT 1's AnchorDrift exception aborts `apply()` before EDIT 2's
    `p.replace()` is ever reached. Verified empirically 2026-09-22: main
    reports exactly one ANCHOR-DRIFT line and writes nothing. That is the
    safe outcome (never half-patch a file), just not a distinct "EDIT 2
    already applied" line -- re-check EDIT 1's anchor first if main-tracking.

DELETE WHEN the pinned image ships a configurer.py at or past #39482 (i.e.
`get_device_sm() in (120, 121)` for the UE8M0 line and the SM121-aware probe
in `_compute_enable_deep_gemm`) natively -- check both edits independently,
since upstream may ship them at different times (as main currently does for
EDIT 2 but not EDIT 1's exact anchor).
"""

from _patchlib import Patch, gate_env

TARGET = "sglang/srt/layers/deep_gemm_wrapper/configurer.py"

GATE_ENV = "SGLANG_OPT_DEEPGEMM_SM121_PACKED_SCALE"

patch = Patch(
    name="DeepGEMM SM121 packed-scale selection (#39482)",
    target=TARGET,
    when=gate_env(GATE_ENV, "1"),
)

OLD_PROBE_GATE = """    # SM120 support (mma.sync block-scale, no TMEM) landed in DeepGEMM#324;
    # probe the entry point since installed builds may predate it.
    if sm_version == 120:"""
NEW_PROBE_GATE = """    # SM120/SM121 support (including GB10) landed in DeepGEMM#324;
    # probe the entry point since installed builds may predate it.
    if sm_version in (120, 121):"""

OLD_SCALE_UE8M0 = """DEEPGEMM_SCALE_UE8M0 = ENABLE_JIT_DEEPGEMM and (
    get_platform().is_sm100 or get_device_sm() == 120
)"""
NEW_SCALE_UE8M0 = """DEEPGEMM_SCALE_UE8M0 = ENABLE_JIT_DEEPGEMM and (
    get_platform().is_sm100 or get_device_sm() in (120, 121)
)"""


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD_PROBE_GATE, NEW_PROBE_GATE, what="_compute_enable_deep_gemm SM121 probe-gate widening")
    p.replace(OLD_SCALE_UE8M0, NEW_SCALE_UE8M0, what="DEEPGEMM_SCALE_UE8M0 SM121 widening")
