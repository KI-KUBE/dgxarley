"""[dgxarley] Shared helpers for the SGLang runtime source patches.

Every `p<NN>_*.py` next to this file is a standalone patch against the SGLang
install in the container's dist-packages. `sglang_launch.sh` runs them all, in
filename order, before starting the server. This module holds the boilerplate
that used to be copy-pasted into every `python3 - <<'PATCH_*_EOF'` heredoc:
target resolution, the already-applied guard, the anchor-drift reporting and
the write-back.

Contract every patch relies on:

* **Never raise, never exit non-zero.** A drifted anchor is a warning, not a
  crash: the launcher runs under `set -e` and an exception here would crashloop
  the pod. Patches degrade to "unpatched SGLang", which is the same behaviour
  the inline heredocs had.
* **Already-applied is checked FIRST**, before the anchor. `new` frequently
  contains `old` as a prefix (we mostly append to an anchor), so an
  `old in code` check would re-apply on a re-run. That exact bug bit the
  buffered-safetensors patch on 2026-07-16.
* **All-or-nothing per file.** Edits are buffered in memory and written once at
  the end; if any edit in the patch drifts, nothing is written. A file
  half-patched by a partially-drifted multi-edit patch is far worse to debug
  than an unpatched one.
* **Idempotent.** Running a patch twice must not change the file the second
  time. The runner is not transactional, and a pod restart re-runs everything.

Adding a patch: copy the shape of `p20_moe_wna16_qzeros_ep.py`. The module
docstring carries the knowledge (why it exists, upstream status, when it can be
deleted); do not add a patch without one.
"""

import os
from collections.abc import Callable

# The SGLang install inside the container image. Single source of truth: patches
# name their target relative to this, so an image that moves dist-packages needs
# one edit here rather than 30.
DIST_PACKAGES = "/usr/local/lib/python3.12/dist-packages"


class AnchorDrift(Exception):
    """An anchor no longer matches the shipped SGLang source.

    Raised by the edit helpers, caught by `Patch.run`, which turns it into an
    ANCHOR-DRIFT line and skips the write. Patches should not catch it.
    """


def gate_model(*needles: str) -> bool:
    """True when SGLANG_MODEL contains any of `needles` (the model-name gate)."""
    model = os.environ.get("SGLANG_MODEL", "")
    return any(needle in model for needle in needles)


def gate_env(name: str, value: str) -> bool:
    """True when env var `name` is exactly `value`."""
    return os.environ.get(name, "") == value


def target_contains(target: str, needle: str) -> bool:
    """True when the target file exists AND contains `needle`.

    The gate for patches whose subject matter was DELETED upstream rather than
    moved. Without it, "upstream removed the buggy function" is indistinguishable
    from "the anchor drifted and someone must rebase it": both surface as one
    ANCHOR-DRIFT line, so the drift report stops being a work list.

    Use it as `when=target_contains(<same target>, <marker of the buggy code>)`.
    A False gate logs one "gate not matched" line, which is the honest statement
    that this patch has nothing to do on this image.

    Missing/unreadable file counts as False: nothing to patch there either.
    """
    path = os.path.join(DIST_PACKAGES, target)
    try:
        with open(path) as fh:
            return needle in fh.read()
    except OSError:
        return False


def is_kernels_namespace() -> bool:
    """True on SGLang >= v0.5.16, i.e. after the RFC #29630 namespace migration.

    A coarse but reliable version discriminator: v0.5.16 moved the kernel ops out
    of `srt/layers/...` into the new `sglang/kernels/ops/` tree. Prefer a precise
    `target_contains()` probe where one exists; use this when a patch is
    deliberately NOT maintained for the new layout, so the log says "gate not
    matched" (a decision) instead of ANCHOR-DRIFT (a work item).
    """
    return os.path.isdir(os.path.join(DIST_PACKAGES, "sglang/kernels/ops"))


def write_module(path: str, source: str, what: str) -> bool:
    """Write a generated Python module and PROVE it imports.

    p30/p35 do not edit SGLang, they generate new modules into its tree. Nothing
    ever checked that the generated source is importable, and on 2026-07-28 that
    cost us: p30 emitted an import of `sglang.srt.layers.quantization.fp8_kernel`,
    which RFC #29630 moved in v0.5.16. The patch reported success, the module was
    unimportable, and the ImportError cascaded through SGLang's model registry —
    31 model classes silently disabled (deepseek_v2/v4, glm4_moe, kimi_*, ...),
    visible only as "Ignore import error when loading ..." far from the cause.

    Returns True when the module was written and imports. On failure the file is
    REMOVED again (a missing module degrades to "unpatched SGLang", an
    unimportable one poisons the registry) and one warning line is printed.
    Never raises: same contract as the rest of this module.
    """
    import importlib.util

    try:
        with open(path, "w") as fh:
            fh.write(source)
    except OSError as exc:
        print(f"ANCHOR-DRIFT: {os.path.basename(path)}: {what} could not be written ({exc})")
        return False

    try:
        spec = importlib.util.spec_from_file_location(f"_dgxarley_probe_{os.path.basename(path)[:-3]}", path)
        if spec is None or spec.loader is None:
            raise ImportError("no import spec")
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    except Exception as exc:  # noqa: BLE001 - any import-time failure must degrade, not crash
        try:
            os.remove(path)
        except OSError:
            pass
        print(
            f"ANCHOR-DRIFT: {os.path.basename(path)}: {what} written but NOT IMPORTABLE "
            f"({type(exc).__name__}: {exc}) - file removed so it cannot poison the model registry"
        )
        return False

    print(f"Wrote {os.path.basename(path)}: {what}")
    return True


class Patch:
    """One patch against one SGLang source file.

    `target` is relative to DIST_PACKAGES. `when` is the gate: False means the
    patch does not apply to this model/config and is skipped with one log line
    (this replaces the bash `if` that used to wrap the heredoc, so gate and
    patch now live in the same file).

    `alt_targets` lists further paths the same code may live under, tried in
    order after `target`; the first one that EXISTS wins. SGLang moves files
    between releases (RFC #29630 moved model_runner_kv_cache_mixin.py's content
    to model_executor/pool_configurator.py in v0.5.16), and the patch set is
    shipped as ONE ConfigMap to instances that may pin DIFFERENT images. So a
    patch cannot simply follow the move: it has to hit whichever layout the
    image in front of it happens to have.
    """

    def __init__(self, name: str, target: str, when: bool = True, alt_targets: tuple[str, ...] = ()) -> None:
        self.name = name
        self.when = when
        for candidate in (target, *alt_targets):
            if os.path.isfile(os.path.join(DIST_PACKAGES, candidate)):
                target = candidate
                break
        self.target = target
        self.path = os.path.join(DIST_PACKAGES, target)
        self.basename = os.path.basename(target)
        self._code = ""
        self._changed = False

    def replace(self, old: str, new: str, marker: str | None = None, what: str | None = None) -> None:
        """Replace the first occurrence of `old` with `new`.

        `marker` is the already-applied probe; it defaults to `new`, which is
        correct whenever `new` is unique to the patched state. Pass an explicit
        marker when `new` is not a reliable probe: when two patches inject the
        same string, or when the injected text is not unique in the file. Both
        cases have burned us, hence the parameter.
        """
        label = what or self.name
        probe = marker if marker is not None else new
        if probe in self._code:
            return
        if old not in self._code:
            raise AnchorDrift(f"{label} anchor missing")
        self._code = self._code.replace(old, new, 1)
        self._changed = True

    def replace_all(self, old: str, new: str, marker: str | None = None, what: str | None = None) -> None:
        """Like replace(), but replaces EVERY occurrence, not just the first.

        Use this whenever the anchor legitimately appears more than once and all
        of them must change (e.g. the same guard expression in two code paths).
        Getting this wrong is silent: replace() would patch the first hit, report
        success, and leave the rest untouched. That is exactly what happened on
        the qwen3_5 attn-quant (2 hits) and kv-scale (4 hits) conversions on
        2026-07-16 -- the patch logged "Patched" while the file was still half
        original. If in doubt, count the hits in the real image before choosing.
        """
        label = what or self.name
        probe = marker if marker is not None else new
        if probe in self._code:
            return
        if old not in self._code:
            raise AnchorDrift(f"{label} anchor missing")
        self._code = self._code.replace(old, new)
        self._changed = True

    def replace_any(
        self,
        variants: "list[tuple[str, str]]",
        marker: str,
        what: str | None = None,
    ) -> None:
        """Apply the FIRST (old, new) pair whose `old` matches; drift only if none do.

        For the recurring case where upstream kept the logic but changed its
        spelling between the versions we serve: a method became a free function
        (so `self.server_args.` lost its `self.`, and the block dedented), or an
        accessor was renamed (`get_global_server_args()` -> `get_server_args()`).

        One ConfigMap feeds instances that pin different images, so "rebase to the
        new spelling" is not an option: both must keep working. Order the variants
        oldest-first only for readability, the probe is exact-match either way.

        `marker` is mandatory: with several possible `new` texts there is no
        sensible default already-applied probe.
        """
        label = what or self.name
        if marker in self._code:
            return
        for old, new in variants:
            if old in self._code:
                self._code = self._code.replace(old, new, 1)
                self._changed = True
                return
        raise AnchorDrift(f"{label} anchor missing (tried {len(variants)} spellings)")

    def prepend(self, text: str, marker: str) -> None:
        """Prepend `text` at the top of the file (for import lines with no anchor)."""
        if marker in self._code:
            return
        self._code = text + self._code
        self._changed = True

    @property
    def code(self) -> str:
        """The in-memory buffer. Assign to it for edits the helpers cannot express."""
        return self._code

    @code.setter
    def code(self, value: str) -> None:
        if value != self._code:
            self._code = value
            self._changed = True

    @property
    def changed(self) -> bool:
        """True once an edit has modified the buffer (i.e. a write will happen)."""
        return self._changed

    def insert_after(self, anchor: str, text: str, marker: str, what: str | None = None) -> None:
        """Insert `text` right after the first occurrence of `anchor`.

        `marker` is mandatory here: the injected text is appended to the anchor,
        so it is never a safe default probe on its own.
        """
        self.replace(anchor, anchor + text, marker=marker, what=what)

    def run(self, fn: Callable[["Patch"], None]) -> Callable[["Patch"], None]:
        """Decorator: run `fn` against this patch's file and write back.

        Used as `@patch.run` on the patch body, so the module reads
        declaratively top to bottom and the file is executed on import as a
        script. Returns `fn` unchanged so the decorated name stays callable
        (handy in tests).
        """
        if not self.when:
            print(f"[patch] {self.name}: gate not matched, skipping")
            return fn
        if not os.path.isfile(self.path):
            print(f"ANCHOR-DRIFT: {self.basename}: {self.name} target file missing (SGLang restructured/renamed?)")
            return fn
        with open(self.path) as fh:
            self._code = fh.read()
        try:
            fn(self)
        except AnchorDrift as exc:
            print(f"ANCHOR-DRIFT: {self.basename}: {exc} (SGLang version drift; re-check anchor)")
            return fn
        if not self._changed:
            print(f"[patch] {self.basename}: {self.name} already applied, skipping")
            return fn
        with open(self.path, "w") as fh:
            fh.write(self._code)
        print(f"Patched {self.basename}: {self.name}")
        return fn
