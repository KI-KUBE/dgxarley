#!/bin/bash
set -e

apt-get update -qq && apt-get install -y -qq tini net-tools iputils-ping iproute2 curl >/dev/null 2>&1

if [ -n "$RSYNC_TARGETS" ]; then
  apt-get install -y -qq rsync openssh-client >/dev/null 2>&1
fi

# huggingface_hub is NOT guaranteed to be in the image any more. The Job used to
# run the SGLang serving image (hub baked in at the recipe's HF_HUB_MIN_VERSION),
# but those builds are arm64-only while the Job now runs on the amd64 master, so
# hf_preload_image defaults to a plain multi-arch python. Install only when the
# image does not already satisfy HF_HUB_MIN_VERSION, which keeps this a no-op if
# the Job is ever pointed back at a serving image.
hub_ok() {
  python3 - "$HF_HUB_MIN_VERSION" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

floor = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    have = version("huggingface_hub")
except PackageNotFoundError:
    sys.exit(1)
if not floor:
    sys.exit(0)


def parts(v: str) -> tuple:
    # Compare on the numeric release prefix only; a trailing rc/dev suffix on an
    # otherwise-new-enough version must not read as "too old". Zero-padded to
    # three components so "1.26" does not sort below "1.26.0".
    out = []
    for chunk in v.split(".")[:3]:
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        out.append(int(digits) if digits else 0)
    return tuple(out + [0] * (3 - len(out)))


sys.exit(0 if parts(have) >= parts(floor) else 1)
PY
}

if ! hub_ok 2>/dev/null; then
  spec="huggingface_hub[hf_xet]"
  [ -n "$HF_HUB_MIN_VERSION" ] && spec="huggingface_hub[hf_xet]>=$HF_HUB_MIN_VERSION"
  echo "[preload] installing $spec"
  # hf_xet is required because hf_hub_disable_xet is back to 0 (the hex-hash
  # crash is fixed from hub 1.26.0 on, see FIXED_UPSTREAM_HF_XET_BUG.md).
  # --break-system-packages only for Debian/Ubuntu system pythons that mark
  # themselves externally managed (PEP 668); the official python images do not.
  pip3 install --no-cache-dir -q "$spec" \
    || pip3 install --no-cache-dir -q --break-system-packages "$spec"
fi

exec tini -- python3 /scripts/download_models.py
