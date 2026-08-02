#!/usr/bin/env bash
#
# verify_sglang_image.sh — image acceptance gate for xomoxcc/dgx-spark-sglang:*
#
# Runs the dgxarley runtime patch set exactly the way sglang_launch.sh runs it
# (roles/k8s_dgx/files/sglang_patches/p[0-9][0-9]_*.py, filename order), then
# asks the two questions the patch run itself CANNOT answer:
#
#   1. Did any patch report ANCHOR-DRIFT?
#      -> an anchor moved; the fix silently does not happen any more.
#   2. Does SGLang's model registry still import cleanly?
#      -> THE check that matters. On 2026-07-28 p30 generated a module importing
#         sglang.srt.layers.quantization.fp8_kernel, which RFC #29630 moved in
#         v0.5.16. The patch run said "ok". The registry swallowed the
#         ImportError ("Ignore import error when loading ...") and 31 model
#         classes were silently disabled: deepseek_v2, deepseek_v4, glm4_moe,
#         kimi_*, mistral_large_3, pixtral, ... Nothing crashed, nothing warned
#         at the top level, and the models simply were not there any more.
#
# "Applies cleanly" is not "works". Run this before promoting any image, and
# after every change under sglang_patches/.
#
# No GPU and no k3s needed: plain podman, CPU only, ~1 minute.
#
# Usage:
#   scripts/verify_sglang_image.sh <image> [podman-connection]
#
#   scripts/verify_sglang_image.sh xomoxcc/dgx-spark-sglang:0.5.16-sm121
#   scripts/verify_sglang_image.sh xomoxcc/dgx-spark-sglang:0.5.16-sm121 spark5
#
# With a connection name the checks run on that remote podman host (the images
# live in spark5's store), otherwise locally.
#
# Exit codes: 0 = clean, 1 = drift or registry damage, 2 = usage/plumbing error.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_SRC="${REPO_ROOT}/roles/k8s_dgx/files/sglang_patches"

IMAGE="${1:-}"
CONNECTION="${2:-}"
[[ -n "${IMAGE}" ]] || { sed -n '2,40p' "$0"; exit 2; }
[[ -d "${PATCH_SRC}" ]] || { echo "ERROR: patch dir not found: ${PATCH_SRC}" >&2; exit 2; }

# Scenarios mirror the model-name gates in the patch set (gate_model()), so the
# model-specific patches are exercised too, not just the ungated majority.
SCENARIOS=(
    "bare::"
    "glm:SGLANG_MODEL=zai-org/GLM-5.2-NVFP4:SGLANG_DSA_INDEXER_TRITON=1"
    "hy3:SGLANG_MODEL=tencent/Hy3-NVFP4:SGLANG_HUNYUAN_TOKEN_SUFFIX=1"
)

run_podman() {
    if [[ -n "${CONNECTION}" ]]; then
        podman --connection "${CONNECTION}" "$@"
    else
        podman "$@"
    fi
}

# The patch set has to be inside the podman host's filesystem. With a remote
# connection, ship it over first (tar over ssh, same host the connection names).
REMOTE_PATCHES="/tmp/dgxarley-verify-patches"
if [[ -n "${CONNECTION}" ]]; then
    ssh_host="$(podman system connection list --format '{{.Name}} {{.URI}}' \
        | awk -v c="${CONNECTION}" '$1==c {print $2}' \
        | sed -E 's#^ssh://([^/]+)/.*#\1#; s#:[0-9]+$##')"
    [[ -n "${ssh_host}" ]] || { echo "ERROR: cannot resolve ssh host for podman connection '${CONNECTION}'" >&2; exit 2; }
    tar -C "${PATCH_SRC}/.." -cf - "$(basename "${PATCH_SRC}")" \
        | ssh -o BatchMode=yes "${ssh_host}" \
            "rm -rf ${REMOTE_PATCHES} && mkdir -p ${REMOTE_PATCHES} && tar -C /tmp -xf - \
             && cp /tmp/$(basename "${PATCH_SRC}")/*.py ${REMOTE_PATCHES}/ \
             && rm -rf ${REMOTE_PATCHES}/__pycache__"
    PATCH_MOUNT="${REMOTE_PATCHES}"
else
    PATCH_MOUNT="${PATCH_SRC}"
fi

echo "=== verifying ${IMAGE}${CONNECTION:+ (on ${CONNECTION})}"
failures=0

# BASELINE: which model classes does the UNPATCHED image already fail to import?
# Upstream ships modules with optional dependencies (bailing_moe_* import vllm,
# which we do not install), so a raw count would fail every image forever. Only
# the DELTA introduced by our patch set is a defect, so measure it as a delta.
registry_failures() { # stdin: registry output -> stdout: sorted module names
    grep 'Ignore import error' | sed -E 's/.*loading (sglang[^:]*):.*/\1/' | sort -u
}
baseline="$(run_podman run --rm "${IMAGE}" python3 -c 'import sglang.srt.models.registry' 2>&1 \
    | registry_failures || true)"
if [[ -n "${baseline}" ]]; then
    echo "  note: $(wc -l <<< "${baseline}") model class(es) already fail to import WITHOUT our patches"
    sed 's/^/        /' <<< "${baseline}"
    echo "        (upstream optional deps — excluded from the check below)"
fi

for scenario in "${SCENARIOS[@]}"; do
    IFS=':' read -r label env1 env2 <<< "${scenario}"
    env_args=()
    [[ -n "${env1}" ]] && env_args+=(-e "${env1}")
    [[ -n "${env2}" ]] && env_args+=(-e "${env2}")

    echo
    echo "--- scenario: ${label}"
    out="$(run_podman run --rm "${env_args[@]}" -v "${PATCH_MOUNT}:/patches:ro" "${IMAGE}" bash -c '
        for p in /patches/p[0-9][0-9]_*.py; do python3 "$p" 2>&1; done
        echo "###REGISTRY###"
        python3 -c "import sglang.srt.models.registry" 2>&1
    ' 2>/dev/null)" || true

    patch_phase="${out%%###REGISTRY###*}"
    registry_phase="${out#*###REGISTRY###}"

    drift="$(grep -c 'ANCHOR-DRIFT' <<< "${patch_phase}" || true)"

    if [[ "${drift}" -gt 0 ]]; then
        echo "  FAIL  ${drift} ANCHOR-DRIFT:"
        grep 'ANCHOR-DRIFT' <<< "${patch_phase}" | sed 's/^/        /'
        failures=$((failures + 1))
    else
        echo "  ok    0 ANCHOR-DRIFT"
    fi

    patched_failures="$(registry_failures <<< "${registry_phase}" || true)"
    regressions="$(comm -13 <(echo "${baseline}") <(echo "${patched_failures}") || true)"

    if [[ -n "${regressions}" ]]; then
        echo "  FAIL  $(wc -l <<< "${regressions}") model class(es) broken BY OUR PATCHES:"
        sed 's/^/        /' <<< "${regressions}"
        echo "        (a generated/patched module is unimportable — these are GONE at runtime)"
        failures=$((failures + 1))
    else
        echo "  ok    model registry: no regression vs the unpatched image"
    fi
done

echo
if [[ "${failures}" -eq 0 ]]; then
    echo "RESULT: clean — ${IMAGE} passes the patch-set acceptance gate"
    exit 0
fi
echo "RESULT: ${failures} failing check(s) — do NOT promote ${IMAGE}"
exit 1
