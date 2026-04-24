#!/usr/bin/env bash
#
# build_comfyui_image.sh — Build xomoxcc/comfyui:sm121.
#
# Builds a ComfyUI image pre-baked for DGX Spark (GB10 / SM_121 / ARM64) on
# top of scitrera/dgx-spark-pytorch-dev. The image ships a frozen ComfyUI
# checkout + all pip deps + xformers + SageAttention v2 compiled for SM_121
# — no git clone / pip install at container start.
#
# Structure mirrors build_sm121_image.sh:
#   - entire script runs on the x86 control host
#   - spark4 (arm64) is a dumb remote podman build runner
#   - built image is streamed back via `podman save | podman load`
#   - push to Docker Hub from x86 using the x86 host's credentials
#
# Build context lives in scripts/comfyui/:
#   - Dockerfile
#   - entrypoint.sh
#   - requirements-extra.txt
#
# Prerequisites on the x86 control host
# --------------------------------------
# - podman (`apt install podman`)
# - Unencrypted SSH key for podman (podman's Go SSH client does not use
#   ssh-agent or encrypted keys):
#     ssh-keygen -t ed25519 -f ~/.ssh/id_podman -N ""
#     ssh-copy-id -i ~/.ssh/id_podman root@spark4
#   Override path via BUILD_COMFYUI_SSH_IDENTITY.
# - `podman login docker.io -u xomoxcc` already done on this host.
# - ~25 GB free disk for the image after scp.
#
# Prerequisites on spark4 (the build host)
# ----------------------------------------
# - podman (`apt install podman`)
# - podman.socket enabled as root: `systemctl enable --now podman.socket`
# - ~60 GB free disk (base + intermediate layers + final image).
# - NO credentials, NO local scripts, NO state between runs (except the
#   local podman image store + layer cache, which accelerate rebuilds).
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_DIR="${SCRIPT_DIR}/comfyui"

IMAGE_TAG="${BUILD_COMFYUI_IMAGE_TAG:-xomoxcc/comfyui:sm121}"
IMAGE_TAG_DATED="${IMAGE_TAG}-$(date +%Y%m%d)"

# Upstream ComfyUI ref to bake into the image. `master` = latest green;
# pin a specific commit SHA here for reproducible builds.
COMFYUI_REF="${BUILD_COMFYUI_REF:-master}"

# Base image aliases. --base <value> or BUILD_COMFYUI_BASE_IMAGE env var
# override these. The 'scitrera' base is the default — xomoxcc is optional
# and only works if the custom 2.11/cu132 base has already been built on
# spark4 (scripts/build_pytorch_base_image.sh) or pushed to Docker Hub.
BASE_SCITRERA_IMAGE="scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131"
BASE_XOMOXCC_IMAGE="xomoxcc/dgx-spark-pytorch-dev:2.11.0-v1-cu132"
BASE_IMAGE_ALIAS=""
BASE_IMAGE_OVERRIDE="${BUILD_COMFYUI_BASE_IMAGE:-}"
EFFECTIVE_BASE_IMAGE=""
BASE_IMAGE_SOURCE=""

# Remote build host + podman connection. Same pattern as build_sm121_image.sh.
REMOTE_HOST="${BUILD_COMFYUI_REMOTE_HOST:-root@spark4.local}"
PODMAN_CONNECTION="${BUILD_COMFYUI_PODMAN_CONNECTION:-}"
PODMAN_SSH_IDENTITY="${BUILD_COMFYUI_SSH_IDENTITY:-${HOME}/.ssh/id_podman}"

# Parallel compile jobs on the build host. GB10 safe ceiling is 8 (16 OOM-
# kills CUTLASS template expansion — see feedback_build_jobs_gb10). Honored
# via the MAX_JOBS env baked into the Dockerfile plus --build-arg overrides.
BUILD_JOBS="${BUILD_COMFYUI_BUILD_JOBS:-8}"

# Optional kernel builds. Both default ON and roughly double end-to-end build
# time. Turn off for quick iterations on the ComfyUI layer itself.
BUILD_XFORMERS=1
BUILD_SAGE_ATTN=1

PUSH_IMAGE=1
NO_LOCAL_COPY=0

# ============================================================================
# Helpers
# ============================================================================

log()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [--base xomoxcc|scitrera|<image>]
                        [--comfyui-ref REF]
                        [--remote-host user@host] [--podman-connection NAME]
                        [--no-xformers] [--no-sage-attn]
                        [--no-local-copy] [--no-push] [--help]

Builds ${IMAGE_TAG} on the remote build host via podman socket, copies
the result back to this host (unless --no-local-copy), and pushes it
from here (unless --no-push or --no-local-copy).

Options:
  --base VALUE  PyTorch dev base image this build sits on:
                  scitrera  ${BASE_SCITRERA_IMAGE}  (default; published upstream)
                  xomoxcc   ${BASE_XOMOXCC_IMAGE}   (custom 2.11/cu132 — must
                                                     already exist on build
                                                     host or Docker Hub)
                  <image>   arbitrary image reference, passed verbatim.
  --comfyui-ref REF
                Git ref (branch, tag, commit SHA) of comfyanonymous/ComfyUI
                to freeze into the image. Default: ${COMFYUI_REF}
  --remote-host user@host
                Remote arm64 build host reachable via SSH + podman socket.
                Default: ${REMOTE_HOST}
  --podman-connection NAME
                Registered podman connection name (or created on demand). If
                omitted, derived from --remote-host (strip user@ and domain).
  --no-xformers   Skip compiling xformers from source (saves ~15 min).
  --no-sage-attn  Skip compiling SageAttention v2 from source (saves ~10 min).
  --no-local-copy Skip streaming the built image back to this host. Implies
                  --no-push (push reads from the local podman store).
  --no-push       Skip 'podman push' after build + local copy.
  --help          Show this help.

Environment overrides:
  BUILD_COMFYUI_IMAGE_TAG           Default: ${IMAGE_TAG}
  BUILD_COMFYUI_REF                 Default: ${COMFYUI_REF}
  BUILD_COMFYUI_BASE_IMAGE          Direct BASE override, wins over --base.
  BUILD_COMFYUI_REMOTE_HOST         Default: ${REMOTE_HOST}
  BUILD_COMFYUI_PODMAN_CONNECTION   Derived from --remote-host if unset.
  BUILD_COMFYUI_SSH_IDENTITY        Default: ${PODMAN_SSH_IDENTITY}
  BUILD_COMFYUI_BUILD_JOBS          Default: ${BUILD_JOBS}
EOF
}

# ============================================================================
# Argument parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)
            shift
            [[ $# -gt 0 ]] || die "--base requires an argument (xomoxcc|scitrera|<image>)"
            BASE_IMAGE_ALIAS="$1"; shift ;;
        --base=*)
            BASE_IMAGE_ALIAS="${1#--base=}"; shift ;;
        --comfyui-ref)
            shift
            [[ $# -gt 0 ]] || die "--comfyui-ref requires a git ref"
            COMFYUI_REF="$1"; shift ;;
        --comfyui-ref=*)
            COMFYUI_REF="${1#--comfyui-ref=}"; shift ;;
        --remote-host)
            shift
            [[ $# -gt 0 ]] || die "--remote-host requires an argument (user@host)"
            REMOTE_HOST="$1"; shift ;;
        --remote-host=*)
            REMOTE_HOST="${1#--remote-host=}"; shift ;;
        --podman-connection)
            shift
            [[ $# -gt 0 ]] || die "--podman-connection requires an argument"
            PODMAN_CONNECTION="$1"; shift ;;
        --podman-connection=*)
            PODMAN_CONNECTION="${1#--podman-connection=}"; shift ;;
        --no-xformers)   BUILD_XFORMERS=0; shift ;;
        --no-sage-attn)  BUILD_SAGE_ATTN=0; shift ;;
        --no-local-copy) NO_LOCAL_COPY=1; PUSH_IMAGE=0; shift ;;
        --no-push)       PUSH_IMAGE=0; shift ;;
        --help|-h)       usage; exit 0 ;;
        *)               die "Unknown argument: $1 (use --help)" ;;
    esac
done

if [[ -z "${PODMAN_CONNECTION}" ]]; then
    PODMAN_CONNECTION="${REMOTE_HOST##*@}"
    PODMAN_CONNECTION="${PODMAN_CONNECTION%%.*}"
fi

# ============================================================================
# Base image resolution
# ============================================================================

resolve_base_image() {
    if [[ -n "${EFFECTIVE_BASE_IMAGE}" ]]; then
        return 0
    fi
    if [[ -n "${BASE_IMAGE_OVERRIDE}" ]]; then
        EFFECTIVE_BASE_IMAGE="${BASE_IMAGE_OVERRIDE}"
        BASE_IMAGE_SOURCE="BUILD_COMFYUI_BASE_IMAGE env"
        return 0
    fi
    case "${BASE_IMAGE_ALIAS}" in
        xomoxcc)  EFFECTIVE_BASE_IMAGE="${BASE_XOMOXCC_IMAGE}";  BASE_IMAGE_SOURCE="--base xomoxcc" ;;
        scitrera) EFFECTIVE_BASE_IMAGE="${BASE_SCITRERA_IMAGE}"; BASE_IMAGE_SOURCE="--base scitrera" ;;
        "")       EFFECTIVE_BASE_IMAGE="${BASE_SCITRERA_IMAGE}"; BASE_IMAGE_SOURCE="default (scitrera)" ;;
        *)        EFFECTIVE_BASE_IMAGE="${BASE_IMAGE_ALIAS}";    BASE_IMAGE_SOURCE="--base (verbatim)" ;;
    esac
}

# ============================================================================
# Preflight
# ============================================================================

preflight() {
    log "Preflight"

    for f in Dockerfile entrypoint.sh requirements-extra.txt; do
        [[ -f "${CONTEXT_DIR}/${f}" ]] || die "Missing build context file: ${CONTEXT_DIR}/${f}"
    done

    for tool in git podman; do
        command -v "${tool}" >/dev/null || die "Required tool not found: ${tool}"
    done

    if [[ ! -f "${PODMAN_SSH_IDENTITY}" ]]; then
        cat >&2 <<EOF

ERROR: SSH identity '${PODMAN_SSH_IDENTITY}' not found.

Podman's Go SSH client cannot use the ssh-agent or encrypted keys, so a
dedicated unencrypted key is required. Create it with:

  ssh-keygen -t ed25519 -f ${PODMAN_SSH_IDENTITY} -N ""
  ssh-copy-id -i ${PODMAN_SSH_IDENTITY} ${REMOTE_HOST}

Then re-run this script.
EOF
        exit 1
    fi

    echo "Build context present, tools available, SSH identity found"
}

# ============================================================================
# Podman connection to the remote build host
# ============================================================================

ensure_podman_connection() {
    log "Ensuring podman connection '${PODMAN_CONNECTION}' → ${REMOTE_HOST}"

    if podman system connection list --format '{{.Name}}' | grep -qxF "${PODMAN_CONNECTION}"; then
        echo "Connection '${PODMAN_CONNECTION}' already registered"
    else
        echo "Registering new podman connection..."
        local remote_uid
        remote_uid="$(ssh -i "${PODMAN_SSH_IDENTITY}" -o BatchMode=yes -o ConnectTimeout=5 \
            "${REMOTE_HOST}" id -u 2>/dev/null)" \
            || die "SSH to ${REMOTE_HOST} failed — verify the key is authorized"

        local sock_path
        if [[ "${remote_uid}" == "0" ]]; then
            sock_path="/run/podman/podman.sock"
        else
            sock_path="/run/user/${remote_uid}/podman/podman.sock"
        fi

        podman system connection add "${PODMAN_CONNECTION}" \
            "ssh://${REMOTE_HOST}${sock_path}" \
            --identity "${PODMAN_SSH_IDENTITY}" \
            || die "Failed to register podman connection '${PODMAN_CONNECTION}'"
    fi

    echo "Validating connection..."
    if ! podman --connection "${PODMAN_CONNECTION}" info >/dev/null 2>&1; then
        cat >&2 <<EOF

ERROR: Podman connection '${PODMAN_CONNECTION}' is not responding.

Check on ${REMOTE_HOST}:
  systemctl status podman.socket
  systemctl enable --now podman.socket

And that the socket path matches what this script registered:
  podman system connection list
EOF
        exit 1
    fi

    local remote_arch
    remote_arch="$(podman --connection "${PODMAN_CONNECTION}" info --format '{{.Host.Arch}}')"
    if [[ "${remote_arch}" != "arm64" && "${remote_arch}" != "aarch64" ]]; then
        die "Remote host is ${remote_arch}, expected arm64/aarch64 (scitrera base is arm64-only)"
    fi
    echo "Remote podman is reachable (arch=${remote_arch})"
}

# ============================================================================
# Verify base image is available on the build host
# ============================================================================

ensure_base_image_present() {
    resolve_base_image
    local base_image="${EFFECTIVE_BASE_IMAGE}"
    log "Verifying base image '${base_image}' on '${PODMAN_CONNECTION}' (from ${BASE_IMAGE_SOURCE})"

    if podman --connection "${PODMAN_CONNECTION}" image exists "docker.io/${base_image}" 2>/dev/null; then
        echo "Base image found as docker.io/${base_image}"
        return 0
    fi
    if podman --connection "${PODMAN_CONNECTION}" image exists "${base_image}" 2>/dev/null; then
        echo "Base image found as ${base_image}"
        return 0
    fi

    echo "Base image not found locally — pulling from Docker Hub..."
    if podman --connection "${PODMAN_CONNECTION}" pull "docker.io/${base_image}"; then
        echo "Base image pulled"
        return 0
    fi

    case "${base_image}" in
        xomoxcc/dgx-spark-pytorch-dev:*)
            die "xomoxcc base image '${base_image}' is not on ${PODMAN_CONNECTION} and not pullable — build it first via scripts/build_pytorch_base_image.sh, or switch to --base scitrera."
            ;;
        *)
            die "Base image '${base_image}' not found locally and pull from Docker Hub failed."
            ;;
    esac
}

# ============================================================================
# Build via remote podman socket
# ============================================================================

run_build() {
    log "Running podman build on '${PODMAN_CONNECTION}' (45–75 minutes expected)"

    resolve_base_image

    # UTC-ISO-8601 timestamp, gestempelt zu Build-Start. Läuft als
    # --build-arg BUILDTIME in den Dockerfile-ARG (Default "unknown"),
    # landet als ENV BUILDTIME + OCI-Label im Image und wird vom
    # Entrypoint in den Container-Log geschrieben.
    local buildtime
    buildtime="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    echo "Build parameters:"
    echo "  IMAGE_TAG          = ${IMAGE_TAG}"
    echo "  IMAGE_TAG (dated)  = ${IMAGE_TAG_DATED}"
    echo "  BASE_IMAGE         = ${EFFECTIVE_BASE_IMAGE}  [${BASE_IMAGE_SOURCE}]"
    echo "  COMFYUI_REF        = ${COMFYUI_REF}"
    echo "  BUILD_JOBS         = ${BUILD_JOBS}"
    echo "  BUILD_XFORMERS     = ${BUILD_XFORMERS}"
    echo "  BUILD_SAGE_ATTN    = ${BUILD_SAGE_ATTN}"
    echo "  BUILDTIME          = ${buildtime}"

    # The build context is scripts/comfyui/. Podman streams it to the remote
    # host over the socket; the build runs natively on arm64 (no QEMU) and
    # the result lands in the remote host's local image store.
    podman --connection "${PODMAN_CONNECTION}" build \
        -f "${CONTEXT_DIR}/Dockerfile" \
        --build-arg "BASE=${EFFECTIVE_BASE_IMAGE}" \
        --build-arg "COMFYUI_REF=${COMFYUI_REF}" \
        --build-arg "BUILD_XFORMERS=${BUILD_XFORMERS}" \
        --build-arg "BUILD_SAGE_ATTN=${BUILD_SAGE_ATTN}" \
        --build-arg "MAX_JOBS=${BUILD_JOBS}" \
        --build-arg "BUILDTIME=${buildtime}" \
        -t "${IMAGE_TAG}" \
        -t "docker.io/${IMAGE_TAG}" \
        -t "docker.io/${IMAGE_TAG_DATED}" \
        "${CONTEXT_DIR}"

    if ! podman --connection "${PODMAN_CONNECTION}" image exists "docker.io/${IMAGE_TAG}"; then
        die "Build finished but docker.io/${IMAGE_TAG} is not in the remote image store — check podman output above"
    fi
    echo "Remote build complete: ${IMAGE_TAG} (also tagged as docker.io/${IMAGE_TAG} and ${IMAGE_TAG_DATED})"
}

# ============================================================================
# Transfer built image from remote to local
# ============================================================================

transfer_image_from_remote() {
    log "Copying docker.io/${IMAGE_TAG} from ${PODMAN_CONNECTION} to local image store"

    podman image rm "${IMAGE_TAG}" 2>/dev/null || true
    podman image rm "docker.io/${IMAGE_TAG}" 2>/dev/null || true
    podman image rm "localhost/${IMAGE_TAG}" 2>/dev/null || true

    local size size_human
    size=$(podman --connection "${PODMAN_CONNECTION}" image inspect \
            --format '{{.Size}}' "docker.io/${IMAGE_TAG}" 2>/dev/null || echo "")
    if [[ -n "${size}" ]] && command -v numfmt >/dev/null 2>&1; then
        size_human=$(numfmt --to=iec --suffix=B "${size}")
        echo "Transfer target: ${size_human} (${size} bytes)"
    elif [[ -n "${size}" ]]; then
        echo "Transfer target: ${size} bytes"
    fi

    if command -v pv >/dev/null 2>&1; then
        local pv_args=(-ptebar)
        [[ -n "${size}" ]] && pv_args+=(-s "${size}")
        set -o pipefail
        podman --connection "${PODMAN_CONNECTION}" image save "docker.io/${IMAGE_TAG}" \
            | pv "${pv_args[@]}" \
            | podman image load \
            || die "streamed image transfer failed"
    else
        set -o pipefail
        podman --connection "${PODMAN_CONNECTION}" image save "docker.io/${IMAGE_TAG}" \
            | podman image load \
            || die "streamed image transfer failed"
    fi

    # podman save | load strips the docker.io/ prefix — retag so push works.
    if podman image exists "localhost/${IMAGE_TAG}"; then
        podman tag "localhost/${IMAGE_TAG}" "docker.io/${IMAGE_TAG}"
    elif podman image exists "${IMAGE_TAG}"; then
        podman tag "${IMAGE_TAG}" "docker.io/${IMAGE_TAG}"
    fi

    podman image inspect "docker.io/${IMAGE_TAG}" >/dev/null \
        || die "Image not present locally after transfer — check podman output"
    echo "Image transferred: docker.io/${IMAGE_TAG}"
}

# ============================================================================
# Push
# ============================================================================

run_push() {
    if (( PUSH_IMAGE == 0 )); then
        log "Skipping push (--no-push)"
        return
    fi

    log "Pushing docker.io/${IMAGE_TAG} to Docker Hub"

    local auth_file="${REGISTRY_AUTH_FILE:-${XDG_RUNTIME_DIR:-/run}/containers/auth.json}"
    [[ -f "${auth_file}" ]] || auth_file="${HOME}/.docker/config.json"
    if [[ ! -f "${auth_file}" ]]; then
        warn "No registry auth file found (checked \$REGISTRY_AUTH_FILE and ~/.docker/config.json)"
        echo "Run 'podman login docker.io -u xomoxcc' on this host, then re-run."
        die "Registry authentication missing"
    fi

    podman push "docker.io/${IMAGE_TAG}"
    echo "Image pushed: docker.io/${IMAGE_TAG}"
}

# ============================================================================
# Next steps
# ============================================================================

print_next_steps() {
    if (( NO_LOCAL_COPY == 1 )); then
        cat <<EOF

$(log "Remote-only build complete")

Image: ${IMAGE_TAG}
Location: docker.io/${IMAGE_TAG} in ${PODMAN_CONNECTION}'s podman store only
          (NOT on this control host, NOT pushed to Docker Hub)

If you need it on the other sparks, distribute it via a throwaway registry
on ${REMOTE_HOST#*@} or push from there manually once credentials are set up.
EOF
        return
    fi

    cat <<EOF

$(log "Build + push complete")

Image: ${IMAGE_TAG}

Next steps:

1. Bump comfyui_image in roles/k8s_dgx/defaults/main.yml:
     comfyui_image: "${IMAGE_TAG}"

2. Simplify roles/k8s_dgx/tasks/comfyui.yml: the git-clone + pip-install
   block in the launch ConfigMap is now obsolete (both are baked into
   the image). The container's ENTRYPOINT starts ComfyUI directly; you
   can drop the \`command\` override + launch-script ConfigMap entirely.

3. Deploy (only after explicit user approval):
     ansible-playbook k8s_dgx.yml --tags comfyui -e comfyui_enabled=true

4. Verify:
     kubectl --context=ht@dgxarley -n comfyui logs -f deploy/comfyui
   Expected in the log:
     [entrypoint] torch: 2.10.x cuda 13.x cap (12, 1)
EOF
}

# ============================================================================
# Main
# ============================================================================

main() {
    preflight
    ensure_podman_connection
    ensure_base_image_present
    run_build
    if (( NO_LOCAL_COPY == 0 )); then
        transfer_image_from_remote
        run_push
    else
        log "Skipping local copy + push (--no-local-copy) — image stays on ${PODMAN_CONNECTION}"
    fi
    print_next_steps
}

main "$@"
