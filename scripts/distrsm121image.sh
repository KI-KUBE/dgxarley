#!/usr/bin/env bash
#
# distrsm121image.sh — Distribute the freshly-built sgl-kernel sm121 image
# from spark4's local podman store to all 4 DGX Spark K3s nodes' containerd
# image stores, using the QSFP network (10.10.10.0/24) for the heavy lifting.
#
# Topology (inverted pull model)
# -------------------------------
# The x86 control host SSHes to each target spark in turn (via management
# ethernet). The target then SSHes back to spark4 via its QSFP interface
# (10.10.10.4) and runs `podman save` there. The resulting docker-archive
# stream flows back over QSFP to the target, through a local `pv` for
# progress, and directly into `k3s ctr image import` on the target — which
# lands the image in the k3s-managed containerd namespace (k8s.io) under
# the docker.io/... FQN so K3s finds it without any registry pull.
#
# Why inverted (target pulls) rather than spark4-fanout (source pushes):
#   - No host→QSFP-IP mapping needed; INNERSOURCE is always spark4 from
#     each target's perspective.
#   - Only two SSH quoting layers instead of three.
#   - The heavy k3s-ctr work runs locally on the target where the image
#     will actually be used, so no extra network hop for the final import.
#
# Prerequisites
# -------------
# - spark4 has `localhost/xomoxcc/dgx-spark-sglang:0.5.10-sm121` in its
#   local podman store (built via scripts/build_sm121_image.sh, either with
#   or without --no-push).
# - Root SSH from x86 control host to all 4 sparks (management) works.
# - Root SSH from spark1/2/3 to 10.10.10.4 (spark4's QSFP IP) works
#   passwordless. Verify once with:
#     for h in spark1 spark2 spark3; do
#       ssh root@${h}.local "ssh -o BatchMode=yes root@10.10.10.4 hostname"
#     done
# - `pv` installed on all 4 target sparks (apt install pv).
#
# image rm on the targets is not needed: `k3s ctr image import` replaces
# the tag atomically when the same name is imported again.
#

set -euo pipefail

SRC_IMAGE="localhost/xomoxcc/dgx-spark-sglang:0.5.10-sm121"
IMAGE="docker.io/xomoxcc/dgx-spark-sglang:0.5.10-sm121"

# Source host: where the built image lives in podman.
SOURCE="spark4.local"              # management address for outer ssh
INNERSOURCE="10.10.10.4"           # spark4's QSFP IP — all targets ssh-back here

# Target sparks (management addresses). Includes spark4 itself, which
# gets special-cased to avoid a needless ssh-to-self loop.
TARGETS=(spark1.local spark2.local spark3.local spark4.local)

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

#set -x

# Auf spark4 — retag für docker.io namespace (wichtig für containerd name-match).
# Ohne das Retag bleibt das Image als localhost/xomoxcc/... im podman store
# liegen; `k3s ctr image import` würde den Content zwar übernehmen, aber unter
# einem Namen den K3s nicht findet wenn der Pod-Spec docker.io/xomoxcc/...
# referenziert.
ssh "${SSH_OPTS[@]}" "root@${SOURCE}" "podman tag '${SRC_IMAGE}' '${IMAGE}'"

# Image-Größe einmal abfragen (unkomprimierte Layer-Summe — docker-archive
# streamt unkomprimiert, also matcht pv's ETA gut genug).
SIZE=$(ssh "${SSH_OPTS[@]}" "root@${SOURCE}" \
        "podman image inspect --format '{{.Size}}' '${IMAGE}'")
echo "SIZE: ${SIZE}"



# Image an alle 4 Nodes streamen.
for host in "${TARGETS[@]}"; do
    short="${host%%.*}"
    echo "=== streaming to ${short} ==="

    if [[ "${host}" == "${SOURCE}" ]]; then
        # spark4 auf sich selbst: kein ssh-hop nötig, direkte Pipe in einer
        # einzelnen SSH-Session. pv läuft auf spark4; stderr kommt durch die
        # SSH-Session zurück zum Control-Host-Terminal.
        ssh "${SSH_OPTS[@]}" "root@${host}" \
            "podman save --format docker-archive '${IMAGE}' \
             | pv -fptebar -s '${SIZE}' -N ${short} \
             | k3s ctr -n k8s.io image import -"
    else
        # Inverted-pull: Control-Host sshes zum Target (management), Target
        # sshes zurück zum spark4 über QSFP für den podman save. Der Stream
        # fließt über QSFP (200 GbE) statt Management (2.5 GbE), pv logged
        # auf dem Target, und k3s ctr import läuft lokal dort wo das Image
        # gebraucht wird.
        #
        # Quoting: die äußeren "..." machen local-expansion von ${IMAGE},
        # ${SIZE}, ${short}, ${INNERSOURCE}. Die inneren '...' um
        # 'root@...' und 'podman save ...' werden als Literale an das Target
        # weitergegeben und dort vom Remote-Shell ausgewertet. 'k3s ctr ...'
        # steht am Ende OHNE single-quotes — Single-Quotes um ein Kommando
        # würden es zu einem Command-Namen-mit-Spaces machen (vormaliger Bug).
        ssh "${SSH_OPTS[@]}" "root@${host}" \
            "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new 'root@${INNERSOURCE}' 'podman save --format docker-archive ${IMAGE}' \
             | pv -fptebar -s '${SIZE}' -N ${short} \
             | k3s ctr -n k8s.io image import -"
    fi
done

echo "=== distribution complete ==="

# Optional sanity check — uncomment to verify each target has the new image
# in the k8s.io namespace with the right digest:
#
# for host in "${TARGETS[@]}"; do
#     echo "--- ${host} ---"
#     ssh "${SSH_OPTS[@]}" "root@${host}" \
#         "k3s ctr -n k8s.io image list -q | grep -F '${IMAGE}' || echo MISSING"
# done
