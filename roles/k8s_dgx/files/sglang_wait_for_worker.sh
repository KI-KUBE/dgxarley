#!/bin/sh
# Poll-loop waiting for a worker pod to become Ready.
# Uses active polling instead of `kubectl wait` to avoid the watch-based
# race condition where the condition is already met before the watch starts.
#
# Environment variables:
#   WAIT_NAMESPACE  — Kubernetes namespace (required)
#   WAIT_LABEL      — pod label selector (required)
#   WAIT_TIMEOUT    — timeout in seconds (default: 600)
#   WAIT_INTERVAL   — poll interval in seconds (default: 5)

NS="${WAIT_NAMESPACE:?WAIT_NAMESPACE is required}"
LABEL="${WAIT_LABEL:?WAIT_LABEL is required}"
TIMEOUT="${WAIT_TIMEOUT:-600}"
INTERVAL="${WAIT_INTERVAL:-5}"

elapsed=0
echo "Waiting for pod (label: ${LABEL}) in namespace ${NS} ..."

while [ $elapsed -lt $TIMEOUT ]; do
  phase=$(kubectl get pod -n "$NS" -l "$LABEL" \
    -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
  ready=$(kubectl get pod -n "$NS" -l "$LABEL" \
    -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
  echo "$(date '+%H:%M:%S') worker: phase=${phase:-not-found} ready=${ready:-unknown} (${elapsed}s/${TIMEOUT}s)"
  if [ "$ready" = "True" ]; then
    echo "Worker pod is Ready."
    exit 0
  fi
  sleep $INTERVAL
  elapsed=$((elapsed + INTERVAL))
done

echo "ERROR: Timed out after ${TIMEOUT}s waiting for worker."
exit 1
