#!/usr/bin/env bash
# SGLang jfs-warmup initContainer script (ConfigMap-mounted at /scripts).
#
# Extracted from the inline `bash -c` heredoc in sglang_instance.yml so the head
# + worker Deployments share ONE copy. The one template-time value it needs is
# passed as an env var (JFS_CACHE).
#
# Pre-warm THIS node's JuiceFS blockcache for the active model(s) so the sglang
# container starts with a WARM cache (fast load) instead of pulling cold shards
# from the single spark4 backend mid-startup. This is the BLOCKING warm guarantee
# WITH live progress: submit a BACKGROUND warmup, then poll `juicefs warmup
# --check` and print the cache-fill (GiB + %) each round until 100%. The
# background warmup runs in the host JuiceFS mount process, so it keeps
# progressing (and RESUMES) across init restarts. On the ~120min cap without 100%
# we exit 1 -> kubelet restarts THIS init only (the head's download + markers
# already succeeded and are NOT re-run) and the warm resumes where it stopped.
# rc=0 (init passes) only once actually warm. `juicefs warmup` is cache-aware ->
# it fetches ONLY uncached blocks, so an already-warm node is a quick no-op.
#
# Env: JFS_CACHE=yes|no, HF_PRELOAD_MODELS (comma-list).
# Mounts: /root/.cache/huggingface (hub), /usr/local/bin/juicefs.
set -u
if [ "${JFS_CACHE:-no}" != "yes" ]; then
  echo "[jfs-warmup] JuiceFS cache off -> no warmup"; exit 0
fi
if [ ! -x /usr/local/bin/juicefs ]; then
  echo "[jfs-warmup] juicefs binary not mounted -> skip"; exit 0
fi

rc=0
for m in $(printf '%s' "$HF_PRELOAD_MODELS" | tr ',' ' '); do
  DIR="/root/.cache/huggingface/hub/models--$(printf '%s' "$m" | sed 's#/#--#g')"
  [ -d "$DIR" ] || { echo "[jfs-warmup] ${DIR} absent -> skip"; continue; }
  echo "[jfs-warmup] warming ${DIR} (background, single-thread)"
  /usr/local/bin/juicefs warmup --threads 1 --background "${DIR}" \
    || echo "[jfs-warmup] submit rc=$? (non-fatal; poll continues)"
  warm=no; last=""; stall=0
  for i in $(seq 1 240); do
    chk=$(/usr/local/bin/juicefs warmup --check "${DIR}" 2>&1)
    prog=$(printf '%s' "$chk" | grep -oE '[0-9.]+ [KMGT]?i?B of [0-9.]+ [KMGT]?i?B \([0-9.]+%\)' | tail -1)
    echo "[jfs-warmup] ${DIR} ${prog:-cached=?} (poll ${i}/240)"
    printf '%s' "$chk" | grep -q '(100.0%)' && { echo "[jfs-warmup] ${DIR} fully cached"; warm=yes; break; }
    # Stall guard: the ONE --background submit above can finish having left a
    # remainder (blocks that transiently failed against the backend are never
    # retried), and then the poll loop spins to the 120min cap on a plateau.
    # Observed: all 3 workers parked at 99.6% for ~40min while a FOREGROUND
    # warmup pulled the missing blocks of a blob in under a second. So on three
    # rounds without progress, re-warm in the foreground (cache-aware, fetches
    # only what is missing; capped so a genuinely slow backend still yields
    # progress lines instead of a silent block).
    if [ "$prog" = "$last" ]; then
      stall=$(( stall + 1 ))
      if [ "$stall" -ge 3 ]; then
        echo "[jfs-warmup] no progress for 3 polls at ${prog:-?} -> re-warming in foreground"
        timeout 600 /usr/local/bin/juicefs warmup --threads 4 "${DIR}" 2>&1 | tail -2
        stall=0
      fi
    else
      stall=0
    fi
    last="$prog"
    sleep 30
  done
  [ "$warm" = yes ] || { echo "[jfs-warmup] ERROR: ${DIR} not fully warm within ~120min cap -> failing init (kubelet retries THIS init; the background warmup keeps resuming)"; rc=1; }
done
exit $rc
