#!/usr/bin/env bash
# ============================================================================
# Per-directory JuiceFS cache occupancy.
#
# The local block cache (juicefs_cache_dir, default /var/lib/jfs-cache) is keyed
# by chunk id and carries no path information, so `du` on it cannot be broken
# down per directory. `juicefs warmup --check` walks a path and reports how many
# of its blocks are present locally; this script runs that per child directory
# and sorts the result.
#
# Run ON a mount node (spark1-4 / k3smaster) — the cache is per-node, so the
# numbers differ from host to host. Needs no META_PASSWORD (it goes through the
# FUSE mount, not the metadata engine).
#
#   scripts/debughelper/jfs_cache_by_dir.sh /mnt/jfs
#   scripts/debughelper/jfs_cache_by_dir.sh -d 2 /mnt/jfs/hub
#   scripts/debughelper/jfs_cache_by_dir.sh -t /mnt/jfs/hub    # sort by total
# ============================================================================
set -uo pipefail

DEPTH=1
THREADS=8
SORT_KEY=cached
JUICEFS="${JUICEFS:-/usr/local/bin/juicefs}"

usage() {
  cat >&2 <<EOF
usage: ${0##*/} [-d DEPTH] [-p THREADS] [-t] [PATH]

  -d DEPTH    directory level below PATH to report (default: ${DEPTH})
  -p THREADS  warmup worker threads (default: ${THREADS})
  -t          sort by total size instead of cached bytes
  PATH        directory inside a JuiceFS mount (default: /mnt/jfs)
EOF
  exit 2
}

while getopts ':d:p:th' opt; do
  case "$opt" in
    d) DEPTH="$OPTARG" ;;
    p) THREADS="$OPTARG" ;;
    t) SORT_KEY=total ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))
ROOT="${1:-/mnt/jfs}"

[ -x "$JUICEFS" ] || { echo "juicefs binary not found at $JUICEFS" >&2; exit 1; }
[ -d "$ROOT" ] || { echo "not a directory: $ROOT" >&2; usage; }
[[ "$DEPTH" =~ ^[0-9]+$ && "$DEPTH" -ge 1 ]] || usage

to_bytes() {  # "703 GiB" -> bytes
  awk -v v="$1" -v u="$2" 'BEGIN {
    m["B"]=1; m["KiB"]=1024; m["MiB"]=1048576; m["GiB"]=1073741824
    m["TiB"]=1099511627776; m["PiB"]=1125899906842624
    printf "%.0f", v * (u in m ? m[u] : 0)
  }'
}

# Emits: <sortkey-bytes> <name> <cached> <total> <pct>
check() {
  local dir="$1" label="$2" line c_v c_u t_v t_u pct
  line=$("$JUICEFS" warmup --check -p "$THREADS" "$dir" 2>&1 \
    | sed -n 's/.*check cache: //p' | tail -1)
  [ -n "$line" ] || { echo "WARN: no result for $dir" >&2; return; }

  c_v=$(sed -E 's/.*checked, ([0-9.]+) ([A-Za-z]+) of .*/\1/' <<<"$line")
  c_u=$(sed -E 's/.*checked, ([0-9.]+) ([A-Za-z]+) of .*/\2/' <<<"$line")
  t_v=$(sed -E 's/.* of ([0-9.]+) ([A-Za-z]+) \(.*/\1/' <<<"$line")
  t_u=$(sed -E 's/.* of ([0-9.]+) ([A-Za-z]+) \(.*/\2/' <<<"$line")
  pct=$(sed -E 's/.*\(([0-9.]+%)\).*/\1/' <<<"$line")

  local key
  if [ "$SORT_KEY" = total ]; then key=$(to_bytes "$t_v" "$t_u")
  else key=$(to_bytes "$c_v" "$c_u"); fi
  printf '%s\t%s\t%s %s\t%s %s\t%s\n' "$key" "$label" "$c_v" "$c_u" "$t_v" "$t_u" "$pct"
}

{
  printf 'BYTES\tDIR\tCACHED\tTOTAL\tPCT\n'
  while IFS= read -r d; do
    check "$d" "${d#"$ROOT"/}"
  done < <(find "$ROOT" -mindepth "$DEPTH" -maxdepth "$DEPTH" -type d | sort)
} | { read -r hdr; printf '%s\n' "$hdr"; sort -t$'\t' -k1,1 -rn; } | cut -f2- | column -t -s$'\t'

echo
# Files sitting directly in PATH are not covered by the per-child rows above.
check "$ROOT" "ALL of $ROOT" | cut -f2- | column -t -s$'\t'
