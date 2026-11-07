#!/usr/bin/env bash
# Fork gate for KI-KUBE/dgxarley (public, customer-deployable base). Run from the repo root on the
# tree that is about to be published / tagged. Exit != 0 on any violation.
#   1. no group_vars/**/vault.yml, host_vars/**/vault.yml, no **/vault/ path (tracked files)
#   2. no git-crypt blob (magic bytes "\x00GITCRYPT") in any tracked file
#   3. gitleaks (pinned, sha256-verified) over the working tree
set -euo pipefail

GITLEAKS_VERSION="8.30.0"
GITLEAKS_SHA256="79a3ab579b53f71efd634f3aaf7e04a0fa0cf206b7ed434638d1547a2470a66e"  # linux_x64 tar.gz
GITLEAKS_URL="${KIKUBE_GITLEAKS_URL:-https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz}"
fail=0

echo "== 1/3 vault paths"
bad=$(git ls-files -z | tr '\0' '\n' | grep -E '(^|/)vault/|(^|/)vault\.yml$' || true)
if [ -n "$bad" ]; then echo "::error::vault paths present:"; printf '%s\n' "$bad"; fail=1; fi

echo "== 2/3 git-crypt blobs"
python3 -I - <<'PY' || fail=1
import subprocess, sys
magic = b"\x00GITCRYPT"
ls = subprocess.run(["git", "ls-files", "-s", "-z"], check=True, capture_output=True).stdout
entries = [e for e in ls.split(b"\0") if e]
hits = []
p = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
for e in entries:
    meta, path = e.split(b"\t", 1)
    mode, sha, _ = meta.split(b" ")
    if mode == b"160000":      # submodule
        continue
    p.stdin.write(sha + b"\n"); p.stdin.flush()
    _, _, size = p.stdout.readline().split()
    data = p.stdout.read(int(size)); p.stdout.read(1)
    if data.startswith(magic):
        hits.append(path.decode(errors="replace"))
p.stdin.close(); p.wait()
if hits:
    print("::error::git-crypt ciphertext tracked:"); print("\n".join(hits)); sys.exit(1)
print(f"ok ({len(entries)} files scanned)")
PY

echo "== 3/3 gitleaks v${GITLEAKS_VERSION}"
if ! command -v gitleaks >/dev/null 2>&1 || ! gitleaks version 2>/dev/null | grep -qF "$GITLEAKS_VERSION"; then
  d=$(mktemp -d)
  curl -fsSL -o "$d/gl.tgz" "$GITLEAKS_URL"
  echo "$GITLEAKS_SHA256  $d/gl.tgz" | sha256sum -c --quiet - || { echo "::error::gitleaks sha256 mismatch"; exit 4; }
  tar -xzf "$d/gl.tgz" -C "$d" gitleaks
  export PATH="$d:$PATH"
fi
# --no-git: scan the tree that will be published (history is rewritten by the sync anyway)
gitleaks detect --no-git --source . --config .gitleaks.toml --redact --no-banner || fail=1

[ "$fail" -eq 0 ] && echo "GATE OK" || { echo "GATE FAILED"; exit 1; }
