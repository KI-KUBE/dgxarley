#!/usr/bin/env bash
# Strip rule for KI-KUBE/dgxarley — the single definition used by sync-strip.yml (daily main) and
# release-tag.yml (base-vX.Y on a given upstream commit).
#
# Usage (repo root, working tree checked out at an upstream commit of vroomfondel/dgxarley):
#   bash <saved>/kikube-gate/strip.sh <saved>
# <saved> is a copy of the fork's own .github/workflows/ and .github/kikube-gate/ taken BEFORE the
# checkout of the upstream commit (the checkout replaces them with the upstream versions). Run this
# script from that copy, not from the working tree.
#
# Result: index staged with
#   - no vault/ directory and no vault.yml file (path/name based, NOT content based, so the rule also
#     holds for SOPS files whose YAML frame is clear text),
#   - .github/workflows/ = exactly the fork's workflows (inherited upstream workflows removed),
#   - .github/kikube-gate/ = the fork's gate + strip scripts.
# The caller commits and runs gate.sh. Exit != 0 if a vault path survives.
set -euo pipefail

saved="${1:?usage: strip.sh <saved-ci-dir>}"
[ -d "$saved/workflows" ] && [ -f "$saved/kikube-gate/gate.sh" ] && [ -f "$saved/kikube-gate/strip.sh" ] \
  || { echo "::error::$saved does not hold workflows/ + kikube-gate/{gate,strip}.sh" >&2; exit 2; }

# vault secrets: whole vault/ directories (split layout) + single vault.yml files (legacy layout)
find . -path ./.git -prune -o -type d -name vault -print0 | xargs -0 -r rm -rf
find . -path ./.git -prune -o -type f -name vault.yml -print0 | xargs -0 -r rm -f

# only the fork's CI
rm -rf .github/workflows .github/kikube-gate
mkdir -p .github/workflows .github/kikube-gate
cp "$saved"/workflows/*.yml .github/workflows/
cp "$saved"/kikube-gate/gate.sh "$saved"/kikube-gate/strip.sh .github/kikube-gate/
chmod +x .github/kikube-gate/*.sh
git add -A

# safety net before anything is committed / pushed (e.g. a new directory convention upstream)
remaining="$(git ls-files -z | tr '\0' '\n' | grep -E '(^|/)vault/|(^|/)vault\.yml$' || true)"
if [ -n "$remaining" ]; then
  echo "::error::ABORT - vault paths still present after strip:" >&2
  printf '%s\n' "$remaining" >&2
  exit 1
fi
echo "STRIP OK"
