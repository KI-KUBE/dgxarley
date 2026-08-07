# Upstream Bug: hf_xet Session-Download-Group — "Unable to parse string as hex hash value"

## Status

**FIXED upstream (huggingface_hub v1.26.0, 2026-07-30) — workaround DROPPED
2026-08-03, in-cluster verification PASSED (confirmed 2026-08-07). CLOSED.**
(First diagnosed 2026-07-13, resolved upstream 2026-07-28.) The bug turned
out NOT to be in `hf_xet` at
all: `huggingface_hub`'s tree-listing cache (PR
[#4394](https://github.com/huggingface/huggingface_hub/pull/4394)) passed a
**redacted placeholder `xetHash` (64×`*`)** from the Hub `/tree` API straight
to `hf_xet` on **gated repos**, and `hf_xet` correctly rejected it with the
"hex hash" error. Fixed by huggingface_hub PR
[#4595](https://github.com/huggingface/huggingface_hub/pull/4595) (merged
2026-07-28, closes `xet-core#895`), released in **huggingface_hub v1.26.0**
(2026-07-30). Our affected repos (e.g.
`nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4`) are gated — exactly the fixed
scenario. The workaround was removed in commit `f901876` (2026-08-03):
`hf_hub_disable_xet: 0` in `group_vars/all/vault/huggingface.yml` (dummy in
`main/` flipped to match), XET re-enabled for all download containers. The
subsequent in-cluster downloads against `huggingface_hub 1.26.0` with XET
active ran clean — no "hex hash" crash. Doc renamed to
`FIXED_UPSTREAM_HF_XET_BUG.md`; kept for the diagnosis history and the
`HF_HUB_MIN_VERSION` floor rationale.

**Version state in-cluster (checked 2026-08-03):** the live image
`xomoxcc/dgx-spark-sglang:0.5.16-sm121` (built 2026-08-02) already carries
`huggingface_hub 1.26.0`, i.e. the fix, purely because nothing pinned hub and
the build resolved latest. Since that made the fixed version an accident of
build date, the recipe now sets a **floor**: `HF_HUB_MIN_VERSION=1.26.0` in
`scripts/patches/sglang-0.5.16-sm121.recipe`, wired through the new
`scripts/patches/dockerfile-hf-hub-floor.patch` (`ARG HF_HUB_MIN_VERSION`, a
gated `uv pip install "huggingface_hub>=…"` as the LAST pip action of the
builder stage). Empty/unset in every older recipe, so those are unchanged, and
the 0.5.16 artefact itself is unchanged as well (it already had 1.26.0). This
only removes the risk that a rebuild re-resolves back into the broken 1.2x
window after `hf_hub_disable_xet` is flipped to 0. Nothing needs pulling in
`requirements*.txt` or `pyproject.toml`: `huggingface-hub` appears only as an
unpinned entry in `requirements.txt` and is not imported anywhere on the
control node (all real users are container-side scripts).

Original status (historical): `hf_preload`
(and any container doing a fresh HuggingFace download) aborts on Xet-backed repos
with `RuntimeError: Task error: Unable to parse string as hex hash value`, thrown
from `hf_xet`'s **session-based** file-download-group API. The bug is present in
every `hf_xet >= 1.5.0` we can install, i.e. **latest stable `1.5.1`** and the
**newest pre-release `1.5.2rc0`** — so a plain image rebuild does NOT fix it (there
is nothing newer to pull). Downgrading `hf_xet` below 1.5.0 is blocked by
`huggingface_hub 1.23.0` (it requires `hf_xet >= 1.5.0`).

**Workaround (deployed):** set env `HF_HUB_DISABLE_XET=1` on every download
container → `snapshot_download` skips Xet and uses classic HTTPS/LFS. Wired as the
repo variable `hf_hub_disable_xet` (dummy `0` in `group_vars/all/main/huggingface.yml`,
real `1` in the vault copy), propagated to sglang head/worker, vllm, the shard Jobs,
`hf_preload`, `sglang_tune_moe`, comfyui and docling. To flip cluster-wide, change
only the vault value and redeploy.

## Symptom

```
Fetching 26 files:  27%|██▋  | 7/26 [00:00<00:00, 19.98it/s]
FAILED: nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4: Task error: Unable to parse string as hex hash value
```

The `hf_preload_download.py` catch-all turns this into `Failed models: [...]` →
`sys.exit(1)` → the whole K8s Job is marked `Failed`. Only a model doing a **fresh**
download trips it; already-cached models never exercise Xet (they "fetch" in 0:00),
which makes "only model X fails" misleading.

Full traceback (huggingface_hub 1.23.0):

```
_snapshot_download.py:513   thread_map( _inner_hf_hub_download ... )
_snapshot_download.py:493   _inner_hf_hub_download -> hf_hub_download(...)
file_download.py:1013       hf_hub_download -> _hf_hub_download_to_cache_dir
file_download.py:1236       -> _download_to_tmp_and_move
file_download.py:1920       -> xet_get(
file_download.py:563        xet_get -> session.new_file_download_group(
RuntimeError: Task error: Unable to parse string as hex hash value
```

## Affected versions

| Component | Version | Source |
|---|---|---|
| `hf_xet` | `1.5.1` | image `xomoxcc/dgx-spark-sglang:0.5.14-sm121` — **broken** |
| `hf_xet` | `1.5.2rc0` | image `xomoxcc/dgx-spark-sglang:0.5.15-sm121` — **broken** |
| `hf_xet` | `1.5.2` (stable, released 2026-07-16) | **confirmed still broken** — reported on `xet-core#895` 2026-07-23 (not yet tested in-cluster) |
| `huggingface_hub` | `1.23.0` | both images |
| `huggingface_hub` | `1.24.0` (released 2026-07-17) | **confirmed still broken** alongside `hf_xet 1.5.2` — same `xet-core#895` report, 2026-07-23 |
| `hf_xet` | `1.5.3.dev0` (diagnostic pre-release, 2026-07-24) | released by maintainer `@seanses` on `xet-core#895`, embeds the failing hash value in the error message, not a fix. Confirmed still broken by two independent reporters, 2026-07-25 and 2026-07-27 |
| `huggingface_hub` | `1.25.0` (released 2026-07-27) | **confirmed still broken**, tested alongside `hf_xet 1.5.3.dev0` in the same `xet-core#895` thread, 2026-07-27 |
| `huggingface_hub` | `1.25.1` (released 2026-07-27) | **confirmed still broken**, same test combination (huggingface_hub 1.25.1 plus hf_xet 1.5.3.dev0), reported on `xet-core#895`, 2026-07-27 |
| `huggingface_hub` | `1.26.0` (released 2026-07-30) | **contains the fix** (PR #4595, verified an ancestor of the v1.26.0 tag via compare API) — **not yet verified in-cluster** |

Neither `hf_xet` nor `huggingface_hub` is pinned in the SGLang build recipes
(`scripts/patches/sglang-0.5.1{4,5}-sm121.recipe`), so a rebuild pulls whatever pip
resolves at build time — currently the broken versions.
(Superseded 2026-08-03 for the 0.5.16 line only: that recipe now sets the floor
`HF_HUB_MIN_VERSION=1.26.0`; the 0.5.1{4,5} recipes stay unpinned.)

xet-core releases (checked 2026-07-13): `v1.5.2-rc0` (2026-07-09, pre-release, newest),
`v1.5.1` (2026-06-08, latest stable), `v1.5.0` (2026-05-06, **"Session based API"** —
where the failing `session.new_file_download_group` was introduced), `v1.4.3` (2026-03-31).

## Root cause

**Confirmed upstream (2026-07-27/28, `@seanses` on `xet-core#895` + fix PR
huggingface_hub#4595):** huggingface_hub PR #4394 added a tree-listing cache
optimization that skips the per-file HEAD call for xet files when the file
metadata can be rebuilt from the cached `/tree` API response. On **gated
repos** where the caller lacks content access, the Hub `/tree` API returns a
**redacted placeholder `xetHash`** (64 `*` characters) instead of the real
hash; huggingface_hub passed that placeholder straight to `hf_xet`, which
correctly rejected it ("Unable to parse string as hex hash value"). The
HTTP/non-xet path never hit this because it goes through HEAD
(`X-Xet-Hash` header), which returns either the real hash or a 401. So
`hf_xet` was never broken — the diagnostic `1.5.3.dev0` build is moot. The
fix (PR #4595, in hub v1.26.0) adds `is_valid_xet_hash()` /
`is_valid_tree_entries()` and falls back to HEAD when the tree-cache hash is
invalid/redacted. This exactly explains the measured split below:
`snapshot_download` consumed the (redacted) tree listing, `hf_hub_download`
took the per-file HEAD path.

### Our measurement (2026-07-13, pre-confirmation — consistent with the above)

The distinguishing variable is the **huggingface_hub download code path**, not the
data, the version, the cache, concurrency, disk, or the CAS service:

| Call | Result |
|---|---|
| `hf_hub_download(repo, filename=<shard>)` | ✅ reliably OK (shard1: 2/2, ~46s each) |
| `snapshot_download(repo, allow_patterns=[<shard>])` (what the Job uses) | ❌ always `hex hash` crash |

Both download the **same file** over the **same Xet CAS**, same clean cache, same
`hf_xet`. `snapshot_download` drives `hf_xet`'s new **session-based** download-group
API (added in hf_xet 1.5.0); `hf_hub_download` takes the per-file path. The crash is
in that session path.

The unparseable "hex hash" is **not** the file-level Xet hash: the `x-xet-hash`
resolve headers for all Scout shards are valid 64-hex (verified via
`curl -sI .../resolve/main/<shard>`), e.g. shard1
`b3bf96b94eefa65ef3c4bc393c3843e4fd41a8c29648d87b35798642025acf47`. The bad hash is
inside the CAS **reconstruction** response
(`cas-server.xethub.hf.co/v1/reconstructions/<hash>`) as consumed by the session
download group.

### Ruled out (with the test that ruled it out)

- **Auth / token** — no (partial files download fine with the `dgx_read_ALL` token).
- **Image bump 0.5.14 → 0.5.15** — no (0.5.15 = hf_xet 1.5.2rc0 fails identically).
- **Concurrency** — no (`snapshot_download(max_workers=1)` still fails).
- **Corrupt local Xet cache** — no (`rm -rf /mnt/jfs/xet`; recreated; still fails).
- **Disk full** — no (red herring; `/mnt/jfs` had 228 G free, shard is 5 G).
- **Server-side CAS flakiness** — no (`hf_hub_download` of a shard succeeds reliably, 2/2).
- **File-specific bad hash** — no (shard1 works via `hf_hub_download`, fails via
  `snapshot_download`; the file data is fine).

## Reproduction (faithful)

Run on spark2 (`192.168.191.202`, hostname-verify first), against the exact image,
with the shared JuiceFS HF cache mounted the way the Job mounts it:

```bash
podman run --rm --network host \
  -e HF_TOKEN="$HF_TOKEN" \
  -v /mnt/jfs:/root/.cache/huggingface \
  xomoxcc/dgx-spark-sglang:0.5.14-sm121 \
  python3 -c 'from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id="nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4", \
      cache_dir="/root/.cache/huggingface/hub", \
      allow_patterns=["model-00002-of-00014.safetensors"], force_download=True)'
# -> RuntimeError: Task error: Unable to parse string as hex hash value
```

Add `-e HF_HUB_DISABLE_XET=1` and it downloads cleanly (classic HTTPS/LFS).
The truly faithful Job repro runs `roles/k8s_dgx/files/{hf_preload_run.sh,hf_preload_download.py}`
via `/bin/bash /scripts/run.sh` with the Job's env; single-file python one-liners
are apples-to-oranges (they take the `hf_hub_download` path, which works).

## Why a rebuild does not help

- Latest stable `hf_xet 1.5.1` is already the broken version; newest pre-release
  `1.5.2rc0` is also broken. There is no fixed release to pull.
- Pinning `hf_xet < 1.5.0` is blocked: with `huggingface_hub 1.23.0`, installing
  `hf_xet==1.4.3` yields `ValueError: To use optimized download using Xet storage,
  you need to install the hf_xet package ...` (hub 1.23.0 requires the ≥1.5.0
  session API). Restoring Xet would need a coordinated downgrade of **both**
  `huggingface_hub` and `hf_xet` — not worth it for a one-time cache warmup where
  Xet only buys download speed.

## Upstream tracking

- **Exact-string issue is now filed:** **[`huggingface/xet-core#895`](https://github.com/huggingface/xet-core/issues/895)**
  ("Download fails with 'Task error: Unable to parse string as hex hash
  value' (hf-xet 1.5.1)"), filed **2026-07-11** — it existed before this
  doc's original 2026-07-13 write-up, our search just missed it. **Still
  OPEN as of 2026-07-23.** Collaborator @seanses acknowledged on
  2026-07-16 that "huggingface_hub is passing incorrect file hash to
  hf-xet leading to this error", asking for the affected repo id.
  Maintainer @Wauplin followed up on **2026-07-23** (today) asking that
  the failing hash value be surfaced in the error message itself
  (`Task error: ... (got '<hash>')`) so the report can be narrowed down —
  no root cause identified, no fix merged. **TODO superseded:** no longer
  need to file our own issue; instead consider adding our reproduction
  details (faithful `snapshot_download` vs `hf_hub_download` split, see
  above) as a comment on #895 if it stays unresolved.
- **Update 2026-07-28: real movement on #895, still no fix shipped.**
  Maintainer `@seanses` released `hf_xet 1.5.3.dev0` on 2026-07-24 (a
  diagnostic build that embeds the failing hash value in the error
  message, not a fix). Two independent reporters reproduced the
  failure on that build, on 2026-07-25 and again on 2026-07-27 (the
  2026-07-27 report also used `huggingface_hub 1.25.1`, confirming that
  release is broken too). On 2026-07-27, `@seanses` posted a full root
  cause investigation summary on the issue, the first concrete
  explanation of how the hash gets corrupted. On 2026-07-28 (today),
  `@Wauplin` replied thanking `@seanses` for the investigation and said
  he would check the `huggingface_hub` side for a fix. Bottom line
  unchanged: nothing merged, nothing released, `HF_HUB_DISABLE_XET=1`
  remains mandatory, but for the first time a fix looks plausible in
  the near term, worth checking back again soon.
- **Update 2026-08-03: RESOLVED.** `xet-core#895` was **CLOSED
  2026-07-28T16:05:37Z** — the same day the previous entry was written.
  `@seanses`' root-cause comment (2026-07-27) identified the gated-repo
  redacted-`xetHash` mechanism (see "Root cause" above); `@Wauplin`'s
  fix landed as huggingface_hub PR
  [#4595](https://github.com/huggingface/huggingface_hub/pull/4595)
  ("[Download] Reject redacted Xet hashes from tree cache", merged
  2026-07-28T16:05:36Z, "Fixes xet-core#895") and shipped in
  **huggingface_hub v1.26.0** (2026-07-30; fix commit confirmed an
  ancestor of the tag via the compare API). `hf_xet` itself needed no
  fix. Workaround-drop procedure below now actionable.
- Related closed reports (symptom cluster, same 1.5.x era, resolved
  independently of this bug):
  - huggingface/xet-core #358 — "errors became very common" with snapshot_download (closed)
  - huggingface/xet-core #399 — "Cannot Download XET Files" (closed)
  - huggingface/xet-core #483 — "Still can't download models" (closed)
  - huggingface/huggingface_hub #3960 — "Downloading not working with hf_xet" (still open, unconfirmed relation)
  - huggingface/huggingface_hub #3643 — snapshot_download blob checksum mismatch (XET) (closed)
- Watch: <https://github.com/huggingface/xet-core/issues/895> directly
  (now the actionable tracking issue), plus
  <https://github.com/huggingface/xet-core/releases> and the
  `huggingface_hub` changelog. `hf_xet 1.5.2` (2026-07-16) and
  `huggingface_hub 1.24.0` (2026-07-17) have both shipped since the
  original diagnosis — neither fixes this (see Affected versions table
  and Changelog below).

## How to know when to drop the workaround

**DONE — all three steps completed.** The trigger fired 2026-08-03
(huggingface_hub v1.26.0 contains the fix; the fixing component is
`huggingface_hub`, any `hf_xet` version is fine):

1. ~~Rebuild an image OR test in a throwaway container with
   `huggingface_hub >= 1.26.0`.~~ Live image 0.5.16-sm121 ships hub 1.26.0
   (+ recipe floor `HF_HUB_MIN_VERSION=1.26.0`).
2. ~~Run the faithful reproduction above **without** `HF_HUB_DISABLE_XET`.~~
   In-cluster downloads with XET re-enabled ran clean (verified, 2026-08-07).
3. ~~If clean → set `hf_hub_disable_xet: 0` in
   `group_vars/all/vault/huggingface.yml` and redeploy; move this doc to
   `FIXED_UPSTREAM_HF_XET_BUG.md`.~~ Flag flipped in commit `f901876`
   (2026-08-03); doc renamed 2026-08-07.

## Changelog

- **2026-07-13** — First diagnosis. Isolated to the `snapshot_download` →
  `session.new_file_download_group` path (hf_xet ≥ 1.5.0). Confirmed rebuild won't
  fix (1.5.1 / 1.5.2rc0 both broken; 1.4.3 blocked by hub 1.23.0). Workaround
  `HF_HUB_DISABLE_XET=1` wired as `hf_hub_disable_xet` across all download containers.
- **2026-07-23** — The bug is tracked upstream after all:
  `huggingface/xet-core#895` was filed 2026-07-11 (before this doc's first
  write-up) and remains **OPEN** — maintainer @Wauplin engaged today
  (2026-07-23) requesting more diagnostic detail, no fix yet. A commenter
  on the issue confirmed the identical failure on **hf_xet 1.5.2**
  (2026-07-16 stable) and **huggingface_hub 1.24.0** (2026-07-17) on
  2026-07-23, so the newer stable releases do not fix it either.
  `HF_HUB_DISABLE_XET=1` workaround unchanged and still required.
- **2026-07-28**: checked `xet-core#895` again, still open, but real
  movement this time. `@seanses` shipped `hf_xet 1.5.3.dev0`
  (2026-07-24), a diagnostic-only release that embeds the failing hash
  in the error message. Reproductions on that build were reported on
  2026-07-25 and 2026-07-27, the latter also on `huggingface_hub 1.25.1`
  (released 2026-07-27, confirmed still broken, not previously listed
  in this doc). `@seanses` posted a root cause investigation on
  2026-07-27, and `@Wauplin` responded on 2026-07-28 saying he would
  check the `huggingface_hub` fix. Nothing merged or released yet.
  `HF_HUB_DISABLE_XET=1` remains required, but a fix now looks close,
  recheck again soon.
- **2026-08-03** — Bug is fixed upstream: `xet-core#895` closed 2026-07-28,
  root cause confirmed as huggingface_hub's tree-cache passing the
  gated-repo redacted `xetHash` placeholder (introduced by hub PR #4394),
  fixed by hub PR #4595, released in **huggingface_hub v1.26.0**
  (2026-07-30). Doc updated throughout (Status, Affected versions, Root
  cause, Upstream tracking). Workaround `HF_HUB_DISABLE_XET=1` stays
  deployed until the reproduction is re-run in-cluster against hub ≥1.26.0;
  then flip `hf_hub_disable_xet: 0` and rename this doc to `FIXED_…`.
- **2026-08-03 (later)** — verified that the live image
  `xomoxcc/dgx-spark-sglang:0.5.16-sm121` (build 2026-08-02, id `e13762c84ff9`)
  ships `huggingface_hub 1.26.0` + `hf_xet 1.6.0.dev0`, i.e. the fix is already
  in the image, by accident (nothing pinned hub). Made it deliberate with a
  FLOOR pin instead of leaving it to pip resolution:
  `scripts/patches/dockerfile-hf-hub-floor.patch` (new, `ARG
  HF_HUB_MIN_VERSION` + gated `uv pip install` as the last pip action of the
  builder stage, trailing-context-only hunk so it applies last and independent
  of the optional patches), plumbed through `scripts/build_sm121_image.sh`
  (apply step 2g, recipe var, `--build-arg`), and
  `HF_HUB_MIN_VERSION=1.26.0` set in `sglang-0.5.16-sm121.recipe`. Older
  recipes leave it empty, so the knob is a no-op there. Patch chain
  dry-run-verified with zero fuzz against the pristine upstream Dockerfile and
  against the full 0.5.16 chain. No change to `requirements*.txt` /
  `pyproject.toml` (control node does not import hub at all). Workaround
  `hf_hub_disable_xet: 1` still deployed; the in-cluster reproduction is still
  the open step before flipping it.
- **2026-08-03 (commit `f901876`)** — workaround dropped:
  `hf_hub_disable_xet: 0` in `group_vars/all/vault/huggingface.yml` (public
  dummy in `main/` flipped to match), re-enabling XET for every
  HF-downloading container. Same commit added the unrelated
  `hf_xet_reconstruct_write_sequentially` knob for the JuiceFS HDD backend.
- **2026-08-07** — in-cluster verification confirmed PASSED: downloads on
  gated repos with XET active against `huggingface_hub 1.26.0` ran clean, no
  "hex hash" crash. Upstream re-checked 2026-08-05 and 2026-08-07: hub
  v1.26.0 still latest, `xet-core#895` closed with no reopen/regression
  reports; `hf_xet` stable v1.6.0 released 2026-08-03 (no relevance — hub was
  the fixed component). Doc renamed `UPSTREAM_HF_XET_BUG.md` →
  `FIXED_UPSTREAM_HF_XET_BUG.md`. Bug closed.
