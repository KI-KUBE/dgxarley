"""[dgxarley] loader.py: ShardedStateLoader per-file progress logging.

Patch ShardedStateLoader to log progress per shard file (no progress bar by
default).

This patch used to be applied with two `sed -i` calls (not a python heredoc):
first substituting the `for` line to add a shard-index counter, then an `a\\`
(append-after) insert of a logger.info() line right below it. The bash guard
combined an outer `grep -q 'for path in filepaths:'` existence pre-check with
a separate `elif grep -q 'for _shard_i, path in enumerate(filepaths'` already-
applied check and a final ANCHOR-DRIFT fallback. Per the conversion contract,
the pre-check is dropped (handled by _patchlib's target-file-missing check)
and the already-applied grep becomes the `marker=` argument below. The two
sed calls become one `replace()` (for-loop shard counter) followed by one
`insert_after()` (the appended logger.info line), preserving the exact
16-space indentation the original `a\\` command hardcoded.
"""

from _patchlib import Patch

patch = Patch(
    name="ShardedStateLoader per-file progress logging",
    target="sglang/srt/model_loader/loader.py",
)

OLD_FOR = "for path in filepaths:"
NEW_FOR = "for _shard_i, path in enumerate(filepaths, 1):"
APPEND_MARKER = 'logger.info(f"Loading shard {_shard_i}/{len(filepaths)}: {os.path.basename(path)}")'
APPEND_TEXT = "\n                " + APPEND_MARKER


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD_FOR, NEW_FOR, what="ShardedStateLoader for-loop shard counter")
    p.insert_after(NEW_FOR, APPEND_TEXT, marker=APPEND_MARKER, what="ShardedStateLoader per-shard logger.info")
