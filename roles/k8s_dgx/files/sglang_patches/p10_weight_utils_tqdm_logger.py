"""[dgxarley] weight_utils.py: tqdm -> logger.info for safetensors shard-load progress.

Patch weight loading iterators to log progress per shard file.
tqdm writes directly to sys.stderr in TP worker subprocesses -- this output is
NOT forwarded by SGLang's logger infrastructure, so it never appears in kubectl
logs. Additionally, BAR_FORMAT lacks a trailing \\n, so tqdm uses \\r (carriage
return) which is invisible in non-TTY kubectl logs.
Fix: replace tqdm loops with logger.info() calls that go through the logging
pipeline.

v0.5.10: enable_multithread_load defaults to True, so the default code path is
buffered_multi_thread_safetensors_weights_iterator (not the old single-thread
one). We patch both to cover all cases.

--- Patch 1: single-thread safetensors_weights_iterator (v0.5.10rc0 path) ---
Straight tqdm -> logger.info swap.

--- Patch 2: buffered_multi_thread_safetensors_weights_iterator (v0.5.15) ---
v0.5.15 stores (st_file, future) TUPLES in `pending` and does
`st_file, future = pending.popleft()` (for the drop_cache_after_load feature).
The old whole-block anchor (expecting `future = pending.popleft()`) silently
MISSED on v0.5.15 -> the multi-thread NVFP4 weight load logged NOTHING (the
tqdm bar writes \\r to non-forwarded stderr; this is the default load path for
the big NVFP4 models like glm-5.2-reap-504B). Minimal + drift-resistant fix:
keep the (invisible) tqdm bar, add ONE logger.info per shard, anchored on the
stable 2-line pop+result. See remote-control 2026-07-15.

Idempotency (2026-07-16): key the already-applied check on a DEDICATED marker
checked FIRST. old_buffered is a PREFIX of new_buffered, so `old_buffered in
code` stays true after patching -> checking it first would re-inject a second
logger.info on any re-run. The marker is also patch-specific: Patch 3 injects
the same "Loading shard (multi-thread): " string, so keying the guard on that
string would cross-contaminate between the two patches.

--- Patch 3: multi_thread_safetensors_weights_iterator (non-buffered variant) ---
RE-ANCHORED 2026-07-16: _load_file() now returns a (st_file, result) TUPLE
(same shape change already handled for the buffered variant in Patch 2 above),
so `future.result()` unpacks as `st_file, state_dict = future.result()`, not a
bare state_dict -- the old single-value anchor never matched. New anchor also
covers the drop_cache_after_load tail so that logic is not silently dropped by
the patch.

Note: weight_utils.py itself does not import `os` at module scope pre-patch;
the original heredoc unconditionally prepended a bare `import os` (needed by
the new logger.info calls' `os.path.basename(...)`) whenever it was missing,
gated on whether any of the three sub-patches above actually changed
something. There is no stable textual anchor for "start of file", so that one
edit is applied directly against the patch's buffered code/changed state
below instead of through replace()/insert_after().
"""

from _patchlib import Patch

patch = Patch(
    name="tqdm -> logger.info progress logging for safetensors weight-loading iterators",
    target="sglang/srt/model_loader/weight_utils.py",
)

# --- logger import (added once, if the file doesn't already define a module logger) ---
OLD_LOGGER_IMPORT = "from tqdm.auto import tqdm"
NEW_LOGGER_IMPORT = "import logging\nfrom tqdm.auto import tqdm\nlogger = logging.getLogger(__name__)"

# --- Patch 1: single-thread safetensors_weights_iterator (v0.5.10rc0 path) ---
OLD_SINGLE = """    for st_file in tqdm(
        hf_weights_files,
        desc="Loading safetensors checkpoint shards",
        disable=not enable_tqdm,
        bar_format=BAR_FORMAT,
        position=tqdm._get_free_pos(),
    ):"""
NEW_SINGLE = """    _total = len(hf_weights_files)
    for _i, st_file in enumerate(hf_weights_files, 1):
        if enable_tqdm:
            logger.info(f"Loading safetensors shard {_i}/{_total}: {os.path.basename(st_file)}")"""

# --- Patch 2: buffered_multi_thread_safetensors_weights_iterator (v0.5.15) ---
OLD_BUFFERED = """                st_file, future = pending.popleft()
                state_dict = future.result()"""
BUF_MARKER = "# [patch] _sgl_buf_iter_per_shard_log_"
BUFFERED_APPEND = (
    "\n                " + BUF_MARKER + "\n"
    "                if enable_tqdm:\n"
    '                    logger.info(f"Loading shard (multi-thread): {os.path.basename(st_file)} '
    '({len(state_dict)} tensors)")'
)

# --- Patch 3: multi_thread_safetensors_weights_iterator (non-buffered variant) ---
MT_MARKER = "# [patch] _sgl_mt_iter_per_shard_log_"
OLD_MT = (
    "        for future in futures_iter:\n"
    "            st_file, state_dict = future.result()\n"
    "            for name, param in state_dict.items():\n"
    "                yield name, param\n"
    "            del state_dict\n"
)
NEW_MT = (
    "        " + MT_MARKER + "\n"
    "        _mt_total = len(hf_weights_files)\n"
    "        _mt_done = 0\n"
    "        for future in futures_iter:\n"
    "            st_file, state_dict = future.result()\n"
    "            _mt_done += 1\n"
    "            if enable_tqdm:\n"
    "                logger.info(\n"
    '                    f"Loading shard (multi-thread): {_mt_done}/{_mt_total} "\n'
    '                    f"{os.path.basename(st_file)} ({len(state_dict)} tensors)"\n'
    "                )\n"
    "            for name, param in state_dict.items():\n"
    "                yield name, param\n"
    "            del state_dict\n"
)


@patch.run
def apply(p: Patch) -> None:
    # Only add a module logger if the file does not already define one. The
    # 0.5.15 weight_utils.py DOES define one, so this must be a no-op there --
    # dropping the condition (as the first conversion did) injected a second
    # `import logging` + `logger = ...` and made the patched file differ from
    # the pre-refactor result.
    if "\nlogger = " not in p.code and "\nlogger=" not in p.code:
        p.replace(OLD_LOGGER_IMPORT, NEW_LOGGER_IMPORT, what="add module logger")
    p.replace(OLD_SINGLE, NEW_SINGLE, what="safetensors_weights_iterator tqdm -> logger.info")
    p.insert_after(
        OLD_BUFFERED,
        BUFFERED_APPEND,
        marker=BUF_MARKER,
        what="buffered_multi_thread_safetensors_weights_iterator per-shard logger.info",
    )
    p.replace(
        OLD_MT,
        NEW_MT,
        marker=MT_MARKER,
        what="multi_thread_safetensors_weights_iterator per-shard logger.info",
    )
    # dgxarley: see module docstring note above -- prepend a bare `import os`
    # when missing, but only when we are about to write the file anyway (i.e.
    # at least one of the edits above actually changed something).
    if p.changed:
        p.prepend("import os\n", marker="import os")
