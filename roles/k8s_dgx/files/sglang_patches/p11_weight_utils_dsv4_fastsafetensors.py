"""[dgxarley] weight_utils.py: fastsafetensors loader for multi-node TP + no-GDS GB10.

fastsafetensors loader: make it usable on multi-node TP + no-GDS GB10 so the
weight load STREAMS disk->device through a bounded bounce buffer instead of
accumulating full shards in host memory (which is what swaps -- confirmed by
memray/smaps: top allocator _load_file weight_utils.py:1060, swapped mapping
= safetensors-mmap). sglang's fastsafetensors_weights_iterator does a WORLD
collective load (-> Gloo connectFullMesh timeout across our 4 nodes) onto
cuda:{world_rank} (invalid on 1-GPU worker nodes). Rewrite to:
  - SingleGroup() : each rank loads its files independently (no collective)
  - device "cuda" : the local current device (not cuda:{world_rank})
  - nogds=True    : 16 MB bounce-buffer streaming (no GPU Direct Storage here)
TP slicing is unchanged -- the per-param weight_loader slices the full
tensors, exactly as the normal safetensors iterator yields them. Inert unless
load_format=fastsafetensors. See UPSTREAM_DSV4_BUGS.md.

Note: this file targets the SAME source file as p10 (weight_utils.py) and
MUST run after it -- filename order (p10 before p11) enforces that.

This patch used to be gated (in the bash heredoc) behind an outer
`grep -q 'device = torch.device(f"cuda:{rank}")'` pre-check before even
attempting the python patch, falling back to a blanket "not needed or already
applied" message otherwise. That pre-check only inspected the LAST line of the
old anchor below: if upstream ever renamed just that device line while
leaving the surrounding pg/rank selection logic untouched, the pre-check would
silently skip the patch (treating it as "already applied") with no
ANCHOR-DRIFT signal at all. Converting to _patchlib drops that pre-check
outright (per the conversion contract: a bash `grep -q` "already applied?"
guard becomes the `marker=` argument) -- replace() below checks the already-
applied marker first, then the FULL multi-line anchor, so a partial upstream
rename now correctly surfaces as an ANCHOR-DRIFT instead of being silently
swallowed.
"""

from _patchlib import Patch

patch = Patch(
    name="fastsafetensors_weights_iterator: SingleGroup + local device + nogds",
    target="sglang/srt/model_loader/weight_utils.py",
)

OLD1 = (
    "    if torch.distributed.is_initialized():\n"
    "        pg = torch.distributed.group.WORLD\n"
    "    else:\n"
    "        pg = SingleGroup()\n"
    "\n"
    "    try:\n"
    "        rank = pg.rank()\n"
    "    except Exception:\n"
    "        rank = 0\n"
    "\n"
    '    device = torch.device(f"cuda:{rank}")'
)
NEW1 = (
    "    # dgxarley: per-rank independent load (no WORLD collective → no Gloo\n"
    "    # connectFullMesh timeout across nodes) onto the LOCAL device (explicit\n"
    '    # index — fastsafetensors set_device rejects bare "cuda"), nogds\n'
    "    # bounce-buffer streaming (no GDS on GB10) → no host full-shard pileup.\n"
    "    pg = SingleGroup()\n"
    '    device = torch.device("cuda", torch.cuda.current_device())'
)

OLD2 = "        loader = SafeTensorsFileLoader(pg, device)"
NEW2 = "        loader = SafeTensorsFileLoader(pg, device, nogds=True)"


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD1, NEW1, what="fastsafetensors pg/device (SingleGroup + local device)")
    p.replace(OLD2, NEW2, what="fastsafetensors SafeTensorsFileLoader nogds=True")
