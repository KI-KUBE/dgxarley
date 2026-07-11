#!/usr/bin/env python3
# ============================================================================
# Localise the FIRST inf/nan in a Hy3-W4A4 forward from SGLang's per-layer
# tensor dump (--debug-tensor-dump-output-folder).
#
# Dump layout (SGLang debug_utils/tensor_dump_forward_hook.py):
#   <base>/TP{r}_PP{p}_Rank{n}_pid{pid}/Pass{NNNNN}.pt
#   each .pt = OrderedDict{ operator_name -> output_tensor }  (forward order)
#
# Run inside a pod/box that has torch AND the dump mounted, e.g.:
#   python3 hy3_dump_find_first_inf.py /root/.cache/huggingface/tensor_dump
# ============================================================================
import sys, os, glob
import torch

base = sys.argv[1] if len(sys.argv) > 1 else "/root/.cache/huggingface/tensor_dump"
rank_dirs = sorted(glob.glob(os.path.join(base, "TP*_PP*_Rank*")))
if not rank_dirs:
    print(f"no rank dirs under {base} (did the dump run? check the folder)")
    sys.exit(1)


def bad(t: object) -> bool:
    if not torch.is_tensor(t):
        return False
    tf = t.float()
    return bool(torch.isinf(tf).any() or torch.isnan(tf).any())


for rd in rank_dirs:
    print(f"\n===== {os.path.basename(rd)} =====")
    passes = sorted(glob.glob(os.path.join(rd, "Pass*.pt")))
    if not passes:
        print("  (no Pass*.pt)")
        continue
    for pf in passes:
        try:
            d = torch.load(pf, map_location="cpu")
        except Exception as e:
            print(f"  {os.path.basename(pf)}: LOAD FAIL {e}")
            continue
        items = list(d.items())
        first_bad = None
        n_bad = 0
        for name, t in items:
            if bad(t):
                n_bad += 1
                if first_bad is None:
                    first_bad = name
        tag = os.path.basename(pf)
        if first_bad is None:
            print(f"  {tag}: {len(items)} ops, all finite ✓")
        else:
            # show the op right BEFORE the first bad one (the likely culprit's input was still finite)
            idx = [n for n, _ in items].index(first_bad)
            prev = items[idx - 1][0] if idx > 0 else "<none>"
            print(f"  {tag}: {len(items)} ops, {n_bad} inf/nan")
            print(f"      FIRST inf/nan op : {first_bad}")
            print(f"      preceding op     : {prev}  (its output was still finite -> the fault is IN {first_bad})")
