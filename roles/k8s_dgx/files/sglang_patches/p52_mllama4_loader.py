"""[dgxarley] Llama-4 NVFP4 mllama4 loader patches (SGLang 0.5.15 / main gaps).

A modelopt-NVFP4 Llama-4 (Llama4ForConditionalGeneration) does NOT load without
these three fixes to models/mllama4.py. Verified against 0.5.15-sm121: all four
anchors match and the model then produces coherent text. Upstream-PR-worthy;
drop them on an image that ships the fixes (grep guards self-noop then).
Full rationale + re-sync steps: UPSTREAM_SGLANG_LLAMA4_NVFP4_BUG.md.
  1) _handle_expert_scale_params: NVFP4 ships a per-expert 3D block-scale
     [num_experts, in_blocks, out] under one (name-less) key. The FP8-era code
     broadcast a single 2D scale into every expert slot -> forces the whole 3D
     tensor into each 2D slot -> expand([16,512,5120],[5120,512]) crash. Fix:
     slice per expert + .T; 0-dim scalars / shared 2D scales keep broadcast.
  2) permute_qk_weight_for_rotary: the permute view used config.hidden_size
     (unpacked 5120), but the packed NVFP4 q/k weight last-dim is hidden/2 (2560)
     -> "shape [8,64,2,5120] invalid for input of size 2621440". Fix: w.shape[-1].
  3) permute_qk_weight_for_rotary: only .weight was row-permuted for rotary, the
     per-output-row weight_scale was NOT -> scale<->weight desync -> wrong q/k
     dequant. Fix: extend both branches to also permute the weight_scale.
(An optional 4th patch, KV-scale name-remap k_proj.k_scale->attn.k_scale, is
 quality-only; output stays coherent without it, so it is NOT included here.)
Serving Llama-4 also needs moe_runner_backend=triton + attention_backend=triton
in the model profile (flashinfer_cutlass asserts on apply_router_weight_on_input,
flashinfer is arg-rejected); that is profile config, not a loader patch.
NO model-name gate: mllama4.py is imported ONLY for the Llama4 arch, so this
patch is inert for every other model. Applying it unconditionally also covers
Llama-4 checkpoints served by a local path or a non-matching repo name, and the
per-anchor guards below keep it idempotent and version-safe.
"""

from _patchlib import Patch

patch = Patch(name="Llama-4 NVFP4 mllama4 loader fixes", target="sglang/srt/models/mllama4.py")

EDITS: list[tuple[str, str, str]] = [
    (
        "expert-3d-scale",
        """        else:
            # No expert ID found - this is a single scale for all experts
            # Load the same scale for all experts
            for expert_id in range(num_experts):
                param.data[expert_id] = loaded_weight""",
        """        else:
            if loaded_weight.dim() == 3:
                for expert_id in range(num_experts):
                    param.data[expert_id] = loaded_weight[expert_id].T
            else:
                for expert_id in range(num_experts):
                    param.data[expert_id] = loaded_weight""",
    ),
    (
        "permute-attn_out",
        "            attn_out = self.language_model.config.hidden_size",
        "            attn_out = w.shape[-1]",
    ),
    (
        "permute-k-scale",
        '        if ("wk" in modules or "k_proj" in modules) and modules[-1] == "weight":',
        '        if ("wk" in modules or "k_proj" in modules) and modules[-1] in ("weight", "weight_scale"):',
    ),
    (
        "permute-q-scale",
        '        elif ("wq" in modules or "q_proj" in modules) and modules[-1] == "weight":',
        '        elif ("wq" in modules or "q_proj" in modules) and modules[-1] in ("weight", "weight_scale"):',
    ),
]


@patch.run
def apply(p: Patch) -> None:
    for tag, old, new in EDITS:
        p.replace(old, new, what=tag)
