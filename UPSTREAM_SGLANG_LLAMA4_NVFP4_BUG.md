# Upstream Bug: SGLang can't serve modelopt-NVFP4 Llama-4 without local patches

## Status

**Open upstream, 5 local patches in place** (added 2026-07-13). A modelopt-NVFP4
Llama-4 checkpoint (`Llama4ForConditionalGeneration`, e.g.
`nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4`) does NOT load/serve on stock SGLang
0.5.15 (bug also on `main`). Five source patches are applied at container start by
`roles/k8s_dgx/files/sglang_launch.sh`, in two adjacent ungated blocks
(`PATCH_MLLAMA4_LOADER_EOF` = 3 load-blockers, `PATCH_MLLAMA4_KVSCALE_EOF` = 2
KV-scale quality). All are upstream-PR-worthy; drop them on an image that ships the
fixes (the per-anchor grep guards self-noop then).

**Ungated on purpose:** `models/llama4.py` and `models/mllama4.py` are imported
ONLY for the Llama4 arch, so patching them is inert for every other model; ungated
also covers Llama-4 checkpoints served by a local path or a non-matching repo name.
Idempotency + version-safety come from the per-anchor guards (an `ANCHOR NOT FOUND`
print on an image drift, not a silent unpatched serve).

**Profile side (NOT patches, config in the model profile):** serving Llama-4 also
needs `attention_backend: triton` (flashinfer is allowlist-rejected for Llama4, same
as Gemma-4) and `moe_runner_backend: triton` (flashinfer_cutlass asserts on
`apply_router_weight_on_input=True`, Llama-4's top-1 input-routing). See
`roles/k8s_dgx/model_profiles/nvidia-llama-4-scout-17b-16e-instruct-nvfp4.yml`.

## Affected

| | |
|---|---|
| SGLang | 0.5.15 (verified) and `main` (per patch author) |
| Image | `xomoxcc/dgx-spark-sglang:0.5.15-sm121` (validated), also 0.5.14-sm121 gaps |
| Files | `sglang/srt/models/mllama4.py`, `sglang/srt/models/llama4.py` |
| Class | `Llama4ForConditionalGeneration` (multimodal: text MoE + vision) |

## The patches

### Block 1: loader (3 fixes, `mllama4.py`) - WITHOUT THESE THE MODEL DOES NOT LOAD

1. **`_handle_expert_scale_params`: fused 3D NVFP4 block-scale.** NVFP4 ships a
   per-expert 3D block-scale `[num_experts, in_blocks, out]` under one (name-less)
   key. The FP8-era code broadcasts a single 2D scale into every expert slot, which
   forces the whole 3D tensor into each 2D slot: `expand([16,512,5120],[5120,512])`
   crash. Fix: `if loaded_weight.dim() == 3` -> slice per expert + `.T`; 0-dim
   scalars (weight_scale_2 / input_scale) and shared 2D scales keep the broadcast.

2. **`permute_qk_weight_for_rotary`: real last-dim, not unpacked hidden.** The
   permute view used `config.hidden_size` (unpacked 5120), but the packed NVFP4 q/k
   weight last-dim is `hidden/2` (2560): `shape [8,64,2,5120] invalid for input of
   size 2621440`. Fix: `attn_out = w.shape[-1]`.

3. **`permute_qk_weight_for_rotary`: permute the weight_scale too.** Only `.weight`
   was row-permuted for rotary, not the per-output-row `weight_scale`, causing a
   scale<->weight desync and wrong q/k dequant. Fix: extend both the k and q
   branches from `modules[-1] == "weight"` to
   `modules[-1] in ("weight", "weight_scale")`.

### Block 2: KV-scale (2 fixes, QUALITY only - serves without them, at scale 1.0)

The modelopt-NVFP4 checkpoint bakes FP8 KV scales (`...k_proj.k_scale` /
`...v_proj.v_scale`). Without these two SGLang never loads them and falls back to
scale 1.0 ("less accurate results"). Both are needed (A alone is a no-op). Verified:
0 not-loaded warnings, and the triton attn backend uses the loaded scale
(`cache_k.div_(k_scale)`).

- **A) `llama4.py`: `RadixAttention` built without `quant_config`.** `Llama4Attention`
  created `self.attn = RadixAttention(...)` without `quant_config` (unlike
  `llama.py` / `qwen2.py`), so `create_weights` never ran and `k_scale`/`v_scale`
  stayed plain `None` attrs, never in `named_parameters()`. Fix: pass
  `quant_config=quant_config` (it is already in scope, used for q/k/v/o).

- **B) `mllama4.py`: `_handle_scale_remapping` never copied.** The function returned
  a bool but never copied the remapped scale into the param (the regular `llama.py`
  loader does; mllama4's reimplementation dropped the copy step). Fix: thread
  `loaded_weight` into the signature + call `self._handle_default_weight(...)` on the
  remapped name (prereq `_handle_default_weight` exists in mllama4.py). Two edits:
  signature/body + the call site.

## Not included (optional)

None outstanding. (An earlier "4th patch" idea, a KV-scale name-remap, turned out to
be exactly Block 2 above.) A custom pythonic-tool chat template is a possible future
add: this image ships NO `tool_chat_template_llama4_pythonic.jinja`, but a built-in
template NAME `llama-4` is registered (`sglang/lang/chat_template.py`) and is
selectable via the profile's `chat_template` knob (wired to `--chat-template`
2026-07-13). The Scout profile leaves `chat_template: ""` (use the tokenizer's own
template) until tool calls are shown to need the built-in.

## Verification (2026-07-13, image 0.5.15-sm121)

- Anchor dry-run: all 7 anchors (4 loader + 3 KV) match verbatim.
- End-to-end apply in a throwaway container: all 7 report `APPLIED`; both
  `llama4.py` and `mllama4.py` `ast.parse` cleanly after patching; second pass is
  idempotent (`already applied`).
- Patch author confirmed: model loads and produces coherent text; KV block gives 0
  not-loaded warnings.
- `bash -n sglang_launch.sh`: clean.
- NOT yet done: a live cluster serve at `tp_size=4` (throughput unmeasured); this
  was a functional TP=1 bring-up.

## Re-sync on image bump

When bumping the SGLang image, the `ANCHOR NOT FOUND (SGLang version drift?)` prints
in the pod log flag that an anchor moved. Re-fetch the two files, re-derive the
anchors, and update the two heredoc blocks in `sglang_launch.sh`. Verify with the
`podman run ... full_verify.py` pattern (all 7 `APPLIED` + `AST_OK`) before serving.

## Upstream tracking

- The 3 loader fixes + 2 KV-scale fixes are PR-worthy against `sgl-project/sglang`.
  **TODO:** file (needs sign-off before posting outward).
- Related: TensorRT-LLM PR #3492 (added `apply_router_weight_on_input` for Llama4
  FusedMoE) is the analogous upstream fix that motivates the profile's triton
  moe_runner requirement. SGLang issue #7994 ("Support NVFP4 masked layout MoE").

## Changelog

- **2026-07-13** - First bring-up. 3 loader patches + 2 KV-scale patches added to
  `sglang_launch.sh` (ungated); `--chat-template` wired from the profile; Scout
  profile set to triton/triton backends. Load + coherent text confirmed at TP=1;
  4-node throughput still unmeasured.
