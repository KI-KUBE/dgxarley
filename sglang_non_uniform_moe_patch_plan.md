# SGLang Non-Uniform MoE (per-layer expert count) Patch, Implementation Plan

Plan only, no implementation. Written 2026-07-16 against image
`xomoxcc/dgx-spark-sglang:0.5.15-sm121` (all line anchors below verified
against the upstream `v0.5.15` tag; the sm121 image tracks it).

Target checkpoint: **`0xSero/GLM-5.2-REAP-NU176-526B`** (non-uniform REAP,
"NU" = per-layer expert budgets via water-filling instead of a uniform
top-168 cut). Sibling of the uniform `0xSero/glm-5.2-reap-504B-v2` we already
serve; that one needs NONE of this (uniform 168 loads on stock SGLang).
The DSA/SM121 patch stack (torch indexer, flashinfer_gather decode/prefill,
cuda-graph plan/run split) is ORTHOGONAL to this plan and carries over
unchanged; see `dsa_cuda_graph_plan.md` and the 504B-v2 profile's
patch-activation contract.

## 1. The problem

The NU176 checkpoint prunes a DIFFERENT number of experts per layer.
`config.json` (fetched + verified 2026-07-16):

- `architectures: ["GlmMoeDsaForCausalLM"]`, `model_type: glm_moe_dsa`
- `n_routed_experts: 216` , now only the MAXIMUM across layers
- `num_routed_experts_per_layer`: list of **79** ints, one per layer 0..78
  (78 decoder layers + 1 MTP/NextN layer):
  - layers 0-2: `0` (dense, `first_k_dense_replace: 3`)
  - layers 3-77: 108..216, mean ~176
  - layer 78 (MTP): 168
- everything else matches the 504B-v2 geometry (hidden 6144,
  moe_intermediate 2048, top-k 8, n_group=1, MHA-via-MLA 64 heads, DSA)

SGLang builds the whole DeepSeek-family MoE stack from the SCALAR
`config.n_routed_experts`. Verified: `num_routed_experts_per_layer` has
**zero occurrences** in sglang upstream (both `main` and the `v0.5.15` tag).
`GlmMoeDsaForCausalLM` is a thin subclass of `DeepseekV2ForCausalLM`
(`glm4_moe.py:1466`), so the entire model build + weight load goes through
`deepseek_v2.py` / `deepseek_common/deepseek_weight_loader.py`. The
`Glm4Moe*` classes in `glm4_moe.py` are the non-DSA GLM-4.x path and are NOT
involved.

Unpatched failure mode: every MoE layer gets built with 216 experts. First
crash is the router gate weight load, e.g. layer 3:
checkpoint `model.layers.3.mlp.gate.weight` is `[116, 6144]`, the param is
`torch.empty((216, 6144))`, `copy_` size mismatch. Same crash class as the
known NEXTN w13 6144-vs-3072 bug on the 504B. Even if the gate were padded,
`FusedMoE` would allocate 216 expert slots per layer (~40 phantom experts x
~19 MB NVFP4 each x 75 layers ~ 55 GB of garbage weights), so "pad to max"
is not a viable dodge; the per-layer count must reach the module builders.

## 2. Verified checkpoint facts (basis for the design)

Checked against `model.safetensors.index.json` (2026-07-16):

1. **Expert indices are contiguous and re-packed per layer**: every MoE layer
   `l` in 3..78 contains exactly `experts.0 .. experts.(k_l - 1)` where `k_l`
   equals `num_routed_experts_per_layer[l]`. NO holes, NO original-index
   remnants. This means weight loading needs no index remapping at all, only
   correctly-sized params.
2. Gate per MoE layer: `model.layers.N.mlp.gate.weight` AND
   `model.layers.N.mlp.gate.e_score_correction_bias` (topk_method noaux_tc),
   both sized `[k_l, ...]` / `[k_l]`.
3. Shared expert stays loose per layer (`mlp.shared_experts.{gate,up,down}_proj`),
   exactly like 504B-v2.
4. MTP layer 78 is present (`eh_proj`, `experts.0..167`), 168 experts.
5. On-disk size 321.6 GB (vs ~293 GiB for the 504B-v2): mind the JuiceFS
   cold-load time and per-node memory headroom notes in §8.

Reference: the model card ships a 12-line vLLM fork patch doing exactly this
("DeepseekV2MoE resolves each layer's expert count from the list,
n_routed_experts stays the max for compatibility; weight loading,
quantization and distributed serving work unchanged"). No patch FILE is
published (the HF repo only carries a prebuilt vLLM docker tar), so the
SGLang port below was derived from first principles against our tree.

## 3. Patch design

One module-level resolver + two touch points in `deepseek_v2.py`. Everything
is keyed on the presence of `config.num_routed_experts_per_layer`, so the
patch is INERT for every other model (the attribute exists only in NU
checkpoints; `getattr` default falls back to the scalar).

### 3.0 Resolver (new, module-level in `deepseek_v2.py`)

```python
def _nu_num_routed_experts(config, layer_id: int, is_nextn: bool = False) -> int:
    """[dgxarley] Non-uniform REAP support (0xSero NU cuts): resolve the
    per-layer routed-expert count from config.num_routed_experts_per_layer
    when present; fall back to the uniform scalar otherwise."""
    per_layer = getattr(config, "num_routed_experts_per_layer", None)
    if per_layer is None:
        return config.n_routed_experts
    # NextN builds its decoder with layer_id=0 (deepseek_nextn.py:149-151);
    # its real slot in the per-layer list is index num_hidden_layers.
    idx = config.num_hidden_layers if is_nextn else layer_id
    n = per_layer[idx]
    # Dense layers carry 0 in the list; DeepseekV2MoE is never constructed
    # for them (_is_layer_sparse gates on first_k_dense_replace), but guard
    # anyway so a misuse degrades to the uniform behaviour, not a 0-expert MoE.
    return n if n > 0 else config.n_routed_experts
```

### 3.1 `DeepseekV2MoE.__init__` (deepseek_v2.py:526)

`layer_id` and `is_nextn` are already parameters, so the resolution is local:

- At the top of `__init__`, resolve once:
  `self.n_routed_experts_layer = _nu_num_routed_experts(config, layer_id, is_nextn)`.
- **deepseek_v2.py:568-571** (the `else` branch, which is OUR path;
  the `_uses_per_rank_shared_slots` branch at 564 is DeepEP/MegaMOE only,
  not used with EP=1):
  `num_experts_for_moe = config.n_routed_experts + self.num_fused_shared_experts`
  -> `self.n_routed_experts_layer + self.num_fused_shared_experts`.
  (For completeness the 564 branch can get the same substitution; it is dead
  on our deployment but the patch then holds for EP users too.)
  This value feeds the `self.experts = get_moe_impl_class(...)` ctor at
  **deepseek_v2.py:618** (`num_experts=num_experts_for_moe + ep_num_redundant_experts`),
  which sizes the FusedMoE weight params, so NVFP4/modelopt param creation
  and the flashinfer_cutlass runner get the true per-layer count with no
  further changes.
- **deepseek_v2.py:593** pass the resolved count into the gate:
  `self.gate = MoEGate(..., num_routed_experts=self.n_routed_experts_layer)`.

Deliberately NOT touched in this method:
- tp-size sanity check (581): `tp_size > config.n_routed_experts` against the
  scalar max is fine (4 << 108 anyway).
- `HashTopK` branch (num_experts at ~638): `num_hash_layers` is 0 for GLM,
  branch never taken.
- The grouped-topk kwargs build NO explicit num_experts; router width comes
  from the gate weight and `correction_bias` (sized by the gate, see 3.2), so
  top-k output IDs are automatically in `0..k_l-1`.

### 3.2 `MoEGate.__init__` (deepseek_v2.py:434)

`MoEGate` has no `layer_id`, so it takes the resolved count as a new optional
kwarg (default `None` -> `config.n_routed_experts`, keeping every other call
site source-compatible):

- **deepseek_v2.py:449-451**:
  `torch.empty((config.n_routed_experts, config.hidden_size))`
  -> `torch.empty((num_routed_experts, config.hidden_size))`
- **deepseek_v2.py:462-464** (noaux_tc correction bias):
  `torch.empty((config.n_routed_experts), ...)` -> `torch.empty((num_routed_experts), ...)`

### 3.3 NextN / MTP (`deepseek_nextn.py`, no code change expected)

`DeepseekModelNextN` constructs `DeepseekV2DecoderLayer(config, 0, ...,
is_nextn=True)` (deepseek_nextn.py:149-151). The `is_nextn` flag flows into
`DeepseekV2MoE` and thus into the resolver, which indexes the list at
`num_hidden_layers` (78 -> 168 experts). So MTP support is handled entirely
by the resolver; `deepseek_nextn.py` itself needs no edit. NOTE: the NU176
is an NVFP4 export like the 504B-v2, so the existing
`_sgl_dsnextn_mixed_mtp_` launch patch (keep MTP experts NVFP4, attn/gate/
shared BF16) is expected to be needed here too; it composes, since it only
adjusts quant excludes, not expert counts. MTP stays OFF for bring-up
(§8), same one-variable-per-step policy as 504B-v2.

## 4. Why the weight loader needs NO change

`deepseek_weight_loader.py::do_load_weights` (the `DeepseekV2WeightLoaderMixin`
used by `DeepseekV2ForCausalLM.load_weights`, deepseek_v2.py:2875):

- **`FusedMoE.make_expert_params_mapping(num_experts=config.n_routed_experts + fused)`
  (weight_loader.py:176-181) can stay on the scalar 216.** The mapping is
  NAME-based: it enumerates candidate checkpoint names
  `...experts.{0..215}.{gate,up,down}_proj...`. Layers with fewer experts
  simply never present the higher names (§2 fact 1), and each present name
  loads into that layer's correctly-sized param at the same index. Building
  the mapping to the MAX is exactly what makes the scalar-as-max convention
  work (mirrors the vLLM patch's "n_routed_experts remains 216 for
  compatibility").
- **Shared-experts fusion is provably OFF for this model**: the allow-list in
  `determine_num_fused_shared_experts` (deepseek_v2.py:2761-2772) only
  enables fusion for `n_routed_experts in (256, 384)`; 216 disables it (same
  reason it is off for the 504B-v2's 168). So the
  `mlp.shared_experts -> mlp.experts.{n}` remap (weight_loader.py:221) is
  never reached and per-layer counts never collide with a fused slot index.
- w4afp8 input-scale mapping (weight_loader.py:187): quant is modelopt NVFP4,
  branch not taken.

## 5. Audited scalar usages that deliberately stay untouched

Full `n_routed_experts` audit of `deepseek_v2.py` @ v0.5.15; everything not
listed in §3 falls into one of these buckets:

| Anchor | What | Why untouched |
|---|---|---|
| 581-584 | tp_size > n_routed_experts guard | conservative with the max; TP=4 |
| 638 | HashTopK num_experts | num_hash_layers=0 for GLM |
| 793 | a2a/DeepEP TopK num_experts | DeepEP/mooncake backends only, EP=1 here |
| 2437, 2465 | `_use_aiter_gfx95` zero-allocator sizing | AMD-only branch |
| 2761-2772 | shared-experts-fusion allow-list | keeps fusion OFF, desired (§4) |
| 2893 | `get_model_config_for_expert_location` num_logical_experts | EPLB/expert-distribution metadata; only consumed with eplb/expert-recording enabled, which we never run. If EPLB is ever wanted on an NU cut, this needs a per-layer story upstream, out of scope. |
| `_is_layer_sparse` (2189-2193) | `n_routed_experts is not None` + first_k_dense_replace | boolean presence check only; the per-layer list's leading zeros agree with first_k_dense_replace=3 |

`glm4_moe.py` scalar usages (376, 379, 417, 432, ...) belong to the
`Glm4MoeForCausalLM` (non-DSA) path, not loaded for `GlmMoeDsaForCausalLM`:
untouched.

## 6. Delivery: new patch block in `roles/k8s_dgx/files/sglang_launch.sh`

Same conventions as the existing blocks (PATCH_DSA_*, `_sgl_dsnextn_mixed_mtp_`):

- One `python - <<'EOF'` block, marker name suggestion `PATCH_NU_MOE`
  (idempotency probe: `if "_nu_num_routed_experts" in code: skip`).
- Real-anchor string replacement with an `ANCHOR-DRIFT:` warning line when an
  anchor is missing (SGLang version drift detection), per house style.
- Three edits, smallest-possible anchors:
  1. insert the resolver after the `class MoEGate` import region (anchor on
     `class MoEGate(nn.Module):`, insert BEFORE it),
  2. `MoEGate.__init__` signature + the two `torch.empty` shapes
     (anchors: `torch.empty((config.n_routed_experts, config.hidden_size))`
     and `torch.empty((config.n_routed_experts), dtype=correction_bias_dtype)`,
     both unique in the file),
  3. `DeepseekV2MoE.__init__`: the `num_experts_for_moe` else-branch, the
     `MoEGate(` call, and the resolved-count line (anchor on
     `self.gate = MoEGate(`).
- Finish with `py_compile` of `deepseek_v2.py` like the other blocks.
- Leave breadcrumb comments `# [dgxarley] NU-REAP per-layer experts` at each
  edit (house rule: no silent divergence from upstream).

Activation contract (mirrors the DSA stack's): the patch is applied
unconditionally at launch but only CHANGES behaviour for checkpoints whose
config carries `num_routed_experts_per_layer`. For every other model the
resolver returns the scalar and the built graph is bit-identical to stock.
No profile knob needed, no gating env var.

## 7. New model profile (follow-up, separate file)

`roles/k8s_dgx/model_profiles/0xsero-glm-5.2-reap-nu176-526b.yml`, cloned
from the 504B-v2 profile with:

- `num_experts: 216` (the max; only used for launch-arg plumbing) and the
  same TP=4 / EP=1 / modelopt_fp4 / trust_remote_code base.
- Bring-up backend combo: start from the DENSE known-good baseline
  (`attention_backend: flashinfer`, no dsa_* keys) to isolate the NU patch;
  switch to the DSA-sparse combo only after NU is proven serving (the
  patch-activation contract in the 504B-v2 profile applies verbatim).
- `speculative_enabled: false` for bring-up (§3.3).
- KV budget: ~28.6 GB more total weight than 504B-v2 (321.6 vs ~293 GiB on
  disk, ~+7 GiB per node at TP=4), so expect `max_total_tokens: 131072` to
  need re-derivation from the measured post-load avail_mem; do NOT copy the
  0.99 mem_fraction + 128k pairing blind.
- Sampler guardrail: card recommends the same min_p/repetition_penalty family
  plus "hesitation-marker penalties" (arXiv:2606.00206) against quantized-
  reasoning overthinking; evaluate once serving, keep 504B-v2's
  `recommended_sampling` as the starting point.

## 8. Validation plan (in order, stop at first failure)

1. **Static**: `bash -n sglang_launch.sh`; apply the patch in a GB10 debug pod
   (`tail -f /dev/null`, NOT labelled app=sglang) against the real image,
   `py_compile` deepseek_v2.py, then an import + unit probe:
   instantiate a dummy config with the real 79-entry list and assert
   `_nu_num_routed_experts(cfg, 3) == 116`, `(cfg, 13) == 216`,
   `(cfg, 0, is_nextn=True) == 168`, and scalar fallback without the attr.
2. **Anchor audit on image bump**: the three anchors of §6 re-checked
   (standing rule for every launch-script patch).
3. **Weight-load smoke** (the real gate): boot TP=4 dense-attention profile;
   success criterion is `Load weight end` with no shape-mismatch crash and
   per-layer expert counts logged (add a one-line rank0 log in the patch:
   `layer {id}: {k} routed experts` when the NU path is active).
4. **Functional**: GSM8K-100 vs the 504B-v2 baseline (same harness as the DSA
   validation runs), finish_reason + pattern-grep + token-distribution checks
   per house rule (no "output coherent" claims from finish_reason alone).
5. **Then, separately**: flip to the DSA-sparse combo, then MTP
   (`_sgl_dsnextn_mixed_mtp_` interplay), one variable per step.

No deploy without explicit go (standing rule); steps 3-5 each need approval.

## 9. Open questions / risks

- **flashinfer_cutlass with per-layer-varying expert counts**: expected fine
  (kernel takes tensor shapes per layer; the vLLM fork serves the same
  checkpoint with per-layer counts through its cutlass path), but this is the
  first NU model on our stack: watch the first decode for expert-id
  out-of-range asserts. Fallback lever: `moe_runner_backend: triton`.
- **CUDA-graph capture**: per-layer counts are STATIC per layer, so capture
  should be unaffected; keep `disable_cuda_graph: true` for step 3 anyway to
  keep the bring-up crash surface minimal.
- **hf_preload / JuiceFS**: 321.6 GB over the USB-HDD-backed JuiceFS is a
  ~15h+ cold pull (known bottleneck, see memory on the intenso backend);
  schedule the download before any test window, with `HF_HUB_DISABLE_XET=1`
  if the hex-hash bug resurfaces.
- **Upstreaming**: this is a clean, generic feature (any non-uniform REAP cut
  of a DeepSeek-family model benefits). Consider an upstream PR after local
  validation; the launch-script patch then becomes a cherry-pick carrier like
  the Qwen3.6 PR #27906 pattern.
