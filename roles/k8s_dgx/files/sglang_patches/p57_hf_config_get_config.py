"""[dgxarley] hf_transformers/config.py: convert dict sub_configs after loading (transformers 5.5.0 bug).

Patch SGLang get_config() to convert dict sub_configs after loading (transformers 5.5.0 bug).
Transformers 5.x auto-generates __init__ for PretrainedConfig subclasses with sub_configs,
bypassing dict→config conversion. from_pretrained() also bypasses __post_init__.
Both vision_config and text_config arrive as raw dicts → AttributeError on .hidden_size etc.
Fix: patch get_config() to convert dict sub-configs after loading for any config with sub_configs.

RE-ANCHORED 2026-07-16: sglang/srt/utils/hf_transformers_utils.py is now a bare
backward-compat shim ("all code has moved to sglang.srt.utils.hf_transformers") — it
has no get_config()/"return config" left, so the old anchor silently never matched on
this image. The real get_config() lives in sglang/srt/utils/hf_transformers/config.py.
The underlying transformers-5.x sub_configs bug is STILL UNSOLVED there: only the
Mistral parser path calls the sglang-native _ensure_sub_configs() helper (for
text_config/vision_config); the generic "hf" parser path (used by everything else,
incl. GLM/DeepSeek/NemotronH) does not call it anywhere — verified by grepping the
whole hf_transformers package. So this patch still applies, just at the new home.
"""

from _patchlib import Patch

patch = Patch(
    name="get_config() sub_configs dict->config conversion", target="sglang/srt/utils/hf_transformers/config.py"
)

MARKER = "sub_configs dict fix"

# Find the final "return config" in get_config() and add sub_configs conversion before it.
# The gguf branch was refactored (model_type/config.update -> _set_architectures helper)
# since this patch was first written; anchor matches the current shape.
OLD = """    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        _set_architectures(config, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type])

    return config"""

NEW = """    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        _set_architectures(config, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type])

    # [patch] sub_configs dict fix — transformers 5.x from_pretrained() leaves sub-configs
    # as raw dicts instead of converting to their declared config classes.
    _sub_cfgs = getattr(config, "sub_configs", None)
    if _sub_cfgs:
        for _key, _cls in _sub_cfgs.items():
            _val = getattr(config, _key, None)
            if isinstance(_val, dict):
                try:
                    setattr(config, _key, _cls(**_val))
                except Exception:
                    pass  # non-critical: some sub-configs may not accept all dict keys

    # [patch] Qwen3.5 MoE: text_config lacks norm_topk_prob (Qwen2MoeSparseMoeBlock expects it).
    # Qwen3.5 uses softmax routing — renormalize=True is correct default.
    _tc = getattr(config, "text_config", None)
    if _tc is not None and not isinstance(_tc, dict) and not hasattr(_tc, "norm_topk_prob"):
        _tc.norm_topk_prob = True

    return config"""


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD, NEW, marker=MARKER, what="get_config sub_configs dict fix")
