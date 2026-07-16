"""[dgxarley] hf_transformers/tokenizer.py: load Mistral-native tokenizer.json offline.

[patch] _sgl_mistral_native_tokenizer_ — let get_tokenizer() load the shipped HF
tokenizer.json OFFLINE for Mistral-native checkpoints (no config.json).

Mistral-native repos (params.json + tekken.json + tokenizer.json, e.g.
Mistral-Small-4-119B-NVFP4) ship NO config.json. SGLang's ModelConfig + get_processor
already handle that via the name-triggered is_mistral_model()/load_mistral_config()
path, BUT get_processor's TokenizersBackend reload (and any DIRECT get_tokenizer call)
hits AutoTokenizer.from_pretrained → which loads config.json → offline OSError, crashing
tokenizer init before any weight load. Upstream only special-cases BARE-tekken
(tekken.json AND no tokenizer.json); these repos ALSO ship tokenizer.json, so the
upstream is_bare_tekken_checkpoint branch is skipped and upstream main fails identically
— an image bump does NOT fix this.

Fix: inject the config built from params.json (load_mistral_config) into common_kwargs
so AutoTokenizer (and _resolve_tokenizers_backend) skip the config.json fetch and load
the shipped tokenizer.json as a standard HF TokenizersBackend. This keeps the full HF
tokenizer API (add_special_tokens / apply_chat_template) that the pixtral MM processor
requires — routing to MistralCommonTokenizer instead would need the ~119-line
multimodal adaptation (add_special_tokens stub, pixtral markers) our image's
patch_mistral_common_tokenizer lacks. Verified in a debug pod (0.5.14-sm121): loads
offline, token IDs match the shipped tokenizer.json, pixtral add_special_tokens passes.
See reference_sglang_mistral_native_support.
"""

from _patchlib import Patch

patch = Patch(
    name="Mistral-native tokenizer.json offline config injection",
    target="sglang/srt/utils/hf_transformers/tokenizer.py",
)

MARKER = "_sgl_mistral_native_tokenizer_"

# Insert AFTER _resolve_tokenizer_name() (which applies _MISTRAL_TOKENIZER_REDIRECTS,
# e.g. Devstral) and BEFORE the AutoTokenizer try-block, so redirects still win.
OLD = """    common_kwargs = dict(
        trust_remote_code=trust_remote_code,
        tokenizer_revision=tokenizer_revision,
        clean_up_tokenization_spaces=False,
        **kwargs,
    )

    try:"""

NEW = """    common_kwargs = dict(
        trust_remote_code=trust_remote_code,
        tokenizer_revision=tokenizer_revision,
        clean_up_tokenization_spaces=False,
        **kwargs,
    )

    # [patch] _sgl_mistral_native_tokenizer_ — Mistral-native (params.json, tekken.json,
    # tokenizer.json, NO config.json) checkpoints make AutoTokenizer fetch config.json
    # -> offline OSError. Inject the config built from params.json so the shipped HF
    # tokenizer.json loads offline as a standard TokenizersBackend (full HF API the pixtral
    # MM processor needs). MistralCommonBackend would miss add_special_tokens here.
    from .mistral_utils import is_mistral_model, load_mistral_config
    if is_mistral_model(tokenizer_name) and "config" not in common_kwargs:
        common_kwargs["config"] = load_mistral_config(
            tokenizer_name, trust_remote_code=trust_remote_code, revision=tokenizer_revision
        )

    try:"""


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD, NEW, marker=MARKER, what="Mistral-native tokenizer config injection")
