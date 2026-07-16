"""[dgxarley] transformers/models/deepseek_v3/configuration_deepseek_v3.py: widen kv_lora_rank to Optional.

NOTE target package: this patches the upstream `transformers` (HuggingFace)
package, NOT sglang. DIST_PACKAGES-relative target resolution in _patchlib
handles that transparently — the path below is simply outside `sglang/`.

Patch DeepseekV3Config.kv_lora_rank to allow None for the DeepSeek-V4-Flash
variant (config.json has kv_lora_rank: null — Flash uses q-LoRA + o-LoRA + GQA,
NO MLA KV compression). The class sglang actually instantiates for model_type
"deepseek_v4" is _DeepseekV4ConfigAlias in sglang/srt/utils/hf_transformers/
common.py, which SUBCLASSES transformers' DeepseekV3Config — so the strict
dataclass field `kv_lora_rank: int` (and its huggingface_hub @strict validator)
is declared in transformers/models/deepseek_v3/configuration_deepseek_v3.py,
NOT in sglang's own configs/deepseek_v4.py (that file is never used for this
model_type). Under transformers 5.x the null value fails at startup with:
  StrictDataclassFieldValidationError: Field 'kv_lora_rank' expected int,
  got NoneType (value: None)
Widening the annotation to `int | None` BEFORE import makes @strict build a
Union validator that accepts None. This is a SAFE widening: DeepSeek-V3 / V3.2
/ Kimi-K2 (the other models sharing this config) always supply an int, so they
are unaffected; only V4-Flash's null now passes. We keep None rather than
coercing to an int (an int would push modeling onto the MLA KV-LoRA path the
Flash weights don't have). pyc is timestamp-invalidated, so the edit takes on
reimport. NOTE: clears the config-parse blocker only — Flash serving may still
hit further upstream issues downstream (sglang #25165 / #23743).
"""

from _patchlib import Patch

patch = Patch(
    name="DeepseekV3Config.kv_lora_rank Optional widening (DeepSeek-V4-Flash)",
    target="transformers/models/deepseek_v3/configuration_deepseek_v3.py",
)

OLD = "    kv_lora_rank: int = 512"
NEW = "    kv_lora_rank: int | None = 512"


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD, NEW, what="kv_lora_rank Optional widening")
