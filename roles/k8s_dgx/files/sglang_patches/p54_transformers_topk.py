"""[dgxarley] transformers.py: add Gemma-4 MoE config attribute names.

Patch sglang Transformers fallback: add Gemma-4 MoE config attribute names.

Gemma-4 (Gemma4ForConditionalGeneration) has no native SGLang implementation
and falls through to the Transformers backend. The MoEMixin.recursive_replace()
method looks up top_k via ("num_experts_per_tok", "top_k") — Gemma-4 uses
"top_k_experts" instead → AssertionError: Cannot determine top_k from config.

Fix: add "top_k_experts" to the _getattr_first lookup tuple on line 1197.

Note: SGLang v0.5.11 has native Gemma-4 support (PR #21952 + follow-ups
#22079, #24048, #22842), so this patch is a no-op there (the grep guard
inside makes it idempotent — pattern not found → skip).
"""

from _patchlib import Patch

patch = Patch(name="Gemma-4 MoE top_k lookup fallback attribute name", target="sglang/srt/models/transformers.py")

MARKER = "# [patch] _sgl_gemma4_topk_"

OLD_TUPLE = '("num_experts_per_tok", "top_k")'
NEW_TUPLE = '("num_experts_per_tok", "top_k", "top_k_experts")'

# Marker goes on a NEW line above to avoid breaking the closing paren of
# _getattr_first(...). This second edit runs against the buffered code AFTER
# the first edit above, so it matches against NEW_TUPLE (already substituted).
OLD_MARKER_LINE = "top_k = _getattr_first(text_config, " + NEW_TUPLE
NEW_MARKER_LINE = MARKER + "\n        " + OLD_MARKER_LINE


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD_TUPLE, NEW_TUPLE, marker=MARKER, what="top_k lookup tuple")
    p.replace(OLD_MARKER_LINE, NEW_MARKER_LINE, marker=MARKER, what="top_k lookup marker comment")
