"""[dgxarley] parser/reasoning_parser.py: HunyuanDetector reasoning think/tool tokens via resolve_hunyuan_tokens().

Hunyuan (Hy3 / HYV3) special-token suffix backport -- SGLang PR #29920.
------------------------------------------------------------------------
This image predates PR #29920 (merged 2026-07-04, "resolve special-token
suffix at runtime"). Its hunyuan tool-call AND reasoning detectors HARDCODE
the bare structural tokens (<think>, <tool_calls>, <tool_call>, <tool_sep>,
<arg_key>, <arg_value>). The shipping Hy3 checkpoints (vroomfondel/Hy3-NVFP4-
W4A4, tencent/Hy3, ...) append a shared suffix to EVERY such token -- the
chat template's HYTK, mirrored by tokenizer_config.json "token_suffix", e.g.
":opensource" so the model emits <think:opensource>, <tool_calls:opensource>,
... (verified: these are single special tokens; the BARE forms are not tokens
at all). With the bare-token detectors, the reasoning split never fires and NO
tool calls are parsed -> breaks honcho/hindsight function-calling.

Faithful backport of PR #29920's resolve_hunyuan_tokens() into the two detector
files (function_call/hunyuan_detector.py in p62_hunyuan_tool_parser.py, and this
one). Upstream threads the tokenizer through ~8 caller files to feed the vocab
to resolve_hunyuan_tokens(); this image threads no tokenizer, so we instead feed
the resolved suffix via the SGLANG_HUNYUAN_TOKEN_SUFFIX env var -- exported by
sglang_launch.sh (read from the model's tokenizer_config.json "token_suffix"
there, BEFORE the patch runner executes this file). Empty suffix (preview
checkpoints, or a non-Hy3 model) -> bare tokens = unchanged upstream-preview
behavior.

RE-SYNC: when bumping to an image that already contains PR #29920, this patch
becomes a no-op by itself -- the "resolve_hunyuan_tokens" grep guard below
already skips it -- but it's still fine to delete once confirmed.

This is sub-patch 2 of 2 (see p62_hunyuan_tool_parser.py for sub-patch 1). A
third, UNRELATED patch colocated under the same Hy3/Hunyuan model gate in the
old script (shared-expert weight remap) now lives in
p64_hunyuan_shared_experts.py.

Patched here: HunyuanDetector reasoning think/tool tokens resolved via the same
backported helper as p62.

RE-CHECKED 2026-07-16: PR #29920 has LANDED upstream on this image --
HunyuanDetector now natively does `t = resolve_hunyuan_tokens(tokenizer)` and
builds think_open/think_close/tool_start_token from it (verified in the live
0.5.15-sm121 source), i.e. exactly what this sub-patch injects. The old
"already applied" check only recognized OUR OWN local-alias import
(`resolve_hunyuan_tokens as _resolve_hunyuan_tokens`, hence the
leading-underscore substring match) and did not recognize upstream's unaliased
native call, so it fell through to the (now-stale) `old` anchor and reported a
false ANCHOR-DRIFT even though there is nothing left to patch. Fixed to match
on the bare function name, same idiom sub-patch (1) above already uses
successfully -- reproduced below as the up-front `"resolve_hunyuan_tokens" in
p.code` check in `apply()` (not a `marker=` on `p.replace()`, for the same
reason p62 explains: this image's native __init__ signature/anchor differs
enough from OLD below that the anchor wouldn't match anyway, but the up-front
check turns that into a calm "already applied" skip instead of a scary, and in
this case wrong, ANCHOR-DRIFT).

[moved 2026-07-16] Was the inline `python3 << 'PATCH_HUNYUAN_REASON_EOF'`
heredoc inside a bash `if [[ $SGLANG_MODEL == *Hy3* || $SGLANG_MODEL == *Hunyuan*
|| $SGLANG_TOOL_CALL_PARSER == "hunyuan" || $SGLANG_REASONING_PARSER == "hunyuan"
]]` gate. That gate is now `when=gate_model("Hy3", "Hunyuan") or gate_env(...)`.
"""

from _patchlib import Patch, gate_env, gate_model

patch = Patch(
    name="HunyuanDetector reasoning think/tool tokens via resolve_hunyuan_tokens()",
    target="sglang/srt/parser/reasoning_parser.py",
    when=gate_model("Hy3", "Hunyuan")
    or gate_env("SGLANG_TOOL_CALL_PARSER", "hunyuan")
    or gate_env("SGLANG_REASONING_PARSER", "hunyuan"),
)

OLD = r"""        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<tool_calls>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )"""

NEW = r"""        # [patch] _sgl_hunyuan_token_suffix_ — backport of SGLang PR #29920
        from sglang.srt.function_call.hunyuan_detector import (
            resolve_hunyuan_tokens as _resolve_hunyuan_tokens,
        )

        _hy = _resolve_hunyuan_tokens()
        _think_open = _hy["think"]
        _think_close = (
            "</" + _think_open[1:] if _think_open.startswith("<") else _think_open
        )
        super().__init__(
            _think_open,
            _think_close,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token=_hy["tool_calls"],
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )"""


@patch.run
def apply(p: Patch) -> None:
    if "resolve_hunyuan_tokens" in p.code:
        # Either our own previous run, or (verified on 0.5.15-sm121) upstream
        # already ships PR #29920 natively -- nothing left to patch either way.
        return
    p.replace(OLD, NEW, what="HunyuanDetector suffixed think/tool tokens")
