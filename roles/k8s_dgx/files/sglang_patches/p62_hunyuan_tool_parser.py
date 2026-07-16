"""[dgxarley] function_call/hunyuan_detector.py: resolve_hunyuan_tokens() + suffixed tool-call tokens.

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
files (this one, and reasoning_parser.py in p63_hunyuan_reasoning_parser.py).
Upstream threads the tokenizer through ~8 caller files to feed the vocab to
resolve_hunyuan_tokens(); this image threads no tokenizer, so we instead feed
the resolved suffix via the SGLANG_HUNYUAN_TOKEN_SUFFIX env var -- exported by
sglang_launch.sh (read from the model's tokenizer_config.json "token_suffix"
there, BEFORE the patch runner executes this file, so it is present by the
time resolve_hunyuan_tokens() below reads it at call time). Empty suffix
(preview checkpoints, or a non-Hy3 model) -> bare tokens = unchanged
upstream-preview behavior.

RE-SYNC: when bumping to an image that already contains PR #29920, this patch
becomes a no-op by itself -- the "resolve_hunyuan_tokens" grep guard below
already skips it -- but it's still fine to delete once confirmed.

This is sub-patch 1 of 2 (see p63_hunyuan_reasoning_parser.py for sub-patch 2,
parser/reasoning_parser.py -- HunyuanDetector reasoning think/tool tokens
resolved via this same backported helper). A third, UNRELATED patch colocated
under the same Hy3/Hunyuan model gate in the old script (shared-expert weight
remap) now lives in p64_hunyuan_shared_experts.py.

Patched here: resolve_hunyuan_tokens + suffixed tool-call tokens (bot/eot/
tool_call/tool_sep/arg_key/arg_value + the two regexes + structure_info).

[moved 2026-07-16] Was the inline `python3 << 'PATCH_HUNYUAN_TOOL_EOF'` heredoc
inside a bash `if [[ $SGLANG_MODEL == *Hy3* || $SGLANG_MODEL == *Hunyuan* ||
$SGLANG_TOOL_CALL_PARSER == "hunyuan" || $SGLANG_REASONING_PARSER == "hunyuan" ]]`
gate. That gate is now `when=gate_model("Hy3", "Hunyuan") or gate_env(...)` below.
The outer `if "resolve_hunyuan_tokens" in code: ... else: ...` guard from the
heredoc is reproduced as an explicit up-front check in `apply()` rather than as
a `marker=` on the individual `p.replace()` calls: the injected HELPER text
itself contains the string "resolve_hunyuan_tokens" (it defines the function),
so using that as a per-edit `marker` would make the __init__ and structure_info
edits silently no-op the moment the helper edit lands -- same half-patched
failure mode _patchlib.py's replace_all() docstring warns about, just via a
`marker` collision instead of a `replace()` vs `replace_all()` mixup. Checking
it once, up front, keeps the three edits atomic (as the original `missing = [...]`
pre-check did) while still being cheap to skip when this image already ships a
native (if differently-shaped) resolve_hunyuan_tokens -- verified against the
0.5.15-sm121 pristine source: hunyuan_detector.py already HAS PR #29920 natively
(same resolve_hunyuan_tokens() call sites, minor implementation differences e.g.
type-hinted `_close`), so on the current image this patch always hits the
"already present" branch and is a no-op; it stays gated on the model/parser
condition (not deleted) per the RE-SYNC note above.
"""

from _patchlib import Patch, gate_env, gate_model

patch = Patch(
    name="resolve_hunyuan_tokens() + suffixed tool-call tokens",
    target="sglang/srt/function_call/hunyuan_detector.py",
    when=gate_model("Hy3", "Hunyuan")
    or gate_env("SGLANG_TOOL_CALL_PARSER", "hunyuan")
    or gate_env("SGLANG_REASONING_PARSER", "hunyuan"),
)

ANCHOR = "logger = logging.getLogger(__name__)\n"

HELPER = r'''
# [patch] _sgl_hunyuan_token_suffix_ — backport of SGLang PR #29920
import os as _os

_HUNYUAN_TOKEN_NAMES = (
    "tool_calls",
    "tool_call",
    "tool_sep",
    "arg_key",
    "arg_value",
    "think",
)

_HUNYUAN_TOKEN_RE = re.compile(
    r"^<(?P<name>" + "|".join(_HUNYUAN_TOKEN_NAMES) + r")(?::[^>]+)?>$"
)


def resolve_hunyuan_tokens(tokenizer=None):
    """Map bare token names to their real (possibly suffixed) strings.

    Prefers suffixed forms in the tokenizer vocab; when no tokenizer is threaded
    (this image predates PR #29920's caller plumbing), falls back to the
    launch-provided SGLANG_HUNYUAN_TOKEN_SUFFIX; finally to the bare literal.
    """
    tokens = {}
    vocab = None
    if tokenizer is not None:
        try:
            vocab = tokenizer.get_vocab()
        except Exception as e:
            logger.warning("Failed to read Hunyuan tokenizer vocab: %s", e)
            vocab = None
    if isinstance(vocab, dict):
        for tok in vocab:
            if not isinstance(tok, str):
                continue
            m = _HUNYUAN_TOKEN_RE.match(tok)
            if m:
                tokens[m.group("name")] = tok
    _suffix = _os.environ.get("SGLANG_HUNYUAN_TOKEN_SUFFIX", "")
    for name in _HUNYUAN_TOKEN_NAMES:
        tokens.setdefault(name, "<" + name + _suffix + ">")
    return tokens

'''

OLD_INIT = r"""    def __init__(self):
        super().__init__()

        self.bot_token = "<tool_calls>"
        self.eot_token = "</tool_calls>"

        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_sep_token = "<tool_sep>"

        self.arg_key_start_token = "<arg_key>"
        self.arg_key_end_token = "</arg_key>"
        self.arg_value_start_token = "<arg_value>"
        self.arg_value_end_token = "</arg_value>"

        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)<tool_sep>(.*?)</tool_call>", re.DOTALL
        )
        self.func_args_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
        )"""

NEW_INIT = r"""    def __init__(self, tokenizer=None):
        super().__init__()

        t = resolve_hunyuan_tokens(tokenizer)
        tool_calls = t["tool_calls"]
        tool_call = t["tool_call"]
        tool_sep = t["tool_sep"]
        arg_key = t["arg_key"]
        arg_value = t["arg_value"]

        def _close(open_tok):
            return "</" + open_tok[1:] if open_tok.startswith("<") else open_tok

        self.bot_token = tool_calls
        self.eot_token = _close(tool_calls)
        self.tool_call_start_token = tool_call
        self.tool_call_end_token = _close(tool_call)
        self.tool_sep_token = tool_sep
        self.arg_key_start_token = arg_key
        self.arg_key_end_token = _close(arg_key)
        self.arg_value_start_token = arg_value
        self.arg_value_end_token = _close(arg_value)

        tc_end = _close(tool_call)
        ak_end = _close(arg_key)
        av_end = _close(arg_value)
        self.tool_call_regex = re.compile(
            re.escape(tool_call)
            + r"(.*?)"
            + re.escape(tool_sep)
            + r"(.*?)"
            + re.escape(tc_end),
            re.DOTALL,
        )
        self.func_args_regex = re.compile(
            re.escape(arg_key)
            + r"(.*?)"
            + re.escape(ak_end)
            + r"\s*"
            + re.escape(arg_value)
            + r"(.*?)"
            + re.escape(av_end),
            re.DOTALL,
        )"""

OLD_STRUCTURE_INFO = r"""        return lambda name: StructureInfo(
            begin=f"<tool_calls>\n<tool_call>{name}<tool_sep>",
            end="</tool_call>\n</tool_calls>",
            trigger="<tool_calls>",
        )"""

NEW_STRUCTURE_INFO = r"""        return lambda name: StructureInfo(
            begin=f"{self.bot_token}\n{self.tool_call_start_token}{name}{self.tool_sep_token}",
            end=f"{self.tool_call_end_token}\n{self.eot_token}",
            trigger=self.bot_token,
        )"""


@patch.run
def apply(p: Patch) -> None:
    if "resolve_hunyuan_tokens" in p.code:
        # Either our own previous run, or (verified on 0.5.15-sm121) upstream
        # already ships PR #29920 natively -- nothing left to patch either way.
        return
    p.replace(ANCHOR, ANCHOR + HELPER, what="resolve_hunyuan_tokens() helper")
    p.replace(OLD_INIT, NEW_INIT, what="__init__ suffixed tool-call tokens")
    p.replace(OLD_STRUCTURE_INFO, NEW_STRUCTURE_INFO, what="structure_info suffixed tokens")
