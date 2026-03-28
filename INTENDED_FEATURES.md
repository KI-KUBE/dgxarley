# Intended Features

Planned improvements and features that aren't urgent but worth tracking.

## MiniMax-M2.5 ThinkingBudgetLogitProcessor

**Status**: Not started
**Effort**: Low (3-line subclass + newline token lookup)
**Upstream PR potential**: Yes — SGLang has no MiniMax thinking budget processor

SGLang's `thinking_budget` parameter is currently **non-functional** with MiniMax-M2.5 because the `Qwen3ThinkingBudgetLogitProcessor` hardcodes Qwen3's token IDs (151667/151668), not MiniMax's (200050/200051).

**What's needed:**

1. Look up the newline token ID from MiniMax's tokenizer:
   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-M2.5")
   print(tok.encode("\n"))
   ```

2. Create a subclass in SGLang's `sglang/srt/sampling/custom_logit_processor.py`:
   ```python
   class MiniMaxThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
       THINKING_START_TOKEN_ID: int = 200050   # <think>
       THINKING_END_TOKEN_ID: int = 200051     # </think>
       NEW_LINE_TOKEN_ID: int = ???            # from step 1
   ```

3. Register it so callers (e.g. LiteLLM) can reference it by name instead of serializing with dill.

**Context:**
- MiniMax-M2.5 has always-on thinking (chat template injects `<think>\n`), no native way to limit it.
- MiniMax has explicitly refused to add thinking control (GitHub MiniMax-AI/MiniMax-M2 issues #25, #68).
- Each model family needs its own hardcoded subclass (Qwen3, Qwen3.5, DeepSeek-R1, GLM-4-MoE all have separate ones).
- Server requires `--enable-custom-logit-processor` (already set in `sglang_launch.sh`).
