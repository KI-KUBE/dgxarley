"""[dgxarley] apply_fp8_linear: coerce a non-half/bf16 input to bf16 before the fp8 GEMM.

Symptom (RecViking/Mistral-Medium-3.5-128B-NVFP4 + EAGLE MTP, fp8 draft, SM121):
  eagle_draft_cuda_graph_runner → mistral_eagle.py:116 self.fc(torch.cat((embed,
  target_hidden), -1)) → fp8.py Fp8LinearMethod.apply → fp8_utils.apply_fp8_linear
  → fp8_scaled_mm(..., out_dtype=input.dtype) → "RuntimeError: out_dtype must be
  Half or BFloat16".

Why: the EAGLE fusion fc concatenates the draft's embedding with the NVFP4
  target's previous hidden state. That fused tensor's dtype is neither fp16 nor
  bf16 (float32 in the captured graph), and apply_fp8_linear passes
  out_dtype=input.dtype straight into the sgl_kernel, which only accepts
  half/bf16. NOT fixable via --speculative-draft-model-quantization unquant:
  SGLang normalizes "unquant" to None → auto-detects the Mistral-native draft's
  params.json qformat_weight fp8_e4m3 → the draft loads fp8 regardless.

ROOT FIX is elsewhere: --dtype bfloat16 (profile `dtype: bfloat16` → SGLANG_MODEL_DTYPE)
  stops the draft's fp32→fp16 fallback so the fc input is bf16 in the first place
  (SGLang cookbook Mistral-Medium-3.5 §3.3). This source patch is a BELT-AND-
  SUSPENDERS safety net behind that, kept deliberately: cast input to bf16 at the
  top of apply_fp8_linear when its dtype is not already half/bf16. The input is
  fp8-quantized immediately after anyway, so the f32→bf16 downcast is lossless in
  effect. Harmless for every normal fp8 linear (their input is already half/bf16
  → branch not taken). Idempotent via marker.
"""

from _patchlib import Patch

patch = Patch(
    name="apply_fp8_linear casts non-half/bf16 input to bf16",
    target="sglang/srt/layers/quantization/fp8_utils.py",
)

ANCHOR = "    # View input as 2D matrix for fp8 methods\n    input_2d = input.view(-1, input.shape[-1])"

# The original heredoc built this string via concatenation with the `marker`
# variable; reproduced here verbatim as a single literal (same resulting text).
NEW = (
    "    # [patch] _sgl_eagle_fp8_out_dtype_fix — EAGLE fc fuses NVFP4 target hidden states → input\n"
    "    # dtype is neither fp16 nor bf16; fp8_scaled_mm out_dtype=input.dtype\n"
    "    # then asserts Half/BFloat16. Cast to bf16 (input is fp8-quantized next).\n"
    "    if input.dtype not in (torch.float16, torch.bfloat16):\n"
    "        input = input.to(torch.bfloat16)\n" + ANCHOR
)


@patch.run
def apply(p: Patch) -> None:
    p.replace(ANCHOR, NEW, marker="# [patch] _sgl_eagle_fp8_out_dtype_fix", what="_sgl_eagle_fp8_out_dtype_fix")
