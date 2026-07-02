#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Minimal SGLang launcher for the EMBEDDING instance (always single-node, tp=1).
#
# Deliberately SEPARATE from the ~1800-line generation launcher
# (sglang_launch.sh): an embedding server needs none of that script's chat/
# generation machinery (reasoning/tool parsers, speculative decoding, MoE runner
# backends, NCCL/QSFP distributed rendezvous, custom logit processors). Selected
# per-instance by sglang_instance.yml when inst.is_embedding is true (mounted as
# /scripts/launch.sh in place of the big script).
#
# Reads the SAME sglang-<prefix>-config ConfigMap env vars as the big launcher,
# but only the small subset an embedding server actually consumes. Single-node
# means NO --nnodes/--node-rank/--nccl-init-addr (no distributed group).
#
# Model choice rationale (Qwen3-Embedding, decoder arch): serving bge-m3
# (XLM-RoBERTa) here would hit sglang#7590 on GB10/SM121 (position-tensor assert
# in roberta.py, crash on the 2nd request). Qwen3-Embedding is unaffected.
# ---------------------------------------------------------------------------

# tini for correct signal handling / zombie reaping (not in the sglang image).
apt-get update -qq && apt-get install -y -qq tini >/dev/null 2>&1

# HF id resolves against the mounted HF cache (HF_HUB_OFFLINE=1 from the ConfigMap;
# the model-download initContainer has already populated /root/.cache/huggingface).
model_path="$SGLANG_MODEL"

args=(
  tini -s --
  python3 -m sglang.launch_server
  --model-path "$model_path"
  --is-embedding
  --tp-size "${TP:-1}"
  --context-length "$SGLANG_CONTEXT_LENGTH"
  --mem-fraction-static "$SGLANG_MEM_FRACTION"
  --port "$SGLANG_PORT"
  # Embedding-mode hygiene: no prefix reuse (radix cache is a decode-time KV
  # optimization, useless for one-shot embed passes) and no prefill chunking
  # (-1 = single-chunk prefill; avoids the chunked-prefill position handling
  # that trips encoder position asserts).
  --disable-radix-cache
  --chunked-prefill-size -1
)

# --host 127.0.0.1 (from the pod env): the HAProxy sidecar forwards
# 0.0.0.0:<inst.port> → 127.0.0.1:<inst.internal_port>, same EADDRINUSE fix as
# the generation head (SGLang's Scheduler binds <pod-ip>:port; uvicorn on
# 0.0.0.0 would collide).
if [ -n "${SGLANG_HOST:-}" ]; then
  args+=(--host "$SGLANG_HOST")
fi
# Encoder-friendly attention backend (profile default triton; SGLang's BGE note).
if [ -n "${SGLANG_ATTENTION_BACKEND:-}" ]; then
  args+=(--attention-backend "$SGLANG_ATTENTION_BACKEND")
fi
if [ "${SGLANG_TRUST_REMOTE_CODE:-false}" = "true" ]; then
  args+=(--trust-remote-code)
fi
# Stable served name (kept == the old ollama logical name so consumers that
# reference the model by name need no change on cutover).
if [ -n "${SGLANG_SERVED_MODEL_NAME:-}" ]; then
  args+=(--served-model-name "$SGLANG_SERVED_MODEL_NAME")
fi
# Prometheus exporter (/metrics on the HTTP port), gated per-instance.
if [ "${SGLANG_ENABLE_METRICS:-false}" = "true" ]; then
  args+=(--enable-metrics)
fi

# Echo the exact launch command to stdout (Loki-captured), same convention as
# the generation launcher.
printf '=== sglang EMBED launch command (%d args) ===\n' "${#args[@]}"
printf '%q ' "${args[@]}"
printf '\n=== end sglang EMBED launch command ===\n'

exec "${args[@]}"
