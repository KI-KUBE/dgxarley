#!/usr/bin/env bash
# Entrypoint for xomoxcc/comfyui:sm121.
#
# The image ships a frozen ComfyUI checkout at /opt/comfyui. Mutable data
# (models, outputs, custom_nodes, user settings) lives under /workspace —
# mount that as a hostPath / PVC in the deployment. ComfyUI's model paths
# are redirected to /workspace/models via extra_model_paths.yaml generated
# on first start.
#
# Env vars
#   COMFYUI_PORT           Listen port (default: 8188)
#   COMFYUI_EXTRA_ARGS     Appended to `python main.py ...`
#                          e.g. "--use-sage-attention --fp8_e4m3fn-text-enc"
#   DOWNLOAD_FLUX          If "1" and the fp8 checkpoint is missing, pull
#                          Comfy-Org/flux1-schnell on first start (default: 0)

set -euo pipefail

DATA=/workspace
APP=/opt/comfyui

mkdir -p \
    "$DATA/models/checkpoints" \
    "$DATA/models/vae" \
    "$DATA/models/clip" \
    "$DATA/models/loras" \
    "$DATA/models/controlnet" \
    "$DATA/models/upscale_models" \
    "$DATA/output" \
    "$DATA/temp" \
    "$DATA/custom_nodes" \
    "$DATA/user"

# Redirect ComfyUI's models/custom_nodes to /workspace without touching
# the image. Only write the file once — users may edit it to add more
# locations (e.g. a shared NFS model cache).
if [ ! -f "$APP/extra_model_paths.yaml" ]; then
    cat > "$APP/extra_model_paths.yaml" <<EOF
comfyui:
    base_path: /workspace
    checkpoints: models/checkpoints
    vae: models/vae
    clip: models/clip
    loras: models/loras
    controlnet: models/controlnet
    upscale_models: models/upscale_models
    custom_nodes: custom_nodes
EOF
fi

# Optional first-start model pull. Off by default — production deploys
# should seed /workspace/models from an external source (rsync, S3) so
# the first pod start is fast and reproducible.
FLUX="$DATA/models/checkpoints/flux1-schnell-fp8.safetensors"
if [ "${DOWNLOAD_FLUX:-0}" = "1" ] && [ ! -f "$FLUX" ]; then
    echo "[entrypoint] pulling Comfy-Org/flux1-schnell (first start)..."
    huggingface-cli download Comfy-Org/flux1-schnell \
        flux1-schnell-fp8.safetensors \
        --local-dir "$DATA/models/checkpoints"
fi

echo "[entrypoint] image buildtime: ${BUILDTIME:-unknown}"
echo "[entrypoint] ComfyUI commit: $(cat "$APP/.commit" 2>/dev/null || echo unknown)"
PY="$(command -v python3 || command -v python)"
echo "[entrypoint] torch: $("$PY" -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability() if torch.cuda.is_available() else "n/a")')"

cd "$APP"
# shellcheck disable=SC2086   # COMFYUI_EXTRA_ARGS intentionally unquoted for word-split
exec "$PY" main.py \
    --listen 0.0.0.0 \
    --port "${COMFYUI_PORT:-8188}" \
    --output-directory "$DATA/output" \
    --temp-directory "$DATA/temp" \
    --user-directory "$DATA/user" \
    ${COMFYUI_EXTRA_ARGS:-}
