# ComfyUI on DGX Spark (GB10, SM121, ARM64) — Image Build Guide

This guide describes, step by step, how to build a working container image
that runs ComfyUI **GPU-accelerated** on a DGX Spark (NVIDIA GB10 Grace‑Blackwell,
compute capability **SM_121**, `aarch64/ARM64`) — including FP8/FP4 checkpoints
such as `flux1-schnell-fp8.safetensors`.

The existing Ansible role path (`roles/k8s_dgx/tasks/comfyui.yml`) uses
`nvcr.io/nvidia/pytorch:25.09-py3` as its default and clones ComfyUI from git
into a hostPath at startup. That variant works — but it's bloated (~20 GB),
pulls pip dependencies on every start, and ships no xformers/flash-attn.
The goal of this guide: a **lean, reproducible image** with all kernels
pre-compiled for SM_121.

---

## 0. Prerequisites

- Build host on `aarch64` (one of the Sparks, e.g. `spark4`) — ComfyUI
  wheels must be built natively on ARM64; QEMU cross-builds are 10–20×
  slower and often break on CUTLASS templates.
- Docker or Podman with `nvidia-container-toolkit`, default runtime `nvidia`.
- At least **80 GB free disk** for build cache + final image.
- NVIDIA driver ≥ **580.x** on the host (Blackwell/SM_121 support).
- Registry account (Docker Hub / GHCR), example here: `xomoxcc/comfyui`.

Quick check on the build host:

```bash
uname -m                               # -> aarch64
nvidia-smi --query-gpu=compute_cap,driver_version --format=csv
# expected: 12.1, 580.x or higher
```

---

## 1. Choosing a base image

For SM_121 we need **CUDA ≥ 13.0** and **PyTorch with Blackwell support**.
Three viable bases (as of 2026-04):

| Base | Pro | Con |
|---|---|---|
| `nvcr.io/nvidia/pytorch:25.09-py3` | ARM64, cu13.1, torch 2.10, officially tested | ~20 GB, packed with things we don't need |
| `scitrera/dgx-spark-pytorch-dev:2.11.0-v1-cu132` | torch 2.11, cu13.2, leaner (~8 GB), ARM64 | Third-party, tag availability fluctuates (see `reference_sm121_build_base_regression`) |
| `nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04` | full control, minimal (~3 GB) | torch has to be installed manually |

**Recommendation:** for a first working version use NGC (`25.09-py3`); for
a lean production build use scitrera. The examples below use the NGC path —
the steps for scitrera are identical, only `FROM` changes.

> Note: before building, **check Docker Hub tags** — scitrera does not
> publish every tag referenced in recipes:
>
> ```bash
> curl -s 'https://hub.docker.com/v2/repositories/scitrera/dgx-spark-pytorch-dev/tags/?page_size=20' | jq '.results[].name'
> ```

---

## 2. Project layout

```
comfyui-sm121/
├── Dockerfile
├── requirements-extra.txt
├── entrypoint.sh
└── patches/
    └── (optional: sm121 patches for xformers/flash-attn)
```

---

## 3. `requirements-extra.txt`

Packages that are **not** in the base image but that ComfyUI really wants.
Install everything with `--upgrade-strategy only-if-needed` so the
`torch`/`torchvision` version preinstalled in the base is **not**
overwritten (otherwise Blackwell support is lost).

```txt
# Core
comfyui-frontend-package
comfyui-workflow-templates
comfyui-embedded-docs

# Samplers / schedulers
einops
torchsde
kornia>=0.7.1
spandrel
soundfile
av>=14.2.0
pydantic~=2.0
pydantic-settings~=2.0
alembic
SQLAlchemy

# Utility
huggingface_hub[cli]
transformers>=4.37.2
tokenizers>=0.13.3
sentencepiece
safetensors>=0.4.2
aiohttp>=3.11.8
yarl>=1.18.0
psutil
tqdm
Pillow
scipy
numpy>=1.25.0
```

> ComfyUI pins its own `requirements.txt` directly in the repo. We do **not**
> copy that file statically into the image; instead we pull it from the
> upstream checkout at build time so we don't chase a moving target.

---

## 4. `Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1.7
ARG BASE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${BASE}

# ---- System deps -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs ffmpeg libgl1 libglib2.0-0 \
        build-essential ninja-build cmake pkg-config \
        ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# ---- Python env --------------------------------------------------
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="12.1"              \
    CMAKE_CUDA_ARCHITECTURES=121             \
    MAX_JOBS=8                               \
    NVCC_THREADS=2                           \
    HF_HOME=/workspace/.cache/huggingface    \
    COMFYUI_PATH=/opt/comfyui

# TORCH_CUDA_ARCH_LIST=12.1 compiles SM_121 kernels exclusively.
# MAX_JOBS=8 is the empirically safe ceiling on GB10 (16 OOM-kills CUTLASS).

# ---- Clone ComfyUI (pinned commit for reproducibility) -----------
ARG COMFYUI_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_PATH} && \
    cd ${COMFYUI_PATH} && git checkout ${COMFYUI_REF} && \
    git rev-parse HEAD > ${COMFYUI_PATH}/.commit

# ---- ComfyUI's own requirements (WITHOUT overwriting torch) ------
RUN pip install --upgrade-strategy only-if-needed \
        -r ${COMFYUI_PATH}/requirements.txt

# ---- Our extra packages -----------------------------------------
COPY requirements-extra.txt /tmp/requirements-extra.txt
RUN pip install --upgrade-strategy only-if-needed \
        -r /tmp/requirements-extra.txt

# ---- Acceleration kernels (from source, SM_121) -----------------
# All three are optional — without them ComfyUI runs on stock
# PyTorch SDPA. With them, significantly more it/s on SDXL/FLUX.

# (a) xformers — memory-efficient attention; an ARM64 wheel exists
#     but often ships without SM_121 kernels. Build from source:
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation -v \
        git+https://github.com/facebookresearch/xformers.git@main

# (b) Sage-Attention v2 — faster than xformers on Blackwell for
#     most ComfyUI workloads. Optional, but strongly recommended.
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/thu-ml/SageAttention.git /tmp/sage && \
    cd /tmp/sage && \
    python -m pip install --no-build-isolation -v . && \
    rm -rf /tmp/sage

# (c) Flash-Attention 3 — only relevant for a few custom nodes,
#     build is RAM-heavy. If memory is tight: comment out.
# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install --no-build-isolation -v \
#         "git+https://github.com/Dao-AILab/flash-attention.git@main#subdirectory=hopper"

# ---- Entrypoint --------------------------------------------------
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
EXPOSE 8188
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
```

**About `TORCH_CUDA_ARCH_LIST=12.1`:**
SM_121 is not always registered yet in recent PyTorch/CUTLASS builds. If
a kernel refuses (error `no kernel image is available`), fall back to:
`TORCH_CUDA_ARCH_LIST="9.0a;12.0;12.1"` (builds SM_90a + SM_120 + SM_121 in
parallel). The image grows, but it's more portable.

---

## 5. `entrypoint.sh`

The image is immutable; dynamic data (models, outputs, custom nodes) lives
in the mounted `/workspace`. The entrypoint prepares that on first start
and then launches ComfyUI from the **image path** (`/opt/comfyui`), but
pulls models from `/workspace/models`.

```bash
#!/usr/bin/env bash
set -euo pipefail

DATA=/workspace
mkdir -p "$DATA"/{models/checkpoints,models/vae,models/clip,models/loras,output,temp,custom_nodes,user}

# Redirect ComfyUI's models folders to /workspace/models
if [ ! -f /opt/comfyui/extra_model_paths.yaml ]; then
cat > /opt/comfyui/extra_model_paths.yaml <<EOF
comfyui:
    base_path: /workspace
    checkpoints: models/checkpoints
    vae: models/vae
    clip: models/clip
    loras: models/loras
    custom_nodes: custom_nodes
EOF
fi

# Optional: pull FLUX.1-schnell on first start (ungated, Apache-2.0)
FLUX="$DATA/models/checkpoints/flux1-schnell-fp8.safetensors"
if [ ! -f "$FLUX" ] && [ "${DOWNLOAD_FLUX:-1}" = "1" ]; then
    huggingface-cli download Comfy-Org/flux1-schnell \
        flux1-schnell-fp8.safetensors \
        --local-dir "$DATA/models/checkpoints"
fi

cd /opt/comfyui
exec python main.py \
    --listen 0.0.0.0 \
    --port "${COMFYUI_PORT:-8188}" \
    --output-directory "$DATA/output" \
    --temp-directory "$DATA/temp" \
    --user-directory "$DATA/user" \
    "${COMFYUI_EXTRA_ARGS:-}"
```

With `--use-sage-attention` or `--use-pytorch-cross-attention` you can tell
ComfyUI explicitly which attention backend to use. Via env var:
`COMFYUI_EXTRA_ARGS="--use-sage-attention --fp8_e4m3fn-text-enc"`.

---

## 6. Building the image

**Always build on a Spark, never on `k3smaster` (x86_64)** — otherwise
kernels get compiled for the wrong architecture or emulated through QEMU.

```bash
# SSH to the build Spark (e.g. spark4, if ComfyUI is going to run there anyway)
ssh root@spark4

# Working directory
mkdir -p /root/comfyui-sm121 && cd /root/comfyui-sm121
#  Place Dockerfile, entrypoint.sh, requirements-extra.txt here

# Build (plain docker, or buildx if you want multi-stage caching)
docker build \
    --build-arg BASE=nvcr.io/nvidia/pytorch:25.09-py3 \
    --build-arg COMFYUI_REF=master \
    -t xomoxcc/comfyui:sm121 \
    -t xomoxcc/comfyui:sm121-$(date +%Y%m%d) \
    .
```

The initial build takes ~40–60 min (xformers + SageAttention are CUDA
compiles). With `--mount=type=cache` and an unchanged Dockerfile, later
builds finish in <10 min.

On OOM kills during kernel builds: try `MAX_JOBS=4` instead of `8`
(see `feedback_build_jobs_gb10` — `16` empirically kills CUTLASS, `8` is
the safe ceiling, but under memory pressure go lower still).

---

## 7. Local smoke test

**Before** pushing to the registry, test on the build host:

```bash
docker run --rm -it --gpus all \
    -p 8188:8188 \
    -v /var/lib/k8s-data/comfyui:/workspace \
    -e DOWNLOAD_FLUX=0 \
    xomoxcc/comfyui:sm121 \
    bash -c "python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())'"
# expected: True (12, 1)
```

Smoke test with the real UI:

```bash
docker run --rm -it --gpus all -p 8188:8188 \
    -v /var/lib/k8s-data/comfyui:/workspace \
    xomoxcc/comfyui:sm121
# -> browser: http://<spark4-ip>:8188
```

In the UI:
1. *Load Default* workflow → simple SD1.5 test prompt (FLUX needs the
   previously pulled checkpoint plus a text encoder).
2. `Queue Prompt` → the pod log should **not** show the line
   `no kernel image is available for execution on the device`.
3. `nvidia-smi dmon -s u` on the host during generation → GPU util > 0.

---

## 8. Pushing to the registry

```bash
docker login docker.io -u xomoxcc
docker push xomoxcc/comfyui:sm121
docker push xomoxcc/comfyui:sm121-$(date +%Y%m%d)
```

---

## 9. Switching the Ansible role over

Change `roles/k8s_dgx/defaults/main.yml`:

```yaml
comfyui_image: "xomoxcc/comfyui:sm121"
```

The launch script in `roles/k8s_dgx/tasks/comfyui.yml` can be slimmed
down drastically: the entire pip install block and the ComfyUI git clone
go away, because both are already baked into the image. Effectively only
the model download is left — and even that is already handled by the new
`entrypoint.sh`. Alternative: remove the ConfigMap launch script
entirely and drop the Deployment's `command` line (the image's ENTRYPOINT
is enough).

Rollout:

```bash
ansible-playbook k8s_dgx.yml --tags comfyui -e comfyui_enabled=true
```

> **Never deploy without explicit approval** — this guide describes the
> procedure; do not run the command above until you have reviewed the
> changes.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `RuntimeError: CUDA error: no kernel image is available` | Arch list missing 12.1 → rebuild image with `TORCH_CUDA_ARCH_LIST="9.0a;12.0;12.1"` |
| Build kills itself during SageAttention compile | RAM exhausted → `MAX_JOBS=4` or `2`, keep the host otherwise idle |
| ComfyUI starts, but generation is **slower** than the NGC image | Base-image skew — check torch/cu version in the new image (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`) and switch to a base with torch ≥ 2.11/cu13.2 if needed (see also the known regression pattern for SM_121 builds) |
| `!` tokens / NaN / black images on FLUX-fp8 | fp8 attention without kernel support → set `--use-sage-attention` or pull the fp16 variant instead |
| Pod stays `ContainerCreating` after an image push | `imagePullPolicy: IfNotPresent` + new tag → recreate the pod, or switch to `Always` + a versioned tag |
| `nvidia-smi` inside the pod shows no GPU | Time-slicing share gone? → `kubectl describe pod` → check the `nvidia.com/gpu` request. The cluster has 4 replicas/GPU via time-slicing (SGLang + ComfyUI share that) |

---

## 11. Variants / outlook

- **Leaner base:** once the build is stable, switch to
  `nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04` + manual torch — the image
  drops from ~20 GB to ~6 GB.
- **Models baked into the image instead of a hostPath:** not recommended —
  FLUX alone is 17 GB, and swapping models forces image rebuilds.
- **Preinstall ComfyUI-Manager:** clone it as a custom node into
  `/opt/comfyui/custom_nodes/ComfyUI-Manager` in the Dockerfile; it
  installs its own deps on first UI start.
- **Multi-GPU / pipeline parallelism:** ComfyUI is single-GPU. For batch
  generation, schedule multiple ComfyUI pods on different Sparks (one
  time-slice share each), don't try to spread a single instance across
  multiple GPUs.
