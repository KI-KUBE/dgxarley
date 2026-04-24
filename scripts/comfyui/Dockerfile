# syntax=docker/dockerfile:1.7
#
# xomoxcc/comfyui:sm121 — ComfyUI pre-baked for DGX Spark (GB10 / SM_121 / ARM64).
#
# Default BASE is scitrera's published pytorch-dev image (ARM64, CUDA, torch
# with Blackwell support). Override with --build-arg BASE=<image> to swap in
# the xomoxcc 2.11/cu132 base for the ~45% perf uplift documented in
# reference_sm121_build_base_regression (if available on the build host).
#
# What this image contains
#   - Frozen ComfyUI checkout at /opt/comfyui (pinned via COMFYUI_REF)
#   - ComfyUI's own pip requirements + our extras (see requirements-extra.txt)
#   - xformers built from source for SM_121
#   - SageAttention v2 built from source for SM_121
#
# What this image deliberately does NOT contain
#   - Model weights (multi-GB, churn fast, belong on hostPath / PVC / S3)
#   - ComfyUI-Manager or any custom nodes (belong under /workspace/custom_nodes
#     so they persist across image upgrades without rebuilds)

ARG BASE=scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131
FROM ${BASE}

# ---- System deps -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs ffmpeg libgl1 libglib2.0-0 \
        build-essential ninja-build cmake pkg-config \
        ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# ---- Env --------------------------------------------------------
# TORCH_CUDA_ARCH_LIST=12.1 keeps the build narrow (SM_121 only) so CUTLASS
# template expansion stays inside the GB10 memory budget. Add 9.0a;12.0 if
# the resulting image must also run on Hopper or early Blackwell silicon.
#
# MAX_JOBS=8 is the empirically safe ceiling on GB10 (128 GB unified);
# 16 OOM-kills CUTLASS template instantiations (see feedback_build_jobs_gb10).
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="12.1" \
    CMAKE_CUDA_ARCHITECTURES=121 \
    MAX_JOBS=8 \
    NVCC_THREADS=2 \
    HF_HOME=/workspace/.cache/huggingface \
    COMFYUI_PATH=/opt/comfyui

# ---- ComfyUI checkout -------------------------------------------
ARG COMFYUI_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_PATH} && \
    cd ${COMFYUI_PATH} && git checkout ${COMFYUI_REF} && \
    git rev-parse HEAD > ${COMFYUI_PATH}/.commit

# ---- Python deps (ComfyUI + extras) -----------------------------
# --upgrade-strategy only-if-needed prevents torch/torchvision/triton from
# the base image being replaced. The base ships Blackwell-capable wheels;
# replacing them silently regresses perf.
RUN pip install --upgrade-strategy only-if-needed \
        -r ${COMFYUI_PATH}/requirements.txt

COPY requirements-extra.txt /tmp/requirements-extra.txt
RUN pip install --upgrade-strategy only-if-needed \
        -r /tmp/requirements-extra.txt

# ---- Acceleration kernels (from source, SM_121) -----------------
# Both are optional; ComfyUI falls back to torch SDPA without them. With
# SageAttention especially, SDXL/FLUX throughput improves substantially.
ARG BUILD_XFORMERS=1
ARG BUILD_SAGE_ATTN=1

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${BUILD_XFORMERS}" = "1" ]; then \
        pip install --no-build-isolation -v \
            git+https://github.com/facebookresearch/xformers.git@main ; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${BUILD_SAGE_ATTN}" = "1" ]; then \
        git clone https://github.com/thu-ml/SageAttention.git /tmp/sage && \
        cd /tmp/sage && \
        pip install --no-build-isolation -v . && \
        rm -rf /tmp/sage ; \
    fi

# ---- Custom nodes (opt-in bake-in) -------------------------------
# Runtime-Install ist der Default-Weg: Custom-Nodes landen per
# `git clone` unter /workspace/custom_nodes/ (hostPath), damit
# Updates keinen Image-Rebuild erzwingen. Beispiel:
#
#   ssh root@spark4 '
#     cd /var/lib/k8s-data/comfyui/custom_nodes &&
#     git clone https://github.com/Acly/comfyui-inpaint-nodes.git
#   '
#   kubectl -n comfyui rollout restart deploy/comfyui
#
# Falls stattdessen hart ins Image gebacken werden soll (reproduzierbar,
# aber Update = Rebuild), folgenden Block entkommentieren:
#
# ARG INPAINT_NODES_REF=main
# RUN git clone https://github.com/Acly/comfyui-inpaint-nodes.git \
#         ${COMFYUI_PATH}/custom_nodes/comfyui-inpaint-nodes && \
#     cd ${COMFYUI_PATH}/custom_nodes/comfyui-inpaint-nodes && \
#     git checkout ${INPAINT_NODES_REF}

# ---- Entrypoint -------------------------------------------------
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ---- Build stamp ------------------------------------------------
# BUILDTIME wird vom build_comfyui_image.sh als --build-arg auf
# UTC-ISO-8601 gesetzt (z. B. "2026-04-24T18:42:01Z"). Als ENV im
# Image verfügbar, damit der Entrypoint-Log + Image-Label-Inspektion
# sofort zeigen, welcher Build läuft — praktisch gegen das Phänomen
# "Pod läuft auf alter Image-Layer trotz neuem Tag".
ARG BUILDTIME=unknown
ENV BUILDTIME=${BUILDTIME}
LABEL org.opencontainers.image.created="${BUILDTIME}"

WORKDIR /workspace
EXPOSE 8188
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
