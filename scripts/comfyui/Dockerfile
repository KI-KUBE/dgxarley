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

# ---- Triton ------------------------------------------------------
# SageAttention has a hard import-time dependency on triton. xformers also
# uses triton for several backend paths and prints
#   "A matching Triton is not available, some optimizations will not be enabled"
# without it. The scitrera 2.10.0-v2-cu131 base does NOT ship triton on
# aarch64, so we install it explicitly here, BEFORE the kernel builds.
#
# Strategy: read torch's own Requires-Dist metadata and install the exact
# triton package + version that torch declares. That way we don't pick a
# random PyPI version that mismatches torch's expected ABI. Falls back to
# PyPI 'triton' only when torch declares no triton dep at all (rare, only
# happens on bases stripped of the dep metadata).
#
# Pitfall encountered before: the package `pytorch-triton` on PyPI is a
# defensive squat stub that hard-crashes on import with
#   RuntimeError("Should never be installed")
# The real pytorch-triton wheels live on download.pytorch.org/whl/. By
# reading torch's metadata we automatically get the right name (whether
# that's `triton`, `pytorch-triton`, or a versioned variant), pinned to
# torch's ABI.
ARG INSTALL_TRITON=1
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${INSTALL_TRITON}" = "1" ]; then \
        TRITON_REQ=$(python3 -c "import importlib.metadata as m; reqs = m.requires('torch') or []; tr = [r for r in reqs if 'triton' in r.split(';')[0].lower() and 'extra ==' not in r]; print('\n'.join(tr))") && \
        if [ -n "${TRITON_REQ}" ]; then \
            echo "torch declares triton requirement: ${TRITON_REQ}" && \
            echo "${TRITON_REQ}" | pip install --upgrade-strategy only-if-needed -r /dev/stdin ; \
        else \
            echo "torch declares no triton dep — falling back to PyPI 'triton'" && \
            pip install --upgrade-strategy only-if-needed triton ; \
        fi && \
        python3 -c "import triton; print('triton', triton.__version__)" ; \
    fi

# ---- Acceleration kernels (from source, SM_121) -----------------
# Both are optional; ComfyUI falls back to torch SDPA without them. With
# SageAttention especially, SDXL/FLUX throughput improves substantially.
ARG BUILD_XFORMERS=1
ARG BUILD_SAGE_ATTN=1

# xformers: pinned to a release tag (v0.0.32) instead of @main. main has
# repeatedly produced broken aarch64 builds in the past, and pinning gives
# us reproducible behaviour. setuptools-scm may still synthesize a
# 0.0.33+<sha>.dXXXXXX version on top of the pinned tag — that's expected.
#
# Verify check explanation: do NOT do `from xformers import _C` — xformers'
# `_C.so` is a torch-ops library (loaded via `torch.ops.load_library`),
# not a regular CPython extension, so it has no `PyInit__C` symbol and
# direct Python import always raises
#   ImportError: dynamic module does not define module export function (PyInit__C)
# even on a perfectly healthy install. The meaningful test is whether the
# public API (ops.memory_efficient_attention) is exposed after import —
# that proves torch.ops.load_library succeeded and the kernels are bound.
ARG XFORMERS_REF=v0.0.32
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${BUILD_XFORMERS}" = "1" ]; then \
        pip install --no-build-isolation -v \
            "git+https://github.com/facebookresearch/xformers.git@${XFORMERS_REF}" && \
        python3 -c "import xformers, xformers.ops as xo; \
                    assert hasattr(xo, 'memory_efficient_attention'), \
                        'xformers build is broken: ops.memory_efficient_attention missing'; \
                    print('xformers', xformers.__version__, 'mea ok')" ; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${BUILD_SAGE_ATTN}" = "1" ]; then \
        git clone https://github.com/thu-ml/SageAttention.git /tmp/sage && \
        cd /tmp/sage && \
        pip install --no-build-isolation -v . && \
        cd / && \
        rm -rf /tmp/sage && \
        python3 -c "import sageattention; print('sageattention ok')" ; \
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
