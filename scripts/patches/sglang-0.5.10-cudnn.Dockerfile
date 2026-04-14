# sglang-0.5.10-cudnn.Dockerfile
#
# Adds nvidia-cudnn-cu12 + nvidia-cudnn-frontend Python wheels on top of
# scitrera/dgx-spark-sglang:0.5.10 so that flashinfer's fi_cudnn FP4 GEMM
# backend (fp4_gemm_backend=flashinfer_cudnn) becomes usable.
#
# The upstream scitrera image ships flashinfer without cuDNN, so every
# fi_cudnn config in the matrix crashes at first FP4 GEMM call with:
#   flashinfer/gemm/gemm_base.py:_check_cudnn_availability
#   RuntimeError: cuDNN is not available.
# The package names below are exactly what flashinfer's error message
# instructs — do not change without re-reading the upstream check.
#
# BASE_IMAGE is overridable at build time. The driver script
# (build_cudnn_image.sh) passes scitrera/dgx-spark-sglang:0.5.10 via
# --build-arg by default — see that script's header for the rationale
# (upstream base avoids the sm121 perf regression, and the sm121
# CUTLASS MoE fix is irrelevant at EP=1).

ARG BASE_IMAGE=scitrera/dgx-spark-sglang:0.5.10
FROM ${BASE_IMAGE}

RUN pip install --no-cache-dir nvidia-cudnn-cu12 nvidia-cudnn-frontend \
 && python3 -m pip show nvidia-cudnn-cu12 nvidia-cudnn-frontend >/dev/null \
 && python3 -c "from flashinfer.gemm.gemm_base import _check_cudnn_availability; _check_cudnn_availability(); print('flashinfer cuDNN check OK')"
