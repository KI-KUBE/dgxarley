# sglang-0.5.12-cudnn.Dockerfile
#
# Adds nvidia-cudnn-frontend Python binding on top of
# scitrera/dgx-spark-sglang:0.5.12 so that flashinfer's fi_cudnn FP4 GEMM
# backend (fp4_gemm_backend=flashinfer_cudnn) becomes usable.
#
# Why only the frontend (not nvidia-cudnn-cu12, unlike the 0.5.10 variant):
# the 0.5.12 base image already ships libcudnn 9.20.0 via apt under
# /usr/lib/aarch64-linux-gnu/ (verified at runtime: ldconfig -p shows all
# libcudnn*.so.9 entries). flashinfer's _check_cudnn_availability only
# needs (a) the loader to find libcudnn at runtime and (b) the python
# 'cudnn' module to be importable. The runtime libs are already there;
# only the python binding is missing. Pulling in nvidia-cudnn-cu12 on
# top would lay a second copy of libcudnn*.so under site-packages and
# risk LD ordering / version-mismatch surprises.
#
# Smoke test in the final RUN layer fails the build if the frontend
# ends up importable but flashinfer's check still rejects the runtime
# (e.g. version skew between the apt libs and what nvidia-cudnn-frontend
# expects), so a broken combination never gets tagged.

ARG BASE_IMAGE=scitrera/dgx-spark-sglang:0.5.12
FROM ${BASE_IMAGE}

RUN pip install --no-cache-dir nvidia-cudnn-frontend \
 && python3 -m pip show nvidia-cudnn-frontend >/dev/null \
 && python3 -c "from flashinfer.gemm.gemm_base import _check_cudnn_availability; _check_cudnn_availability(); print('flashinfer cuDNN check OK')"
