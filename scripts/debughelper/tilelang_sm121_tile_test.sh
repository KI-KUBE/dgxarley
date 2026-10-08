#!/usr/bin/env bash
set -u
D=/root/tilelang_test
IMAGE=localhost/xomoxcc/dgx-spark-sglang:0.5.21-sm121
podman run --rm --device nvidia.com/gpu=all --ipc=host --network=host \
    -v "${D}:/work" -v "${D}/cache:/root/.tilelang" -e TILELANG_CACHE_DIR=/root/.tilelang \
    "${IMAGE}" python3 /work/tilelang_sm121_tile_test.py
