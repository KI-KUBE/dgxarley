#!/bin/bash
set -e

# Install tools (not included in sglang image).
# Head (rank 0) needs rsync + ssh for syncing metadata to the worker node.
apt-get update -qq && apt-get install -y -qq tini iproute2 iputils-ping net-tools rsync openssh-client

# Prime ARP table on the QSFP P2P link before NCCL tries to connect.
if [ "$NODE_RANK" = "0" ]; then
  peer="$QSFP_IP_SPARK2"
else
  peer="$QSFP_IP_SPARK1"
fi
echo "Waiting for QSFP peer ${peer} ..."
until ping -c10 -W1 "$peer" ; do
  sleep 1
done
echo "QSFP peer ${peer} reachable."

# Patch moe_wna16 weight loader for EP-aware qzeros handling (SGLang 0.5.9 bug).
# Two bugs in the w13_qzeros/w2_qzeros branches:
#   1. Uses raw global expert_id (0-127) instead of local EP index (0-63)
#   2. Uses global tp_rank for TP-slice, but moe_tp_size=tp/ep — need tp_rank % moe_tp_size
# Safe no-op when ep_size=1 (identity mapping, tp_rank unchanged).
MOE_WNA16="/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/moe_wna16.py"
if grep -q 'param\.data\[expert_id' "$MOE_WNA16" 2>/dev/null; then
  python3 << 'PATCH_QZEROS_EOF'
import sys
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/moe_wna16.py"
with open(f) as fh:
    code = fh.read()
old_w13 = '''            if "w13_qzeros" in weight_name:
                tensor = loaded_weight.view(
                    layer.moe_tp_size, -1, loaded_weight.size(1)
                )[tp_rank]
                if shard_id == "w1":
                    param.data[expert_id, : shard_size // 2] = tensor
                else:
                    param.data[expert_id, shard_size // 2 :] = tensor'''
new_w13 = '''            if "w13_qzeros" in weight_name:
                _local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
                if _local_id == -1:
                    return
                _moe_tp_rank = tp_rank % layer.moe_tp_size
                tensor = loaded_weight.view(
                    layer.moe_tp_size, -1, loaded_weight.size(1)
                )[_moe_tp_rank]
                if shard_id == "w1":
                    param.data[_local_id, : shard_size // 2] = tensor
                else:
                    param.data[_local_id, shard_size // 2 :] = tensor'''
old_w2 = '''            elif "w2_qzeros" in weight_name:
                param.data[expert_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.moe_tp_size, -1
                )[:, tp_rank]'''
new_w2 = '''            elif "w2_qzeros" in weight_name:
                _local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
                if _local_id == -1:
                    return
                _moe_tp_rank = tp_rank % layer.moe_tp_size
                param.data[_local_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.moe_tp_size, -1
                )[:, _moe_tp_rank]'''
if old_w13 not in code:
    print("w13_qzeros: already patched or source changed, skipping")
    sys.exit(0)
if old_w2 not in code:
    print("w2_qzeros: already patched or source changed, skipping")
    sys.exit(0)
code = code.replace(old_w13, new_w13, 1)
code = code.replace(old_w2, new_w2, 1)
with open(f, 'w') as fh:
    fh.write(code)
print("Patched moe_wna16.py: EP-aware expert_id + tp_rank remapping for qzeros")
PATCH_QZEROS_EOF
fi

# Clean stale shard files from previous failed runs so we only detect
# freshly written shards in the post-save check below.
model_slug=$(echo "$SGLANG_MODEL" | sed 's|/|--|g')
shard_suffix="TP${TP}"
if [ -n "$EP" ] && [ "$EP" != "1" ]; then
  shard_suffix="${shard_suffix}-EP${EP}"
fi
if [ -n "$SGLANG_QUANTIZATION" ]; then
  shard_suffix="${shard_suffix}-${SGLANG_QUANTIZATION}"
fi
shard_dir="/root/.cache/huggingface/sharded/${model_slug}-${shard_suffix}"
if [ ! -f "$shard_dir/model.safetensors.index.json" ]; then
  rm -f "$shard_dir"/model-rank-*-part-*.safetensors 2>/dev/null || true
  rm -f "$shard_dir"/model.safetensors.index.json 2>/dev/null || true
fi

# Timestamp marker: only shard files newer than this count as valid.
# Protects against partial writes from a crash mid-save in THIS run.
mkdir -p "$shard_dir"
touch "$shard_dir/.shard_run_start"

# Run the shard script. On the worker (rank != 0), Engine() blocks and
# exits via SIGQUIT when the head disconnects — expected behavior.
# The worker's scheduler writes its shard files before the disconnect.
# Disable set -e: worker exits non-zero (SIGQUIT) and we need to handle it.
set +e
tini -s -- python3 /scripts/save_sharded.py
rc=$?
set -e

# Worker post-save: if the Python script exited non-zero (SIGQUIT from
# head disconnect), check if shard files were written and finalize.
if [ "$NODE_RANK" != "0" ] && [ $rc -ne 0 ]; then
  # ShardedStateLoader only writes model.safetensors.index.json on rank 0.
  # On the worker, we only check for rank-specific shard files.
  # The head rsyncs index.json + metadata to the worker after saving.
  rm -f "$shard_dir/.shard_run_start"
  shard_count=$(find "$shard_dir" -name "model-rank-${NODE_RANK}-part-*.safetensors" 2>/dev/null | wc -l)
  if [ "$shard_count" -gt 0 ]; then
    echo "[rank $NODE_RANK] Save complete: ${shard_count} shard files present."
    # Copy metadata from HF cache (config.json, tokenizer, etc.)
    hub_path=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$SGLANG_MODEL', cache_dir='/root/.cache/huggingface/hub', local_files_only=True))" 2>/dev/null || true)
    if [ -n "$hub_path" ] && [ -d "$hub_path" ]; then
      for f in "$hub_path"/*; do
        base=$(basename "$f")
        case "$base" in *.bin|*.pt|*.safetensors) continue ;; esac
        [ -e "$shard_dir/$base" ] && continue
        cp -rL "$f" "$shard_dir/$base"
      done
    fi
    echo "[rank $NODE_RANK] Sharding complete."
    exit 0
  else
    echo "[rank $NODE_RANK] ERROR: No shard files found after Engine exit."
    exit 1
  fi
fi

# Head post-save: sync index.json + metadata to the worker node so both
# nodes have the complete shard directory for --load-format sharded_state.
# ShardedStateLoader only writes model.safetensors.index.json on rank 0.
if [ "$NODE_RANK" = "0" ] && [ -n "$RSYNC_TARGET" ]; then
  ssh_opts="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
  dst="root@${RSYNC_TARGET}:${HF_CACHE_HOST_PATH}/sharded/${model_slug}-${shard_suffix}/"
  echo "[rank $NODE_RANK] Syncing index + metadata to ${RSYNC_TARGET} ..."
  rsync -ah \
    -e "ssh $ssh_opts" \
    --exclude='model-rank-*-part-*.safetensors' \
    --exclude='.shard_run_start' \
    "$shard_dir/" "$dst"
  echo "[rank $NODE_RANK] Sync to ${RSYNC_TARGET} complete."
fi
