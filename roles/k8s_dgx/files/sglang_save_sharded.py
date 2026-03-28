"""Pre-shard an SGLang model for fast multi-node TP loading.

Runs as a multi-node job (one process per TP rank). Each rank's Engine
loads only its portion of the model, then saves its shard locally.
After completion each node has its rank's files under the sharded path.

Architecture:
    Engine() on rank 1 (worker) BLOCKS — it runs the scheduler event loop
    in-process and never returns. The worker's scheduler receives the
    save_sharded_model RPC via broadcast from the head's scheduler.
    Therefore, only rank 0's script calls save_sharded_model(); rank 1
    just needs to start Engine() and stay alive.

Environment variables (set via ConfigMap):
    SGLANG_MODEL          Model ID (e.g. QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ)
    SGLANG_QUANTIZATION   Quantization method (e.g. awq, gptq, or empty)
    TP                    Tensor parallel size (default: 2)
    EP                    Expert parallel size (default: 1). Partitions TP group
                          for MoE layers. With EP=TP, MoE uses all-to-all.
    NNODES                Number of nodes (default: 2)
    NODE_RANK             This node's rank (0 = head, 1 = worker)
    QSFP_IP_SPARK1        NCCL init address (head IP)
    NCCL_PORT             NCCL bootstrap port
    HF_TOKEN              HuggingFace token (optional)
    SHARD_OUTPUT_DIR      Override output directory (optional)

After sharding, start SGLang with:
    --model-path <sharded-dir> --load-format sharded_state
"""

import os
import shutil
import sys
from pathlib import Path


def main():
    model_id = os.environ.get("SGLANG_MODEL", "")
    quantization = os.environ.get("SGLANG_QUANTIZATION", "") or None
    attention_backend = os.environ.get("SGLANG_ATTENTION_BACKEND", "") or None
    moe_runner_backend = os.environ.get("SGLANG_MOE_RUNNER_BACKEND", "") or None
    tp = int(os.environ.get("TP", "2"))
    ep = int(os.environ.get("EP", "1"))
    nnodes = int(os.environ.get("NNODES", "2"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    nccl_init_addr = f"{os.environ.get('QSFP_IP_SPARK1', '10.10.10.101')}:{os.environ.get('NCCL_PORT', '50000')}"

    if not model_id:
        print("ERROR: SGLANG_MODEL not set", flush=True)
        sys.exit(1)

    model_slug = model_id.replace("/", "--")
    shard_suffix = f"sglang-TP{tp}"
    if ep > 1:
        shard_suffix += f"-EP{ep}"
    if quantization:
        shard_suffix += f"-{quantization}"
    default_output = f"/root/.cache/huggingface/sharded/{model_slug}-{shard_suffix}"
    output_dir = os.environ.get("SHARD_OUTPUT_DIR", default_output)

    print(f"[rank {node_rank}] Model:        {model_id}", flush=True)
    print(f"[rank {node_rank}] Quantization: {quantization or '(none)'}", flush=True)
    print(f"[rank {node_rank}] TP size:      {tp}, EP size: {ep}, nnodes: {nnodes}, rank: {node_rank}", flush=True)
    print(f"[rank {node_rank}] NCCL init:    {nccl_init_addr}", flush=True)
    print(f"[rank {node_rank}] Output:       {output_dir}", flush=True)

    # Check if already sharded (index.json is written last by ShardedStateLoader)
    force = os.environ.get("FORCE_RESHARD", "").lower() in ("1", "true", "yes")
    if (Path(output_dir) / "model.safetensors.index.json").exists() and not force:
        print(f"[rank {node_rank}] Sharded checkpoint already exists (index.json found), skipping.", flush=True)
        print(f"[rank {node_rank}] Use -e force_sglang_shard=true to re-shard.", flush=True)
        sys.exit(0)
    if force and (Path(output_dir) / "model.safetensors.index.json").exists():
        print(f"[rank {node_rank}] FORCE_RESHARD set — removing existing shards at {output_dir}", flush=True)
        shutil.rmtree(output_dir)
        print(f"[rank {node_rank}] Removed {output_dir}", flush=True)

    # Ensure the model is downloaded locally
    from huggingface_hub import snapshot_download

    print(f"[rank {node_rank}] Ensuring model is downloaded...", flush=True)
    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir="/root/.cache/huggingface/hub",
    )
    print(f"[rank {node_rank}] Model cached at: {local_path}", flush=True)

    # Create Engine with multi-node TP params
    from sglang import Engine

    # mem_fraction_static from env (set by Ansible from model profile), fallback 0.80.
    # Shard job only loads weights and saves — no inference. Use the same fraction
    # as the serving profile so the Engine doesn't OOM on heavy models (e.g.
    # MiniMax-M2.5-NVFP4 at ~70 GB/GPU needs > 0.60). context_length=128 minimizes
    # the actual KV cache allocation within that fraction.
    mem_frac = float(os.environ.get("SGLANG_MEM_FRACTION", "0.80"))

    engine_kwargs = {
        "model_path": local_path,
        "tp_size": tp,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": nccl_init_addr,
        "mem_fraction_static": mem_frac,
        "context_length": 128,
        "disable_cuda_graph": True,
        "disable_piecewise_cuda_graph": True,
    }
    if ep > 1:
        engine_kwargs["ep_size"] = ep
    if quantization:
        engine_kwargs["quantization"] = quantization
    if attention_backend:
        engine_kwargs["attention_backend"] = attention_backend
    if moe_runner_backend:
        # Critical for NVFP4+sharded_state: process_weights_after_loading takes
        # different branches depending on the MoE backend (flashinfer_cutlass →
        # scalar input_scale vs. else → per-expert). Shards must be saved with
        # the same backend as serving, otherwise tensor shapes won't match at load.
        engine_kwargs["moe_runner_backend"] = moe_runner_backend
    # Speculative decoding params (speculative_algo, etc.) are NOT passed to
    # Engine — they only affect inference, not weight sharding. The shard job
    # loads and saves weights identically regardless of speculative mode.
    # They are still used for directory naming (shard_suffix) so the sharded
    # path matches what the runtime launch script expects.

    if os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM", "").lower() == "false":
        print(f"[rank {node_rank}] DeepGemm JIT disabled", flush=True)

    # Ensure output dir exists before Engine (scheduler saves to this path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[rank {node_rank}] Initializing Engine...", flush=True)

    if node_rank != 0:
        # Worker: Engine() BLOCKS on rank != 0 — it runs the scheduler event
        # loop in-process and never returns. The scheduler receives the
        # save_sharded_model RPC via broadcast from the head and writes the
        # rank-1 shard files to output_dir. When the head exits and NCCL
        # disconnects, SGLang sends SIGQUIT killing this process. The launch
        # shell script handles post-save cleanup (metadata copy + marker).
        llm = Engine(**engine_kwargs)
        # If Engine() ever returns on worker, just exit cleanly.
        sys.exit(0)
    else:
        # Head: Engine() returns on rank 0, allowing us to call RPCs.
        llm = Engine(**engine_kwargs)

        print(f"[rank {node_rank}] Saving sharded state to {output_dir}...", flush=True)
        # Workaround for SGLang 0.5.9: the RPC dispatcher unpacks parameters
        # as **kwargs, but the mixin expects a single positional `params` dict.
        llm.save_sharded_model(
            params={
                "path": output_dir,
                "pattern": None,
                "max_size": 5 * 1024**3,  # 5 GB per shard file
            }
        )
        print(f"[rank {node_rank}] save_sharded_model returned.", flush=True)

    # Copy metadata files (config.json, tokenizer, etc.)
    for file in os.listdir(local_path):
        src = os.path.join(local_path, file)
        dst = os.path.join(output_dir, file)
        if os.path.exists(dst):
            continue
        ext = os.path.splitext(file)[1]
        if ext in (".bin", ".pt", ".safetensors"):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)

    # model.safetensors.index.json is written by ShardedStateLoader.save_model
    # as the last step — serves as the completion marker.
    print(f"[rank {node_rank}] Sharding complete.", flush=True)


if __name__ == "__main__":
    main()
