# SGLang API Endpoints (v0.5.x)

## Health / Info

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/health_generate` | Health check with test generate |
| GET | `/ping` | Ping (SageMaker-compatible) |
| GET | `/get_model_info` | Model name, parameter count, etc. |
| GET | `/get_server_info` | Server configuration |
| GET | `/get_weight_version` | Current weight version |
| GET | `/get_load` | Current load (queue length, etc.) |
| GET | `/metrics` | Prometheus metrics (requires `--enable-metrics`) |

## Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/generate` | Native SGLang generate (non-OpenAI) |
| POST | `/generate_from_file` | Generate from uploaded file (for `input_embeds`) |
| POST | `/encode` | Embedding request |
| POST | `/classify` | Reward model / classification request |
| POST | `/v1/completions` | OpenAI Completions API |
| POST | `/v1/chat/completions` | OpenAI Chat Completions API |
| POST | `/v1/score` | Score / Embedding (OpenAI-compatible) |
| POST | `/v1/responses` | OpenAI Responses API |
| GET | `/v1/responses/{id}` | Retrieve a response |
| POST | `/v1/responses/{id}/cancel` | Cancel a single response |
| POST | `/invocations` | SageMaker-compatible |
| POST | `/vertex_generate` | Vertex AI-compatible (env `AIP_PREDICT_ROUTE`) |

## Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/models` | List available models |
| GET | `/v1/models/{model}` | Single model info |

## Control

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/abort_request` | Abort request(s) — `{"rid": "", "abort_all": true}` to abort all |
| POST | `/pause_generation` | Pause generation (also aborts all running requests) |
| POST | `/continue_generation` | Resume generation after pause |
| GET/POST | `/flush_cache` | Flush KV cache (skipped if requests are running) |
| GET/POST | `/configure_logging` | Change logging configuration at runtime |

## Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/open_session` | Open a session, returns unique session ID |
| GET/POST | `/close_session` | Close a session |

## LoRA Adapters

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/load_lora_adapter` | Load a LoRA adapter without restarting |
| POST | `/unload_lora_adapter` | Unload a LoRA adapter |

## Memory Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/release_memory_occupation` | Release GPU memory temporarily |
| GET/POST | `/resume_memory_occupation` | Resume GPU memory occupation |

## Weight Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/update_weights_from_disk` | Reload weights from disk |
| POST | `/init_weights_update_group` | Initialize weight update group |
| POST | `/update_weights_from_tensor` | Update weights from tensor |
| POST | `/update_weights_from_distributed` | Distributed weight update |
| POST | `/update_weight_version` | Set weight version |
| GET/POST | `/get_weights_by_name` | Get model parameter by name |

## Post-Processing

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/parse_function_call` | Parse tool/function call from output |
| POST | `/separate_reasoning` | Separate reasoning from output |

## Internal / Testing

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/set_internal_state` | Set internal server state (e.g. `max_micro_batch_size`) |
| GET/POST | `/slow_down` | Deliberately slow down system (testing only) |

## Notes

- There is **no endpoint to list active requests** by ID. Use `/get_load` for queue counts or `/metrics` for Prometheus gauges (`sglang_num_running_reqs`, `sglang_num_waiting_reqs`).
- Most endpoints accept both GET and POST where noted as GET/POST.
