# SGLang Upstream Bug: AttributeError on abort in streaming /v1/chat/completions

## Status

**Present** in our image `scitrera/dgx-spark-sglang:0.5.10`. First observed 2026-04-10
after calling `POST /abort_request` with `{"abort_all": true}` while streaming chat
completions were in-flight.

## Affected Configuration

- Endpoint: `/v1/chat/completions` with `stream: true`
- Trigger: scheduler-side abort of an in-flight streaming request (e.g. via
  `POST /abort_request` with `abort_all: true`, or any other path that sets
  `finish_reason.type == "abort"`)

Non-streaming `/v1/chat/completions` and `/v1/responses` are not affected by this
code path.

## The Bug

In `sglang/srt/entrypoints/openai/serving_chat.py` (lines 698-709 in 0.5.10), the
scheduler-abort branch crashes with:

```
File ".../sglang/srt/entrypoints/openai/serving_chat.py", line 706, in _generate_chat_stream
    code.name,
    ^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'name'
```

The offending block:

```python
if finish_reason_type == "abort":
    code = finish_reason.get(
        "status_code", HTTPStatus.INTERNAL_SERVER_ERROR
    )
    error = self.create_streaming_error_response(
        finish_reason.get("message", "Generation aborted."),
        code.name,
        code.value,
    )
```

`finish_reason` is a dict produced by the scheduler. When the abort path sets
`status_code` to `None` explicitly (rather than omitting the key), `dict.get(key, default)`
returns the stored `None` — the default is **only** used when the key is missing.
`code.name` then raises `AttributeError`.

Separately, even if `status_code` is present, the code assumes it is an `HTTPStatus`
enum instance. If the scheduler stores a plain `int` (e.g. `499`), `code.name`/`code.value`
would also fail — the code needs to coerce via `HTTPStatus(code)` first.

## Observed Trigger

Streaming chat completions were in-flight; a `POST /abort_request` with
`{"rid": "", "abort_all": true}` was issued. The streaming generator crashed inside
Starlette's `stream_response` → `_generate_chat_stream`, emitting a full traceback
into the pod log per aborted request. The 200 response to `/abort_request` succeeded;
the fallout is purely in the active streams being cleaned up.

Sample log excerpt (truncated):

```
File ".../starlette/responses.py", line 250, in stream_response
    async for chunk in self.body_iterator:
File ".../sglang/srt/entrypoints/openai/serving_chat.py", line 623, in prepend_first_chunk
    async for chunk in generator:
File ".../sglang/srt/entrypoints/openai/serving_chat.py", line 706, in _generate_chat_stream
    code.name,
AttributeError: 'NoneType' object has no attribute 'name'
[2026-04-10 15:04:40] INFO:     10.68.0.140:0 - "POST /abort_request HTTP/1.1" 200 OK
```

## Suggested Fix (upstream)

Normalize `code` to an `HTTPStatus` before accessing `.name`/`.value`, and treat a
`None` value the same as a missing key:

```python
if finish_reason_type == "abort":
    raw_code = finish_reason.get("status_code")
    if raw_code is None:
        raw_code = HTTPStatus.INTERNAL_SERVER_ERROR
    if not isinstance(raw_code, HTTPStatus):
        try:
            raw_code = HTTPStatus(int(raw_code))
        except ValueError:
            raw_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error = self.create_streaming_error_response(
        finish_reason.get("message", "Generation aborted."),
        raw_code.name,
        raw_code.value,
    )
```

## Our Workaround

None. The exception is raised inside the per-stream async generator, so it only kills
the individual stream that was being aborted — the server keeps running and subsequent
requests are unaffected. The log noise is cosmetic but worth filing upstream.

## Related

- `SGLANG_REASONING_TOKENS_UPSTREAM_BUG.md`
- `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md`
- `SGLANG_SHARDED_SPECULATIVE_UPSTREAM_BUG.md`
