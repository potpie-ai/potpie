# Reproducing silent chat streaming failures

This repo's historical silent-stream symptom was tied to the **Redis/SSE bridge** rather than to a magic prompt string. The captured repro pattern is:

1. A normal chat/task stream is opened with `stream=true`.
2. The backend starts waiting for Redis stream events for the generated run/session.
3. Redis consumption fails, times out, or the stream expires before the UI receives a useful terminal error frame.

That is the real failing pattern this repro validates. Prompt-level hangs can still occur when a model or tool stalls, but those should be tracked with the exact prompt, conversation, timestamps, and tool context because they are a different failure class.

## Captured task pattern

Use any normal streaming chat/task request. The prompt content is intentionally boring because the bug is in stream delivery, not prompt semantics:

```text
Concurrent timing test message.
```

For manual API/UI validation, any short prompt such as `hello` is sufficient as long as the stream is interrupted using one of the triggers below.

---

## Inputs you need

| Input | Purpose |
|--------|---------|
| `CONVERSATION_ID` | Existing conversation the API can stream for |
| `RUN_ID` / `session_id` | From UI URL or API response (`X-Run-Id` / normalised session id) |
| API base URL | e.g. local FastAPI `/api/v1` |
| Auth | Same cookies / bearer as the failing browser session |
| Redis URL | Worker + API must share the **same** Redis for streams; mismatches reproduce “nothing on stream” |

---

## Repeatable repro (automated, recommended)

Runs the same failure signatures the debuggability work targets (trace/stamp/logging + client error frame), without depending on flaky infra.

From repo root, with `.venv` active:

```bash
.venv/bin/python -m pytest tests/unit/conversations/test_streaming_debuggability.py -v
```

Pattern encoded in tests:

- **Redis consume blows up**: `consume_stream.side_effect = ConnectionError(...)` → server logs stack + `run_id` / `trace_id` / `failing_phase=sse_consume`; SSE yields a sanitized `event: error` JSON frame before close.
- **Published events**: asserts `trace_id` on Redis `publish_event` payloads.
- **Terminal `end` with error**: forwarded as sanitized chat-compatible error frame.

Re-run anytime after changing `conversation_routing`, `redis_streaming`, or `agent_tasks`.

---

## Optional concurrent-stream timing repro

Stress/integration check for event-loop blocking (different failure class, but overlapping “stream weirdness”):

```bash
.venv/bin/python -m pytest tests/integration-tests/conversations/test_concurrent_stream_timing.py -v -m stress
```

Uses a fixed form body (`content`: `"Concurrent timing test message."`) and concurrent POSTs to `/message?stream=true` (via test client).

---

## Manual infra repro (destructive, local/dev only)

Use when you want the **full** HTTP + SSE path against real Redis/Celery.

1. Start API + Celery pointing at **one** Redis.
2. POST a normal streaming message (`stream=true`) using the captured task pattern above so a stream key is created.
3. While the SSE client is connected, **`FLUSHDB` / disconnect Redis used by API** so `xread` / `exists` fail or the generator hits the consume error path; or **pause/stop Celery** so stream creation waits past timeout (`stream_creation_wait` -> `timeout` end).
4. Inspect: API/worker logs for `failing_phase` + stack; browser devtools `[stream-debug]` logs (frontend) should show `trace_id` / `run_id` when present.

Repeatable as long as the same teardown steps are scripted; automate with caution in CI.

---

## Definition-of-done checklist (this issue)

- [x] **Captured pattern**: Redis/SSE bridge failure during a normal streaming chat/task request.
- [x] **Repeated trigger**: `.venv/bin/python -m pytest tests/unit/conversations/test_streaming_debuggability.py -v`.
- [x] **Documented inputs**: conversation/run identifiers, API base, auth, Redis URL, and exact commands above.

If you have a production incident with a **particular prompt**, append: exact message text, `conversation_id`, `run_id`, timestamp, environment, and, if allowed, an anonymized Redis stream snapshot.
