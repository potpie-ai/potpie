# Stress tests

## Running stress tests

- **Conversation profile** (POST /message, GET /active-session, GET /task-status):
  ```bash
  CONVERSATION_ID=<id> AUTH_HEADER="Bearer <token>" BASE_URL=http://localhost:8001 \
    uv run python scripts/stress_harness.py --profile conversation
  ```
- **Tunnel + GitHub profile**:
  ```bash
  AUTH_HEADER="Bearer <token>" uv run python scripts/stress_harness.py --profile tunnel_github
  ```
- Options: `CONCURRENCY=5,10,20`, `ROUNDS=2`, `TIMEOUT=120` (default for streaming).

## Why POST /message (stream) fails or times out

### 1. "Background task failed to start within 30s" (app log)

**Meaning:** The API waited 30 seconds for the Celery task to reach status **"running"** but it stayed **"queued"**.

**Cause:** There are only **4 Celery workers** (prefork). If you send more than 4 concurrent POST /message requests:

- The first 4 tasks get a worker and set status to `"running"` quickly (Celery log: "Task status set to running (Redis ok)").
- The 5th, 6th, … requests stay in the **Celery queue** until a worker is free.
- The API’s `wait_for_task_start(..., timeout=30, require_running=True)` keeps polling Redis; if the task is still queued after 30s, it logs the warning and continues (it still returns a streaming response).

So this warning is expected when **concurrency > number of workers** and workers are busy (e.g. on LLM or DB).

### 2. Client timeouts (stress script reports "timeouts=10")

**Meaning:** The HTTP client gave up before receiving enough data (e.g. default 120s).

**Cause:** For requests where the task stayed queued past 30s, the stream has no events until a worker picks the task up. If all workers are busy for a long time (e.g. slow LLM), the task may only start after the client has already timed out. So you see timeouts when:

- Concurrency is higher than worker count (e.g. 5 or 10 with 4 workers), and/or  
- Each task runs for a long time (LLM, tools, DB).

### 3. Celery logs show no errors

Celery logs show:

- Task received → "Task status set to running (Redis ok)" → "Entering run_async" → "run_agent: async_db acquired, resolving user".

So once a task gets a worker, status is set to `"running"` immediately. Failures in the stress run are from **tasks that never got a worker within 30s** (queue backlog), not from Redis or status-write bugs.

### What to do

- **Lower concurrency** for a fair test: e.g. `CONCURRENCY=3,4` so you don’t exceed 4 workers.
- **Increase workers** for higher load: in `scripts/start.sh` / `start.sh`, increase `--concurrency` (e.g. 8) and ensure DB pool and Redis can handle it.
- **Optional:** Increase the “task start” timeout (e.g. in `conversation_routing.py`, `wait_for_task_start(..., timeout=60)`) if you want the API to wait longer before logging the warning (streaming still works once the task runs).
- Before a stress run: clear the Celery queue and restart workers so old tasks don’t block slots:  
  `uv run python scripts/clear_celery_queue.py` then restart Celery.

## Troubleshooting

- **POST /message (stream) 100% timeouts:** Time-to-first-byte can exceed the client timeout when the Celery worker is busy or the queue is deep. Set `TIMEOUT=120` or higher (e.g. `TIMEOUT=180`) when running the conversation profile.
- **403 or 500 at high concurrency:** 403 may be auth/session limits under load; 500 is often DB pool exhaustion. Increase `DB_POOL_SIZE` and `DB_MAX_OVERFLOW` in `.env` (e.g. `DB_POOL_SIZE=20`, `DB_MAX_OVERFLOW=20`) when using multiple Gunicorn workers or stress testing. Check server logs for `QueuePool limit reached` or auth errors.
