# Task failure – log analysis and root cause

## Why POST /message fails while GET /active-session and GET /task-status pass

- **POST /message** enqueues a Celery task (`execute_agent_background`) and streams the response from Redis. The **worker** runs the task; if the worker blocks or errors, the stream never (or slowly) gets data and the client times out or sees a failure.
- **GET /active-session** and **GET /task-status** are plain HTTP handlers: they hit the FastAPI app (and DB) directly and return immediately. They **do not** run Celery tasks. So they pass as long as the API and DB are up.

So when only /message fails, the cause is in the **Celery task** path (worker blocking or error), not in the HTTP layer.

---

## Where logs are

| Component | Where logs go |
|-----------|----------------|
| **Celery worker** | If started with `nohup celery ... > /tmp/celery_worker.log 2>&1`, then `/tmp/celery_worker.log`. If started by `scripts/start.sh` or `start.sh`, stdout/stderr of that terminal (no file by default). |
| **Backend (Gunicorn/FastAPI)** | stdout/stderr of the process that ran `start.sh` or `gunicorn`. No log file unless you redirect (e.g. `gunicorn ... 2>&1 | tee gunicorn.log`). |

So for a run started with `./scripts/start.sh` in a terminal, both backend and Celery logs appear in that terminal.

---

## What the Celery log shows

From `/tmp/celery_worker.log` (or equivalent):

- **No ERROR or exception tracebacks** in the captured slice.
- Every task does:
  1. `Task ... execute_agent_background[...] received`
  2. `Starting background agent execution ...`
  3. `Task status set to running (Redis ok)`
  4. `Entering run_async (agent coroutine)`
  5. `run_agent: coroutine started, acquiring async_db`
  6. `run_agent: async_db acquired, resolving user (sync DB on main thread)`
  7. `run_agent: opening SessionLocal for user lookup`
- **No line after that** (e.g. no `run_agent: user resolved, creating ConversationService`, no `Background agent execution completed`, no `Background agent execution failed`).

So in the worker, execution consistently **stops right after** “resolving user (sync DB on main thread)”. The next step is either `SessionLocal()` (waiting for a DB connection; pool has pool_timeout=30) or `UserService(_db).get_user_by_uid(user_id)` (sync query; can block on lock or slow query). While workers are stuck here, the stream for each /message request gets no events and the client reports failure (timeout).

---

## Route that triggers the task

The task that runs this path is **`execute_agent_background`**. It is enqueued from:

1. **Streaming message**  
   - **Route:** `POST /api/v1/conversations/{conversation_id}/message` (and v2 equivalent).  
   - **Code:** `app/modules/conversations/utils/conversation_routing.py` → `start_celery_task_and_stream()` → `execute_agent_background.delay(...)`.

2. **Non‑streaming (wait for full response)**  
   - Same route, different code path: `start_celery_task_and_wait()` → `execute_agent_background.delay(...)`.

So the **route that causes** the behaviour you see in the Celery log is:

- **`POST /conversations/{conversation_id}/message`** (streaming or wait).

---

## Root cause (from logs)

- **In the worker:**  
  The run **hangs or blocks** somewhere in the block that runs **right after** the log line “resolving user (sync DB on main thread)”:
  - `SessionLocal()` (get a DB connection), or  
  - `UserService(_db).get_user_by_uid(user_id)` (sync query), or  
  - `_db.close()`.

  So the **route cause** is the code path that leads to that sync DB block in `execute_agent_background` (i.e. the `/message` route above). The **concrete cause** is that one of those sync DB steps is blocking (e.g. pool exhaustion, slow or locked query, or connection wait).

- **In the backend (if you see “task failed to start”):**  
  You may also see in **backend** logs:
  - `"Background task failed to start within 30s for <conversation_id>:<run_id> - may still be queued"`  
  That comes from `conversation_routing.py` when `wait_for_task_start(..., timeout=30)` does not see status `"running"` in time. So either:
  - The task is still **queued** (no free worker), or  
  - The task **did** get a worker and set `"running"` but then **hangs** in the sync DB block (so the stream never progresses and the client may timeout).

---

## What was added to narrow it down

In `app/celery/tasks/agent_tasks.py`, two **info** logs were added around the sync DB block:

- `run_agent: opening SessionLocal for user lookup` — right before `SessionLocal()`.
- `run_agent: get_user_by_uid returned (user=True/False)` — right after `get_user_by_uid`.

On the next run:

- If you see **“opening SessionLocal”** but **not** “get_user_by_uid returned”, the hang is in **getting a connection** (`SessionLocal()`) or in **`get_user_by_uid`** (query).
- If you see **“get_user_by_uid returned”**, the hang is **after** that (e.g. “user resolved, creating ConversationService” or later).

---

## What to do next

1. **Restart Celery** so it loads the new logging.
2. **Reproduce** (e.g. one POST to `/conversations/{id}/message`).
3. **Check Celery logs** for the new lines to see exactly where it stops.
4. **Check backend logs** (terminal where you ran `start.sh` or Gunicorn) for:
   - `"Background task failed to start within 30s"`  
   - Any 5xx or Redis/DB errors.
5. If the hang is at `SessionLocal()` or `get_user_by_uid`:
   - Check DB pool size and usage (e.g. `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, Postgres `max_connections`).
   - Check for long-running or blocking queries/locks on the DB.
