# Async Migration Plan — Native Libraries, Step by Step

## Context

The previous PRs used `asyncio.to_thread` to wrap blocking calls. The code review flagged:
- Thread-safety bugs (DB sessions passed to worker threads)
- Unbounded queue in custom stream generator
- Daemon thread-per-stream with no cap
- `StreamingResponse` with sync generator already runs in Starlette's thread pool (~40 tokens) — the custom thread+queue was unnecessary overhead
- Thread pool saturation risk under load

**New direction:** Replace blocking libraries with native async equivalents. Use `to_thread` only as a temporary bridge for code that cannot be converted yet. Measure every step.

---

## Step 0: Revert thread-based changes

**Branch:** `fix/wait-for-task-start-thread` (5 commits)

Revert these changes that introduced `asyncio.to_thread` wrappers and the custom stream generator:

| File | What to revert |
|------|----------------|
| `conversation_routing.py` | Remove `redis_stream_generator_async`, revert `start_celery_task_and_stream` to sync, remove `asyncio.to_thread` around `ensure_unique_run_id` and `wait_for_task_start` |
| `conversations_router.py` | Remove `asyncio.to_thread` wrappers around `session_service`, `redis_manager`, `ensure_unique_run_id`; restore `local_mode` / request / UA detection in `create_conversation` |
| `conversation_service.py` | Remove `asyncio.to_thread` around `history_manager.*`, `session_service.*`, `redis_manager.*`, `celery_app.control.revoke` |
| `redis_streaming.py` | Revert `stop_event` parameter, revert xread block from 1000ms back to 5000ms. Keep `require_running` in `wait_for_task_start` (that's a correctness fix, not a threading change) |
| `api/router.py` | Remove `asyncio.to_thread` around `ensure_unique_run_id` |
| `tunnel_router.py` | Remove `asyncio.to_thread` wrappers |
| `tunnel_service.py` | Revert scan_iter and TTL changes only if they were part of the thread PR. Keep if they were independent fixes |
| `socket_service.py` | Remove `close()` method if it was added as part of the thread PR |
| `main.py` | Remove shutdown handler if it was part of the thread PR |
| `github_service.py` | Remove `asyncio.to_thread` around `redis.get`/`redis.setex` in `get_project_structure_async` |

**Keep from the thread PR:**
- `wait_for_task_start(require_running=True)` — correctness fix, not a threading change
- Timeout additions (timeout=30 on requests calls) — safety fix

**Keep from `fix/sync-timeouts-only`:**
- All timeout additions (auth_service, github_service, parse_webhook_helper)
- `parsing_controller._query_parsing_status_sync` with `asyncio.to_thread` — acceptable bridge until async DB migration


---

## Step 1: AsyncRedisStreamManager for FastAPI

**What:** Create `redis.asyncio` client alongside the existing sync client. Add async methods for all operations called from FastAPI.

**Files to change:**
- `redis_streaming.py` — add `AsyncRedisStreamManager` class using `redis.asyncio`
  - `async def get_task_status()`
  - `async def set_task_status()`
  - `async def set_task_id()`
  - `async def get_task_id()`
  - `async def publish_event()`
  - `async def set_cancellation()`
  - `async def get_stream_snapshot()`
  - `async def clear_session()`
  - `async def wait_for_task_start()` — use `asyncio.sleep` instead of `time.sleep`
- `conversation_routing.py` — use `AsyncRedisStreamManager` in `start_celery_task_and_stream` (make it `async def`), `start_celery_task_and_wait`
  - `ensure_unique_run_id` → `async_ensure_unique_run_id` using async Redis
  - Leave sync `redis_stream_generator` as-is (Starlette thread pool handles it)
- `session_service.py` — add `AsyncSessionService` using `AsyncRedisStreamManager`. Use `await async_redis.scan(...)` instead of `keys()`
- `conversations_router.py` — switch to async versions
- `api/router.py` — switch to async versions
- `conversation_service.py` — `stop_generation` uses async Redis methods

**Keep unchanged:**
- Sync `RedisStreamManager` — still used by Celery tasks (`agent_tasks.py`, `parsing_tasks.py`)
- Sync `redis_stream_generator` — Starlette runs it in thread pool

**Test script: `scripts/benchmark_redis_async.py`**
```
Measures:
1. Time to complete N concurrent POST /message requests (streaming)
2. Time to complete N concurrent GET /active-session requests
3. Time to complete N concurrent GET /task-status requests
4. Compare: before (sync Redis on event loop) vs after (native async Redis)
5. Report: p50, p95, p99 latency; requests/sec; event loop blocked time (via middleware)
```

**PR:** `feat/async-redis-stream-manager`

---

## Step 2: AsyncSession for hot-path DB services

**What:** Migrate services called from FastAPI to use `AsyncSession`.

### 2a: UsageService

**Why first:** Called on every `create_conversation` and `post_message` (check_usage_limit). Highest frequency.

- `usage_service.py` — replace `SessionLocal()` + `session.query()` with `AsyncSessionLocal` + `await async_session.execute(select(...))`
- Inject `AsyncSession` from FastAPI dependency

**Test script: `scripts/benchmark_usage_check.py`**
```
Measures:
1. N concurrent POST /conversations (each triggers check_usage_limit)
2. Latency distribution before/after
```

**PR:** `feat/async-usage-service`

### 2b: ChatHistoryService (dual-path) ✅

**Why:** Used from both FastAPI (store_message, stop_generation) and Celery (agent_tasks). Cannot just swap to async.

- Create `AsyncChatHistoryService` with `AsyncSession`:
  - `async def get_session_history()`
  - `add_message_chunk()` (in-memory buffer only)
  - `async def flush_message_buffer()`
  - `async def save_partial_ai_message()`
- `ConversationService.create()` — accepts optional `async_db`; when provided, builds `AsyncChatHistoryService` and uses it on FastAPI path
- FastAPI path uses async variant; Celery path uses existing sync `ChatHistoryService`

**Test script: `scripts/benchmark_chat_flow.py`**
```
Measures:
1. N concurrent POST /message (streaming) — end-to-end including DB writes
2. N concurrent POST /stop — stop_generation with DB save
3. Latency distribution
```

**PR:** `feat/async-chat-history-service`

### 2c: ShareChatService, AccessService, UserService ✅

- **AsyncShareChatService**: New class in `access_service.py` using `AsyncSession`; `share_chat`, `get_shared_emails`, `remove_access` (select/update/commit).
- **conversations_router**: Share/access endpoints use `get_async_db` and `AsyncShareChatService(async_db)`.
- **AsyncUserService**: New class in `user_service.py`; async `get_user_by_uid`, `get_user_id_by_email`, `get_user_by_email`, `get_user_ids_by_emails`, `create_user`, `update_last_login`. Sync `UserService` retained for Celery/main.
- **api/router**: `get_api_key_user` uses `AsyncUserService(async_db).get_user_by_uid`. Auth routes (2d) remain on sync UserService for now.

**Test script: `scripts/benchmark_share_access.py`**
```
Measures:
1. N concurrent POST /share
2. N concurrent GET /shared-emails
3. N concurrent DELETE /access
```

**PR:** `feat/async-share-access-user-services`

### 2d: Auth routes (signup, sso_login, provider endpoints) ✅

- auth_router.py: inject `get_async_db`, use `AsyncUserService(async_db)` for user lookups (get_user_by_uid, get_user_by_email, update_last_login) in signup, sso_login, get_my_account
- `UnifiedAuthService` remains sync for this PR; full async migration can follow

**Test script: `scripts/benchmark_auth.py`**
```
Measures:
1. N concurrent POST /signup
2. N concurrent POST /sso/login
3. N concurrent GET /providers/me
4. Latency distribution
```

**PR:** `feat/async-auth-db`

### 2e: GithubService DB queries ✅

- `get_combined_user_repos(user_id, async_session=...)` and `get_repos_for_user(user_id, async_session=...)` accept optional `AsyncSession`; when provided use `select()` + `await session.execute()`
- `code_provider_controller.get_user_repos` and github_router pass `async_db` into `get_combined_user_repos`

**PR:** `feat/async-github-db` (included in feat/async-share-access-user-services)

---

## Step 3: Async Redis for non-streaming services ✅

**What:** Add `redis.asyncio` to services that do simple Redis get/set from FastAPI.

| Service | Change |
|---------|--------|
| `tunnel_service.py` | ✅ Async client + `get_workspace_tunnel_record_async`, `set_workspace_tunnel_record_async`, `list_user_tunnels_async`. Sync `list_user_tunnels` uses `scan_iter` not `keys`. Workspace record `set` uses TTL (`WORKSPACE_TUNNEL_RECORD_TTL`). |
| `tunnel_router.py` | ✅ Both routes call `await tunnel_service.get_workspace_tunnel_record_async(workspace_id)`. |
| `github_service.py` | ✅ Lazy shared `_get_async_redis_cache()`; `get_project_structure_async` uses async Redis for cache get/setex when available. |
| `branch_cache.py` | ✅ Optional `_async_redis_client`; `get_branches_async` for FastAPI path. Controller `get_branch_list` uses `await branch_cache.get_branches_async`. |

**Test script: `scripts/benchmark_tunnel_github.py`**
```
Measures:
1. N concurrent GET /tunnel/workspace/{id}
2. N concurrent GET /tunnel/workspace/{id}/socket-status
3. N concurrent GET /github/user-repos (triggers branch cache / repo listing)
4. Latency distribution (p50, p95, p99); req/s
Usage: WORKSPACE_ID=<16-hex> AUTH_HEADER="Bearer <token>" CONCURRENT=5 ROUNDS=2 uv run python scripts/benchmark_tunnel_github.py
```

**PR:** `feat/async-redis-tunnel-github`

---

## Step 4: Async HTTP clients ✅

**What:** Replace `requests` with `httpx` where called from FastAPI; offload sync-only SDKs to threads.

| Location | Change |
|----------|--------|
| `auth_service.py` | Sync login uses `httpx.Client`; added `login_async` with `httpx.AsyncClient` (timeout 10s/30s). Route calls `await auth_handler.login_async`. |
| `auth_router.py` | `send_slack_message` uses `httpx.AsyncClient().post` with timeout. |
| `parse_webhook_helper.py` | `send_slack_notification` uses `httpx.AsyncClient().post` with timeout. |
| `linear_client.py` | Sync `execute_query` uses `httpx.Client` (timeout 10s/30s); added `execute_query_async`, `get_issue_async`, `update_issue_async`, `comment_create_async`. Linear tools use async methods. |
| `email_helper.py` | `resend.Emails.send` wrapped in `await asyncio.to_thread(resend.Emails.send, params)` in both `send_email` and `send_parsing_failure_alert`. |
| `posthog_helper.py` | `send_event`: when event loop is running, uses `loop.run_in_executor(None, _capture_sync)` (fire-and-forget); otherwise runs `_capture_sync` inline. |

**Test script: `scripts/benchmark_login.py`** (optional; not added in this PR)

**PR:** Combined with Step 3 in `feat/async-redis-tunnel-github`

---

## Step 5: SearchService async DB

- `search_service.py` — migrate `search_codebase` to `AsyncSession`
- `knowledge_graph_router.py` — use async session for project verification query

**PR:** `feat/async-search-service`

---

## Test infrastructure

### Shared benchmark harness: `scripts/benchmark_harness.py`

```python
"""
Usage: python scripts/benchmark_harness.py --target <script> --concurrency 10,20,50 --rounds 3

For each concurrency level:
1. Warm up (5 requests)
2. Run N concurrent requests × rounds
3. Report: min, p50, p95, p99, max latency; total wall time; requests/sec
4. Compare with baseline file if provided (--baseline results.json)
"""
```

### Event loop monitor middleware: `scripts/eventloop_monitor.py`

```python
"""
FastAPI middleware that measures event loop blocking.
Logs a warning when any request handler blocks the event loop for > 100ms.
Reports: blocked_time_ms, endpoint, method.
Run with: add middleware to app in development/staging.
"""
```

### Per-step test scripts

Each step has a dedicated script (listed above) that:
1. Hits the actual API endpoints of a running local server
2. Uses `httpx.AsyncClient` with `asyncio.gather` for concurrency
3. Requires `CONVERSATION_ID`, `AUTH_TOKEN` env vars (real auth against local)
4. Outputs JSON results for before/after comparison
5. Validates: latency improved, no errors, functional correctness (response shape)

---

## Execution order and PR strategy

| Order | PR | Risk | Impact |
|-------|-----|------|--------|
| 0 | Revert thread wrappers | Low (restores known-good state) | Removes thread-safety bugs |
| 1 | AsyncRedisStreamManager | Medium (new component) | Unblocks streaming hot path |
| 2a | Async UsageService | Low (isolated) | Unblocks every message send |
| 2b | Async ChatHistoryService | Medium (dual-path) | Unblocks message DB writes |
| 2c | Async Share/Access/User | Low (simple CRUD) | Unblocks sharing endpoints |
| 2d | Async Auth DB | High (large surface) | Unblocks login/signup |
| 2e | Async GitHub DB | Low (small) | Unblocks repo listing |
| 3 | Async Redis (tunnel/github cache) | Low | Unblocks tunnel + cache |
| 4 | Async HTTP clients | Low | Unblocks login, notifications |
| 5 | Async SearchService | Low | Unblocks search |

Each PR:
1. Run benchmark **before** (on main or previous step)
2. Merge the PR
3. Run benchmark **after**
4. Record results in `docs/async-migration-results.md`
5. If regression detected, revert before proceeding

---

## What we do NOT change

- **Celery tasks** — sync by design. No changes.
- **Neo4j** — all usage is in Celery agent tasks via `to_thread(self.run)`. Fine as-is.
- **File I/O / subprocess** — all in Celery parsing tasks. Fine.
- **Sync `redis_stream_generator`** — Starlette's thread pool handles sync iterators in `StreamingResponse`. No custom thread/queue.
- **`to_thread` inside Celery's event loop** (tool `arun` methods) — Celery creates its own loop per task; `to_thread` there is fine.

---

## Dependencies to add

```
# Already in the project:
redis           # has redis.asyncio built-in
sqlalchemy      # has AsyncSession built-in
httpx           # already used in some places

# No new packages needed — all async support is in existing deps.
```

---

## Success criteria

Per step:
- p95 latency for affected endpoints improves or stays flat
- No new errors in logs
- Event loop blocked time (via middleware) decreases
- Thread count (via `/proc` or metrics) decreases or stays flat

Overall:
- All FastAPI async routes have zero sync I/O on the event loop
- Thread pool usage is only from Starlette's managed pool for `StreamingResponse` sync generators
- Measured ~30-40% more requests/instance capacity (Duolingo benchmark)
