# Redis sync/blocking fixes — tracking

Track progress on removing sync Redis usage from async paths and fixing related issues.

**Scope:** Redis-related items only (streaming, wait_for_task_start, get/setex, tunnel, ensure_unique_run_id, decode_responses, lifespan).

---

## Summary

| Status | Count |
|--------|--------|
| Done | 1 |
| In progress | 0 |
| Not started | 6 |
| **Total** | **7** |

---

## 1. Redis stream XREAD/listen blocks event loop

**Status:** [ ] Not started

**Files:**
- `app/modules/conversations/utils/redis_streaming.py` — `consume_stream()` (L65–151): `xread(..., block=5000)`, `time.sleep(0.5)`, `exists()`, `xrange()`, `xrevrange()`
- `app/modules/conversations/utils/conversation_routing.py` — `redis_stream_generator()` (L61–75): sync generator passed to `StreamingResponse()`

**Impact:** Every streaming chat request blocks the event loop for up to 5s per chunk (and during initial wait).

**Fix direction:** Use async Redis (`redis.asyncio`) for stream consumption and an async generator for SSE; or run sync `consume_stream` in a thread and feed an `asyncio.Queue`, with the endpoint consuming from the queue in an async generator.

**Notes:**

---

## 2. wait_for_task_start blocks event loop

**Status:** [x] Done

**Files:**
- `app/modules/conversations/utils/redis_streaming.py` — `wait_for_task_start()` (L296–308): sync loop with `get_task_status()` (Redis GET) + `time.sleep(0.5)`
- Call sites: `conversation_routing.py` (L160, L230), `conversations_router.py` (L422, etc.)

**Impact:** Up to 30s (or 10s) blocking at the start of each streaming/non-streaming request.

**Fix direction:** Use async Redis + `await asyncio.sleep(0.5)` in an async loop, or run existing sync `wait_for_task_start` in a thread: `await asyncio.to_thread(redis_manager.wait_for_task_start, ...)`.

**Notes:** Implemented: `start_celery_task_and_stream` made async; all call sites use `await asyncio.to_thread(redis_manager.wait_for_task_start, ...)`. Routers updated to `await start_celery_task_and_stream(...)`.

**Timing verification:**
- **Pytest:** `tests/integration-tests/conversations/test_concurrent_stream_timing.py` — asserts N concurrent streaming requests complete in wall time &lt; 85% of (N × single wait), proving `wait_for_task_start` runs off the event loop.
- **Before vs after (same run):** Run `pytest tests/integration-tests/conversations/test_concurrent_stream_timing.py -k before_vs_after -s -v` to see both timings in one run: "BEFORE (sync on event loop)" vs "AFTER (thread offload)" and the speedup (e.g. ~2x for 3 requests with a 1.2s simulated delay).

---

## 3. Sync Redis get/setex in async method (github_service)

**Status:** [ ] Not started

**Files:**
- `app/modules/code_provider/github/github_service.py` — `get_project_structure_async()`: L1126 `self.redis.get(cache_key)`, L1171 `self.redis.setex(...)`

**Impact:** Project-structure (repo tree) requests block the event loop for Redis round-trips.

**Fix direction:** Use async Redis here and `await redis.get()` / `await redis.setex()`, or wrap: `await asyncio.to_thread(self.redis.get, cache_key)` (and same for setex).

**Notes:**

---

## 4. Sync Redis in tunnel/socket status (is_workspace_online)

**Status:** [ ] Not started

**Files:**
- `app/modules/tunnel/socket_service.py` — `is_workspace_online()` (L101–113): sync `_get_sync_redis()`, `sync_redis.exists(key)`
- `app/modules/tunnel/tunnel_router.py` — L50 `get_workspace_metadata`, L72 `get_workspace_socket_status`: both call `is_workspace_online()` from async handlers

**Impact:** Every workspace metadata/socket-status request blocks the event loop for one Redis round-trip.

**Fix direction:** Add `async def is_workspace_online_async()` using `await self._get_redis()` and `await redis.exists(key)`, and use it from the two async route handlers; or call sync version in thread: `await asyncio.to_thread(get_socket_service().is_workspace_online, workspace_id)`.

**Notes:**

---

## 5. ensure_unique_run_id sync Redis on every message

**Status:** [ ] Not started

**Files:**
- `app/modules/conversations/utils/conversation_routing.py` — `ensure_unique_run_id()` (L42–57): loop with `redis_manager.redis_client.exists(stream_key)`
- Call sites: `conversations_router.py` (L282, L359), `app/api/router.py` (L183)

**Impact:** Sync Redis in hot path of post-message/start-stream.

**Fix direction:** Use async Redis for this check, or run in thread: `run_id = await asyncio.to_thread(ensure_unique_run_id, conversation_id, run_id)`.

**Notes:**

---

## 6. decode_responses consistency (Celery vs FastAPI Redis)

**Status:** [ ] Not started

**Files:**
- `app/modules/conversations/utils/redis_streaming.py` — L15: `redis.from_url(config.get_redis_url())` — no `decode_responses` (default False, bytes)
- Celery uses same `RedisStreamManager` (sync, bytes)
- When adding async Redis in FastAPI, must match

**Impact:** If FastAPI async client uses `decode_responses=True` while Celery writes with False, keys/values can be misread (e.g. type/None issues).

**Fix direction:** For any new async Redis client that shares keys/streams with Celery, use the same `decode_responses=False` as `RedisStreamManager`, or document and test the chosen convention.

**Notes:**

---

## 7. Async Redis client lifecycle (aclose on shutdown)

**Status:** [ ] Not started

**Files:**
- `app/main.py` — no lifespan/shutdown handler today; no app-scoped async Redis client yet

**Impact:** When an async Redis client is added (e.g. for streaming), failing to close it on shutdown leaks connections on restart.

**Fix direction:** When adding async Redis, create it in a FastAPI lifespan and call `await redis_client.aclose()` on shutdown:

@asynccontextmanager
async def lifespan(app):
    redis_client = aioredis.from_url(..., decode_responses=False)
    yield
    await redis_client.aclose()
app = FastAPI(lifespan=lifespan)

---

## Edge cases & follow-up (same theme, not in main 7)

These are additional async paths that still do sync Redis (or sync work). Fix in later PRs.

| Where | What blocks | Fix idea |
|-------|----------------|----------|
| **Resume session** | `conversations_router.py` L566–589: async `resume_session` calls `redis_manager.redis_client.exists(stream_key)` and `redis_manager.get_task_status(...)` (sync). | Offload with `asyncio.to_thread` for both, or add async Redis in this flow. |
| **get_active_session** | `conversations_router.py` L505: async handler calls `session_service.get_active_session(conversation_id)` (sync). Session service uses `redis_manager.redis_client.keys(pattern)`, `xrevrange`, `get_task_status`. | Run `get_active_session` in thread or add async Redis to session service. |
| **get_task_status (router)** | `conversations_router.py` L535: async handler calls `session_service.get_task_status(conversation_id)` (sync). Session service uses `redis_manager.redis_client.keys(pattern)` and multiple `get_task_status` in a loop. | Same as above: thread or async Redis. |
| **conversation_service** | `conversation_service.py` L1491: `self.session_service.get_active_session(conversation_id)` (sync) — caller may be async. | Offload or make session_service async. |

**Tests:** Existing mocks (`mock_redis_stream_manager`, patch of `wait_for_task_start`) still apply: `asyncio.to_thread(redis_manager.wait_for_task_start, ...)` invokes the same bound method, so the mock is still called and `assert_called_once()` passes. No test change required for the wait_for_task_start PR.

**Python:** `requires-python = ">=3.10"` — `asyncio.to_thread` is available (3.9+). No fallback needed.
