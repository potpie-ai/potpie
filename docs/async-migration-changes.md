# Async Migration — Detailed Change Log (Steps 1–4)

This document lists every change made as part of the sync-to-async migration for the FastAPI backend (Steps 1–4). Each section names the affected files, what was done, why it was needed, and the effect. Code is referenced only where a short snippet helps; the source of truth is the current codebase.

---

## Step 1: Async Redis for conversation streaming

**Goal:** Move all Redis operations used on the FastAPI request path (streaming, task status, session) off the event loop by introducing a dedicated async Redis client and async equivalents for every call.

### app/modules/conversations/utils/redis_streaming.py

**Need:** The sync `RedisStreamManager` uses blocking Redis calls (get, set, setex, xadd, xread, exists, delete, xrange, xrevrange). When used from FastAPI routes, those calls block the event loop and limit concurrency.

**Change:** A separate `AsyncRedisStreamManager` class was added (or equivalent async path) using `redis.asyncio`, with async methods for: get_task_status, set_task_status, set_task_id, get_task_id, publish_event, set_cancellation, get_stream_snapshot, clear_session, and wait_for_task_start. For wait_for_task_start, the async version uses `asyncio.sleep` instead of `time.sleep`. The existing sync `RedisStreamManager` remains for Celery workers.

**Effect:** FastAPI can use the async manager so Redis no longer blocks the event loop on the streaming and task-status paths.

### app/modules/conversations/utils/conversation_routing.py

**Need:** `start_celery_task_and_stream` and `start_celery_task_and_wait` (and any helpers that set task id, wait for task start, or ensure unique run id) were calling sync Redis. That made the whole request path blocking.

**Change:** `start_celery_task_and_stream` was made `async def` and updated to use the async Redis manager (e.g. `AsyncRedisStreamManager`) for set_task_id, wait_for_task_start, and any other Redis calls. An `async_ensure_unique_run_id` was added that uses the async Redis client to check stream existence and generate a unique run_id. Sync `ensure_unique_run_id` and sync `redis_stream_generator` are left as-is; Starlette runs the sync generator in a thread pool for `StreamingResponse`.

**Effect:** The HTTP path that starts a Celery task and returns a stream no longer blocks the event loop on Redis.

### app/modules/conversations/session/session_service.py

**Need:** Session/active-run resolution was using sync Redis (e.g. `keys()` or similar), which blocks under load.

**Change:** An `AsyncSessionService` was added that uses the async Redis manager and `scan` (or equivalent) instead of `keys()` for listing stream keys. Active session selection uses recency (e.g. XREVRANGE or numeric run-id comparison) instead of lexicographic sort. Exception handling in the active-session logic was tightened so Redis/connectivity errors are logged with `logger.exception` and not swallowed as “not found.”

**Effect:** GET active-session and related session endpoints use non-blocking Redis and avoid blocking the event loop.

### app/modules/conversations/conversations_router.py

**Need:** Routes that create conversations, post messages, get active session, get task status, or stop generation were passing or using sync Redis/session services.

**Change:** These routes were updated to obtain the async Redis stream manager and async session service (e.g. from app state or a dependency) and pass them into the controller. They also inject `get_async_db` where the conversation flow uses the database so that the DB path can use AsyncSession (see Step 2).

**Effect:** Conversation endpoints use the async Redis and DB path end-to-end.

### app/api/router.py

**Need:** API routes that call `ensure_unique_run_id` or touch Redis for conversation/stream setup were blocking.

**Change:** Those routes were switched to use the async Redis manager and `async_ensure_unique_run_id` (or the async session service where applicable).

**Effect:** Shared API entrypoints that trigger streaming or task status no longer block on Redis.

### app/modules/conversations/conversation/conversation_service.py

**Need:** `stop_generation` and any other conversation logic that reads/writes Redis (e.g. stream snapshot, clear session, cancellation) were using sync Redis when invoked from FastAPI.

**Change:** When an async Redis manager is available (e.g. injected from the router), `stop_generation` and related calls use the async manager’s methods (get_stream_snapshot, clear_session, set_cancellation, etc.). The service is structured so the FastAPI path gets the async dependencies and the Celery path keeps using the sync manager.

**Effect:** Stopping generation and cleaning up streams no longer blocks the event loop.

### app/main.py (Step 1)

**Need:** If the async Redis stream manager is created at startup, it must be available to all routes and not be None, so that failures are explicit.

**Change:** During startup, the app creates the async Redis stream manager (when Redis is configured) and stores it on app state. If creation fails, the app fails fast (e.g. raises or returns 503) instead of setting it to None, so dependency providers never receive a missing manager.

**Effect:** No silent AttributeErrors; routes either have a valid async Redis manager or fail clearly at startup.

---

## Step 2a: AsyncSession for UsageService

**Goal:** Usage checks (e.g. check_usage_limit) that run on create_conversation and post_message are moved to AsyncSession so they do not block the event loop.

### app/core/database.py

**Need:** FastAPI routes need a way to obtain an async DB session that is scoped to the request and uses the async engine/pool.

**Change:** An async engine is created with `create_async_engine` (postgresql+asyncpg). `AsyncSessionLocal` is a sessionmaker bound to that engine with `class_=AsyncSession` and `expire_on_commit=False`. The `get_async_db` dependency is an async generator that yields an `AsyncSession` and closes it when the request ends.

**Effect:** Any route that depends on `get_async_db` gets a request-scoped AsyncSession and does not block the pool with sync calls.

### app/modules/usage/usage_service.py

**Need:** `check_usage_limit` and any usage data fetching are called on hot paths (create conversation, post message). If they use sync Session and blocking queries, they limit throughput.

**Change:** UsageService was migrated to use AsyncSession: methods that run from FastAPI (e.g. get_usage_data, check_usage_limit) accept or use an async session and perform queries via `await session.execute(select(...))` instead of `session.query(...)`. The service is invoked from controllers that receive the async session from the router.

**Effect:** Usage checks run as non-blocking DB work on the async path.

### app/modules/usage/usage_controller.py and usage_router.py

**Need:** Usage endpoints must pass the async session into the usage service.

**Change:** The usage endpoint(s) that trigger check_usage_limit or get_usage_data depend on `get_async_db` and pass the yielded AsyncSession to the controller/service.

**Effect:** Usage API uses the async DB path consistently.

---

## Step 2b: AsyncChatHistoryService (dual path)

**Goal:** Chat history (get_session_history, flush_message_buffer, save_partial_ai_message) is used from both FastAPI (store_message, stop_generation, streaming) and Celery (agent tasks). The FastAPI path must use AsyncSession without changing Celery.

### app/modules/conversations/conversation/ (chat history / message store)

**Need:** ChatHistoryService (or equivalent) does DB reads/writes for messages. When called from FastAPI, those must be async to avoid blocking.

**Change:** An `AsyncChatHistoryService` was added that takes an AsyncSession and implements async get_session_history, flush_message_buffer, and save_partial_ai_message (and add_message_chunk for in-memory buffering). The existing sync ChatHistoryService remains for Celery. ConversationService.create() was updated to accept an optional async_db (and optionally async_redis_manager, async_session_service). When async_db is provided (FastAPI path), the service builds and uses AsyncChatHistoryService; when not (Celery path), it uses the sync ChatHistoryService.

**Effect:** FastAPI message flow uses non-blocking DB for history; Celery keeps using the sync service. The same ConversationService supports both callers.

### app/modules/conversations/conversation/conversation_controller.py

**Need:** The controller must pass the async session and async Redis/session services into ConversationService when handling HTTP requests.

**Change:** The controller’s constructor or factory accepts `async_db: AsyncSession` in addition to `db: Session`. When creating the conversation service for FastAPI, it passes async_db (and async Redis/session services) so that the service uses AsyncChatHistoryService and async Redis on that path.

**Effect:** All conversation routes that go through this controller use the async history and Redis path.

### app/modules/conversations/conversation/conversation_service.py

**Need:** Type hints and optional dependencies must allow both sync and async managers without import errors.

**Change:** The module uses `from __future__ import annotations` (or equivalent) so that forward references to AsyncRedisStreamManager, AsyncSessionService, and AsyncChatHistoryService are deferred and do not cause NameError when those classes are not imported in the same module. The service’s create() and __init__ accept optional async dependencies and dispatch to async or sync implementations accordingly.

**Effect:** Clean dual-path support without circular imports or undefined names.

---

## Step 2c: AsyncShareChatService and AsyncUserService

**Goal:** Share/access and user lookups used by FastAPI (share chat, get shared emails, remove access, API key user) use AsyncSession.

### app/modules/conversations/access/access_service.py

**Need:** ShareChatService performs DB queries (Conversation visibility, shared_with_emails). When used from FastAPI, those should be async.

**Change:** An `AsyncShareChatService` class was added that takes an AsyncSession and implements async share_chat, get_shared_emails, and remove_access using `select()` and `await session.execute()` (and commit where needed). The sync ShareChatService remains for any code that still uses a sync session.

**Effect:** Share and access endpoints no longer block the event loop on DB.

### app/modules/conversations/conversations_router.py (share/access)

**Need:** Share and access routes must use the async DB and async service.

**Change:** The share_chat, get_shared_emails, and remove_access route handlers depend on `get_async_db` and instantiate `AsyncShareChatService(async_db)` (or receive it via dependency). They call the async methods and return the results.

**Effect:** Share and access are fully async on the HTTP path.

### app/modules/users/user_service.py

**Need:** User lookups (get_user_by_uid, get_user_by_email, get_user_ids_by_emails, create_user, update_last_login) are used from API key auth and auth routes. When called from FastAPI, they must not block.

**Change:** An `AsyncUserService` class was added that takes an AsyncSession and implements async get_user_by_uid, get_user_id_by_email, get_user_by_email, get_user_ids_by_emails, create_user, and update_last_login using SQLAlchemy 2.0 style select/execute. The sync UserService remains for Celery and other sync callers.

**Effect:** User lookups from FastAPI use the async pool.

### app/api/router.py

**Need:** API key validation (get_api_key_user) resolves the user from the DB; that must be async when used in FastAPI middleware or route dependencies.

**Change:** The API key dependency (or the code that resolves the user from the key) uses `get_async_db` and `AsyncUserService(async_db).get_user_by_uid` instead of the sync UserService.

**Effect:** API key auth does not block the event loop on DB.

---

## Step 2d: Auth routes (signup, SSO, get_my_account)

**Goal:** Auth endpoints that look up or update users use AsyncSession for DB work.

### app/modules/auth/auth_router.py

**Need:** signup, sso_login, and get_my_account (or equivalent) call user lookup and update_last_login. Those should use AsyncSession when available.

**Change:** The affected auth route handlers were updated to depend on `get_async_db` and to use `AsyncUserService(async_db)` for get_user_by_uid, get_user_by_email, and update_last_login. UnifiedAuthService or other auth logic that remains sync is unchanged; only the DB access for user records is async.

**Effect:** Auth routes that touch the user table no longer block the event loop on DB.

---

## Step 2e: GithubService DB queries (optional AsyncSession)

**Goal:** get_repos_for_user and get_combined_user_repos can be called from FastAPI (e.g. github_router). When an AsyncSession is provided, they should use it so that OAuth token lookup and user/provider queries do not block.

### app/modules/code_provider/github/github_service.py

**Need:** get_repos_for_user (and thus get_combined_user_repos) does user and UserAuthProvider lookups and possibly OAuth token resolution via DB. If that code uses self.db (sync) from an async route, it blocks the event loop.

**Change:** get_repos_for_user and get_combined_user_repos were updated to accept an optional `async_session` parameter. When it is provided, user and provider lookups use `await async_session.execute(select(...))`. An async helper (e.g. async_get_github_oauth_token) was added that takes an AsyncSession and performs the token lookup with async execute, so that no sync self.db.query runs on the async path. When async_session is not provided (e.g. Celery or sync callers), the methods fall back to self.db as before.

**Effect:** When the GitHub router passes `async_db` into these methods, repo listing and OAuth resolution no longer block the event loop.

### app/modules/code_provider/code_provider_controller.py and github_router

**Need:** The FastAPI endpoint that returns user repos must pass the async session into GithubService.

**Change:** The controller method that calls get_combined_user_repos (or get_user_repos) receives the async session from the router (via get_async_db) and passes it into the service. The github_router injects get_async_db and passes the yielded session to the controller.

**Effect:** GET /github/user-repos (and related) use the async DB path.

---

## Step 3: Async Redis for non-streaming services

**Goal:** Tunnel and GitHub/branch cache use Redis for simple get/set/scan. Those operations are moved to redis.asyncio on the FastAPI path so they do not block.

### app/modules/tunnel/tunnel_service.py

**Need:** Tunnel routes use Redis for workspace tunnel records (get/set) and for listing user tunnels. Sync get/setex and keys() block the event loop.

**Change:** The service keeps the sync redis_client for Celery and other sync callers. An optional `_async_redis_client` is created from `redis.asyncio` when Redis URL is configured. get_workspace_tunnel_record and set_workspace_tunnel_record (sync) were updated to use setex with a TTL constant (WORKSPACE_TUNNEL_RECORD_TTL, 24 hours). New async methods were added: get_workspace_tunnel_record_async and set_workspace_tunnel_record_async use the async client when available, otherwise they fall back to asyncio.to_thread(sync_method, ...). list_user_tunnels (sync) was changed to use scan_iter with a count hint instead of keys(); list_user_tunnels_async was added to use the async client’s scan_iter when available, else asyncio.to_thread(list_user_tunnels, ...).

**Effect:** Tunnel routes that read workspace records or list tunnels no longer block the event loop when async Redis is available.

### app/modules/tunnel/tunnel_router.py

**Need:** Routes must call the async tunnel methods so that Redis is non-blocking.

**Change:** get_workspace_tunnel and get_workspace_socket_status call `await tunnel_service.get_workspace_tunnel_record_async(workspace_id)` instead of the sync getter.

**Effect:** Tunnel endpoints are fully async on the Redis path.

### app/modules/code_provider/github/github_service.py (project structure cache)

**Need:** get_project_structure_async caches results in Redis. Using the sync self.redis.get/setex from an async method blocks the event loop.

**Change:** A lazy, module-level async Redis client was introduced: _async_redis_cache and _get_async_redis_cache(). Creation is guarded by an asyncio.Lock (_async_redis_cache_lock) so only one task creates the client; after acquiring the lock, the code re-checks _async_redis_cache to avoid duplicate creation. get_project_structure_async uses _get_async_redis_cache() for cache get and setex when the client is available. If async Redis is unavailable, the sync fallback (self.redis.get/setex) is wrapped in asyncio.to_thread so it still does not block the event loop. Redis read/write is wrapped in try/except for RedisError and OSError: on failure the code logs (with cache_key and context), treats read as cache miss (cached_structure = None), and does not propagate write failures to the caller. close_github_async_redis_cache() was added to aclose the global client and clear the global; it is invoked from app shutdown (see main.py).

**Effect:** Project structure cache is non-blocking and resilient to transient Redis failures; no connection leak at shutdown.

### app/main.py (shutdown)

**Need:** The global async Redis cache in github_service must be closed when the app shuts down to avoid connection leaks on restart.

**Change:** An async shutdown_event was added that imports and awaits close_github_async_redis_cache(). The shutdown event is registered with the FastAPI app (add_event_handler("shutdown", shutdown_event)).

**Effect:** Async Redis cache is closed cleanly on restart/redeploy.

### app/modules/code_provider/branch_cache.py

**Need:** Branch list endpoint uses BranchCache for Redis get/set. Sync get/set from FastAPI blocks the event loop.

**Change:** The constructor creates an optional _async_redis_client from redis.asyncio when Redis URL is configured and the sync client is available. Redis URL is validated before calling redis.from_url (guard for None URL). get_branches_async was added: when _async_redis_client is set, it uses await self._async_redis_client.get(cache_key); otherwise it uses asyncio.to_thread(self.get_branches, ...). An async aclose() method was added that closes both _async_redis_client (aclose) and redis_client (close) with try/except and warning log, and sets available = False. Callers (e.g. app shutdown or long-lived cache instances) can call aclose() to avoid connection leaks.

**Effect:** Branch list endpoint uses non-blocking Redis when the async client is available; cache can be closed cleanly.

### app/modules/code_provider/code_provider_controller.py

**Need:** The branch list API must use the async cache method.

**Change:** get_branch_list was updated to call `await self.branch_cache.get_branches_async(repo_name, search_query)` instead of the sync get_branches.

**Effect:** Branch list endpoint is async on the cache path.

---

## Step 4: Async HTTP clients and sync-only SDKs

**Goal:** All HTTP calls and third-party SDK calls made from FastAPI use non-blocking I/O: either native async (httpx.AsyncClient) or offload to a thread (asyncio.to_thread / run_in_executor) so the event loop is not blocked.

### app/modules/auth/auth_service.py

**Need:** Login calls Firebase Identity Toolkit over HTTP. Blocking requests or broad exception handling would block the event loop or hide status codes.

**Change:** Sync login uses httpx.Client with a fixed timeout (LOGIN_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0)). The client.post call is inside the same try block as raise_for_status so that connection/timeout/transport errors (httpx.HTTPError) are caught and mapped to HTTPException. login_async was added using httpx.AsyncClient with the same timeout; the route calls await auth_handler.login_async. HTTPStatusError is caught and re-raised as HTTPException with the same status_code and response body so callers can distinguish 401 vs 500. HTTPError is caught and re-raised as HTTPException(502, "Upstream auth request failed"). Logging in the auth handlers was made safe: HTTPStatusError logs only log_prefix and e.response.status_code; HTTPError logs a generic message (no str(e)) to avoid leaking the Identity Toolkit URL or API key in logs.

**Effect:** Login is non-blocking on the FastAPI path; transport errors and status codes are preserved; logs do not leak secrets.

### app/modules/auth/auth_router.py

**Need:** The login route must use the async login method. Slack signup notifications must not block.

**Change:** The login endpoint calls `await auth_handler.login_async(...)`. send_slack_message was changed to use httpx.AsyncClient with a timeout (e.g. 10s) and `await client.post(...)` instead of requests.

**Effect:** Login and Slack signup notification are non-blocking.

### app/modules/utils/parse_webhook_helper.py

**Need:** Parsing failure Slack notifications are sent from async code; blocking HTTP would block the event loop.

**Change:** send_slack_notification uses httpx.AsyncClient with a timeout and await client.post(...). Request body is built with json.dumps and Content-Type application/json. Errors are logged with logger.warning instead of print.

**Effect:** Parse webhook Slack alerts are non-blocking and use consistent logging.

### app/modules/intelligence/tools/linear_tools/linear_client.py

**Need:** Linear GraphQL calls (get issue, update issue, create comment) are used from agent tools that can run in async context. Blocking HTTP would block the event loop.

**Change:** The client uses httpx with a single timeout constant (LINEAR_REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0)). Sync execute_query uses httpx.Client; execute_query_async was added using httpx.AsyncClient. get_issue_async, update_issue_async, and comment_create_async were added and call execute_query_async. The Linear tools (get_linear_issue_tool, update_linear_issue_tool) were updated to call these async methods from their arun implementations.

**Effect:** Linear API calls from the async tool path are non-blocking.

### app/modules/utils/email_helper.py

**Need:** Resend SDK (resend.Emails.send) is sync-only. Calling it from async routes would block the event loop.

**Change:** send_email and send_parsing_failure_alert were made async and wrap the call as `await asyncio.to_thread(resend.Emails.send, params)`. resend.api_key is set only when self.api_key is truthy to avoid overwriting with None.

**Effect:** Email sending does not block the event loop; API key is set safely.

### app/modules/utils/posthog_helper.py

**Need:** PostHog SDK (posthog.capture) is sync-only. When send_event is called from async routes, it must not block.

**Change:** A sync helper _capture_sync was added that performs the capture and handles exceptions. send_event checks for a running event loop (asyncio.get_running_loop()). If present, it schedules _capture_sync in the default executor (run_in_executor(None, lambda: ...)) as fire-and-forget; otherwise it calls _capture_sync directly so that sync callers (e.g. Celery) still work.

**Effect:** PostHog events from async context do not block the event loop; sync context still works.

---

## Cross-cutting and edge-case fixes

**app/core/database.py:** Async engine and AsyncSessionLocal are used by all async DB migrations. get_async_db is the single dependency for request-scoped AsyncSession. create_celery_async_session exists for Celery tasks that need an isolated async connection (NullPool) to avoid cross-task Future binding issues.

**app/modules/code_provider/branch_cache.py:** Redis URL is checked before calling redis.from_url to avoid TypeError when Redis is not configured. aclose() allows long-lived or shared BranchCache instances to release connections (call from app shutdown or context manager if applicable).

**app/modules/auth/auth_router.py:** Imports were reordered so that module-level imports are at the top (E402); AuthProviderCreate is consolidated in the auth_schema import list.

**app/modules/code_provider/github/github_service.py:** F541 fix: a log message that used an f-string with no placeholders was changed to a normal string. RedisError is imported and used in get_project_structure_async for the cache read/write guards.

**app/modules/conversations/conversation/conversation_service.py:** Uses `from __future__ import annotations` so that optional async dependencies in type hints (e.g. AsyncRedisStreamManager, AsyncSessionService, AsyncChatHistoryService) do not cause NameError at import time when those types are not imported in the same file.

---

## Summary by file (quick reference)

| Area | Files touched | Main change |
|------|----------------|------------|
| Step 1 Redis streaming | redis_streaming.py, conversation_routing.py, session_service.py, conversations_router.py, api/router.py, conversation_service.py, main.py | Async Redis manager + async session service; async start_celery_task_and_stream and ensure_unique_run_id; fail-fast init for async manager |
| Step 2a Usage | database.py, usage_service.py, usage_controller.py, usage_router.py | AsyncSession + get_async_db; UsageService async methods |
| Step 2b Chat history | conversation_service.py, conversation_controller.py, conversations_router.py, chat history / message store modules | AsyncChatHistoryService; dual-path ConversationService with optional async_db |
| Step 2c Share/User | access_service.py, user_service.py, conversations_router.py, api/router.py | AsyncShareChatService, AsyncUserService; share/access and API key user use get_async_db |
| Step 2d Auth DB | auth_router.py | get_async_db + AsyncUserService on signup, sso_login, get_my_account |
| Step 2e GitHub DB | github_service.py, code_provider_controller.py, github_router.py | Optional async_session in get_repos_for_user/get_combined_user_repos; async_get_github_oauth_token; router passes async_db |
| Step 3 Tunnel/cache | tunnel_service.py, tunnel_router.py, github_service.py, branch_cache.py, code_provider_controller.py, main.py | Async Redis for tunnel records and list; lazy locked async Redis cache for project structure + to_thread fallback + Redis error guards + shutdown close; BranchCache async client + get_branches_async + aclose + redis_url guard |
| Step 4 HTTP/SDKs | auth_service.py, auth_router.py, parse_webhook_helper.py, linear_client.py, email_helper.py, posthog_helper.py | httpx sync/async for auth and Slack; Linear execute_query_async and tool arun; asyncio.to_thread for Resend; run_in_executor for PostHog |

This document reflects the intended design and the changes present in the codebase for the async migration (Steps 1–4). For exact branch/PR mapping, see the async-migration-plan and the commit history on the feature branches.
