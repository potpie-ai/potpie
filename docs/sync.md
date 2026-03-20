# Sync-in-Async Scaling Audit

Critical list of remaining blocking calls in async paths. Each item blocks the FastAPI event loop and can stall other requests under load.

**Scope:** `app/` only. Celery workers are sync by design; this doc focuses on FastAPI async routes and services they call.

---

## Critical (hot paths — fix first)

| # | Location | What blocks | Fix |
|---|----------|-------------|-----|
| 1 | **conversation_routing.py** — `redis_stream_generator`, `start_celery_task_and_stream` | Sync Redis (`set_task_status`, `publish_event`, `set_task_id`, `wait_for_task_start`), then sync generator that calls `redis_manager.consume_stream()` (XREAD block). Used for every streaming chat. | Use `redis_stream_generator_async` (thread + queue); wrap all Redis and `wait_for_task_start` in `asyncio.to_thread`. |
| 2 | **conversations_router.py** L288, L365, **api/router.py** L183 | `ensure_unique_run_id(conversation_id, run_id)` — sync Redis loop in async route. | `await asyncio.to_thread(ensure_unique_run_id, conversation_id, run_id)`. |
| 3 | **conversations_router.py** L294, L364, **api/router.py** L215 | `return start_celery_task_and_stream(...)` — returns sync `StreamingResponse(redis_stream_generator(...))`. Generator runs on event loop when client reads. | Call async `start_celery_task_and_stream` that uses `redis_stream_generator_async` and `await asyncio.to_thread` for Redis; return `StreamingResponse(async_generator)`. |
| 4 | **conversation_service.py** — `store_message`, `_generate_and_stream_*`, `stop_generation`, etc. | `history_manager.add_message_chunk`, `flush_message_buffer`, `get_session_history`, `save_partial_ai_message` — all sync DB (ChatHistoryService uses `self.db.query`, `self.db.add`, `self.db.commit`). | Offload to thread: `await asyncio.to_thread(history_manager.flush_message_buffer, ...)` etc., or move to async session. Same for any sync DB in conversation_service. |
| 5 | **conversations_router.py** L506, L537 | `session_service.get_active_session(conversation_id)` and `session_service.get_task_status(conversation_id)` — sync Redis (`keys`, `exists`, `xinfo_stream`, `xrevrange`, `get_task_status`). | `await asyncio.to_thread(session_service.get_active_session, conversation_id)` and same for `get_task_status`. |
| 6 | **conversations_router.py** L582 (resume_session) | `redis_manager.get_task_status(conversation_id, session_id)` — sync Redis in async route. | `await asyncio.to_thread(redis_manager.get_task_status, conversation_id, session_id)`. |
| 7 | **usage_service.py** L17–32 | `get_usage_data` is `async` but uses `with SessionLocal() as session` and `session.query(...).all()` — sync DB on event loop. Called from `check_usage_limit` on every create/post_message. | Use async session (e.g. `AsyncSession`) or run the sync query in `asyncio.to_thread`. |
| 8 | **github_service.py** L1115–1172 | `get_project_structure_async`: `self.redis.get(cache_key)`, `self.redis.setex(...)` and sync `self.get_repo(repo_name)` / `repo.get_contents(path)` in async method. | `await asyncio.to_thread(self.redis.get, cache_key)`; same for setex; offload `get_repo`/repo calls or use async GitHub client. |
| 9 | **github_service.py** L935–950 | `get_combined_user_repos`: `self.db.query(Project)...` and `.all()` in async method. Called from code_provider_controller.get_user_repos. | Run the DB query in `asyncio.to_thread` or use async session. |
| 10 | **tunnel_router.py** L46, L50, L70, L72 | `tunnel_service.get_workspace_tunnel_record(workspace_id)` and `get_socket_service().is_workspace_online(workspace_id)` — sync Redis in async route. | `await asyncio.to_thread(tunnel_service.get_workspace_tunnel_record, workspace_id)` and `await asyncio.to_thread(get_socket_service().is_workspace_online, workspace_id)`. |

---

## High (auth, sharing)

| # | Location | What blocks | Fix |
|---|----------|-------------|-----|
| 11 | **auth_router.py** L91–258+ (signup) | Sync DB throughout: `db.expire_all()`, `user_service.get_user_by_uid(link_to_user_id)`, `db.query(UserAuthProvider)`, `db.commit()`, `db.rollback()`. | Offload DB blocks: `await asyncio.to_thread(user_service.get_user_by_uid, ...)`, and run commit/rollback/query blocks in a single sync helper in thread. |
| 12 | **unified_auth_service.py** L419 | `get_user_providers(existing_user.uid)` — sync DB, called from async `authenticate_or_create`. | `await asyncio.to_thread(self.get_user_providers, existing_user.uid)`. |
| 13 | **share_chat_service.py** L29–89, L80–89, L96–99 | `share_chat`, `get_shared_emails`, `remove_access` are async but use `self.db.query(...).first()`, `self.db.commit()`, `self.db.rollback()`. | Run each sync block in `asyncio.to_thread` or switch to async session. |

---

## Medium (external HTTP / other services)

| # | Location | What blocks | Fix |
|---|----------|-------------|-----|
| 14 | **parse_webhook_helper.py** L19 | `requests.post(self.url, ...)` in async `send_slack_notification`. Timeout may be set; still blocks. | `await asyncio.to_thread(requests.post, ...)` or use httpx.AsyncClient. |
| 15 | **linear_client.py** L50 | `requests.post(self.API_URL, ...)` — no timeout. Called from async tools (get_linear_issue_tool, update_linear_issue_tool) via `client.*` after `await get_linear_client_for_user`. | Add timeout; wrap the call in `asyncio.to_thread` at the tool’s `_arun` or make LinearClient use httpx.AsyncClient. |
| 16 | **email_helper.py** L77, L170 | `resend.Emails.send(params)` in async `send_email` and `send_parsing_failure_alert`. | `await asyncio.to_thread(resend.Emails.send, params)` (or use async Resend client if available). |
| 17 | **provider_factory.py** L215 | `requests.get(url, ...)` in `create_github_app_provider`. Used when creating provider from async-triggered paths (e.g. code_provider_controller, tools). | If caller is async, offload: `await asyncio.to_thread(CodeProviderFactory.create_github_app_provider, repo_name)`. Timeout already 60. |
| 18 | **github_service.py** L442 | `requests.get(orgs_url, ...)` in sync method (get_repos_for_user path). If that path is ever invoked from an async route without to_thread, it blocks. | Ensure callers use to_thread; add timeout if missing. |

---

## Lower (analytics, non-critical path)

| # | Location | What blocks | Fix |
|---|----------|-------------|-----|
| 19 | **posthog_helper.py** — `PostHogClient.send_event` | `self.posthog.capture(...)` — sync HTTP. Called from async routes (auth, conversation, parsing, provider, key_management). | `await asyncio.to_thread(self.posthog.capture, ...)` or fire-and-forget in thread so request path isn’t blocked. |
| 20 | **conversation_routing.py** — `start_celery_task_and_wait` | Before `run_in_executor(collect_from_stream)`: `set_task_status`, `publish_event`, `set_task_id`, `wait_for_task_start` — all sync Redis. | Wrap the Redis + wait block in `asyncio.to_thread` so only the stream collection runs in executor. |

---

## Already fixed on this branch (reference)

- **auth_router**: login and send_slack_message — `asyncio.to_thread` + timeout.
- **parsing_controller**: fetch_parsing_status — DB query via `asyncio.to_thread(_query_parsing_status_sync, ...)`.
- **parse_webhook_helper**: timeout=30 on requests.post (offload still recommended).
- **github_service**: get_github_repo_details (installation) — timeout=30.

---

## Summary

- **Critical:** Streaming and session Redis, ensure_unique_run_id, history_manager/DB in conversation_service, session_service in router, resume_session Redis, usage_service sync DB, github_service Redis and DB in async methods.
- **High:** Signup and auth DB, get_user_providers, ShareChatService DB, parsing_controller fetch_parsing_status (if not yet offloaded).
- **Medium:** Slack/Linear/email HTTP, provider_factory and github_service HTTP when called from async.
- **Lower:** PostHog, remaining sync Redis in start_celery_task_and_wait.

**Tunnel service (when offloaded):** `tunnel_service.get_workspace_tunnel_record` and internal helpers use `redis_client.keys()` and `set(key, value)` without TTL. Prefer `scan_iter(match=...)` instead of `keys()` and `setex(key, ttl, value)` for workspace records so the worker thread doesn’t block Redis.

No bullshit: any sync I/O (DB, Redis, HTTP, file) in an `async def` path blocks the event loop until it completes. Under concurrency, that stalls all other requests on the same process. Fix by offloading to `asyncio.to_thread`, using async drivers (AsyncSession, httpx, redis.asyncio), or moving work to a worker (e.g. Celery).

Last updated: critical audit of current codebase (fix/sync-timeouts-only).
