# Celery tasks: async/sync and DB usage rules

## Why this matters

Tasks that use `run_async()` (e.g. agent and regenerate) run a coroutine via `asyncio.run()`, which creates a **fresh event loop per task** and tears it down when the coroutine finishes. Doing **sync** DB or other blocking I/O **inside** that coroutine:

- Blocks the event loop (no other async work can run).
- In forked workers, can hang indefinitely: the sync engine’s connection pool is process-inherited; using `SessionLocal()` or a second sync session inside the coroutine can wait on a bad or contended pool.

So: **do all sync DB and blocking I/O before `run_async()`, and pass results into the coroutine.**

## Guarantee (Celery agent/regenerate)

- **`execute_agent_background`** and **`execute_regenerate_background`**: All sync DB (user resolution) is done **before** `run_async()`. The coroutines do **not** call `SessionLocal()` or use sync DB; they only use `self.async_db()` and the pre-resolved `user_email`. These two tasks will not hang from sync-in-async in the coroutine.
- **`process_parsing`**: Still runs `ParsingService.parse_directory` (async) which internally uses sync DB (e.g. `ProjectService`). That blocks the event loop but does not create a second session in the coroutine, so it does not exhibit the same "stuck" hang. Prefer moving sync work before `run_async()` or into `run_in_executor` in a later refactor.

## How to confirm / find remaining issues

Run the audit script from the repo root:

```bash
python scripts/audit_sync_in_async.py
```

It lists **async functions that contain sync DB usage** (e.g. `self.db.query`, `SessionLocal()`, `.commit()`), excluding async session usage (`async_db`, `AsyncSessionLocal`) and a small blocklist. Exit code 1 means at least one such usage was found. Many are in **FastAPI** async route handlers; they block the event loop but do not run in a forked Celery worker. Use the task table below to interpret which paths are Celery vs FastAPI.

## Rules (enforced by convention + BaseTask docstring)

1. **Before `run_async(coro)`**
   - Resolve anything that needs sync DB (e.g. user, config) using `self.db` only.
   - Do not call `SessionLocal()` or create another sync session in the task body if that session would later be used from the coroutine (or from code that runs on the same thread as the coroutine).

2. **Inside the coroutine passed to `run_async()`**
   - `run_async()` uses `asyncio.run(coro)`, so each task gets a fresh event loop; no stale callbacks or context leakage between tasks.
   - Use **only** `async with self.async_db() as session` for DB access.
   - Do **not** call `SessionLocal()` or use `self.db` for new work. You may pass `self.db` into stores/services that are designed to hold a sync session (e.g. `ConversationStore(self.db, async_db)`) as long as those stores do not perform sync DB from the async call path in a blocking way (they should use `run_in_executor` for sync DB if needed).

3. **Shared helpers**
   - Use `_resolve_user_email_for_celery(db, user_id)` in `agent_tasks` for any task that needs user email before `run_async()`. Keeps user resolution in one place and avoids sync DB inside the coroutine.

## Tasks audited

| Task | Uses run_async? | Sync DB before run_async? | Sync DB inside coroutine? |
|------|-----------------|---------------------------|---------------------------|
| `execute_agent_background` | Yes | Yes (user via `_resolve_user_email_for_celery`) | No |
| `execute_regenerate_background` | Yes | Yes (user via `_resolve_user_email_for_celery`) | No |
| `process_parsing` | Yes | N/A (only passes `self.db` into `ParsingService`) | Yes, indirectly: `ParsingService.parse_directory` uses sync DB (e.g. `ProjectService`) inside async. Prefer moving those calls to sync-before-run_async or run_in_executor in a later refactor. |

## Remaining sync-in-async (audit summary)

Running `python scripts/audit_sync_in_async.py` reports async functions that contain sync DB calls. As of the last audit:

- **Celery**: Only `process_parsing` still has sync DB inside the async path (`ParsingService.parse_directory`). No `SessionLocal()` in the coroutine, so no hang like agent/regenerate; the event loop is still blocked during those calls.
- **FastAPI / other**: Many async route handlers and services (auth, integrations, media, projects, prompts, usage, etc.) call sync DB directly. They block the event loop but do not run in a forked worker. Refactoring to async DB or `run_in_executor` is recommended over time.

Safe patterns: **MessageService** and **ConversationService** use `run_in_executor` for sync DB; **ConversationStore** / **MessageStore** use `async_db` only in their async methods.

## Other code paths (non-Celery)

- **ConversationService / MessageService**: Sync DB in async methods is wrapped in `run_in_executor` (e.g. `_sync_create_message`, `_sync_mark_message_archived`), so the event loop is not blocked.
- **ConversationService._clone_repo_if_missing**: Sync work (including `GithubService(self.db).get_github_oauth_token`) runs in `_clone_repo_if_missing_sync` via `run_in_executor`, so it does not block the loop.
- **ProjectService**: Several “async” methods (e.g. `get_project_from_db_by_id`, `get_project_name`) perform sync `self.db.query()` and thus block the event loop when called from async code. Acceptable for now when used from FastAPI or low-concurrency paths; if used from Celery async coroutines, prefer resolving data before `run_async()` or offloading to run_in_executor.

## Adding a new task that uses run_async()

1. Do all sync DB and blocking I/O in the task body (before `run_async()`).
2. Pass the results into the coroutine via closure or arguments.
3. Inside the coroutine, use only `self.async_db()` for DB; do not call `SessionLocal()` or use `self.db` for new queries.
4. If you need user email, use `_resolve_user_email_for_celery(self.db, user_id)` before `run_async()`.
