# Sync/Async and GitHub Blocking in Hot Paths

## Summary

**Parsing** and **conversations** are the most-called APIs; **GitHub** is the slowest 3rd-party dependency. This doc covers sync/async and blocking issues in those hot paths and the fixes applied.

The parsing and parsing-status endpoints run in **async** FastAPI handlers but previously called code that performed **synchronous** I/O (DB and GitHub API), blocking the event loop and making the server unresponsive under load.

---

## 1. Routes use sync `Session` in async handlers

- **`POST /api/v1/parse`** and **`GET /api/v1/parsing-status/{project_id}`** use `db: Session = Depends(get_db)` (sync SQLAlchemy session).
- The controller methods are `async` and `await` helpers that ultimately do **sync** DB and HTTP.

**Location:** `app/api/router.py` (e.g. `parse_directory`, `get_parsing_status`).

---

## 2. `ProjectService`: async methods with sync DB

These methods are declared `async` but perform **blocking** DB work with no `run_in_executor` / `asyncio.to_thread`:

| Method | Blocking calls |
|--------|----------------|
| `get_project_from_db` | `self.db.query(Project).filter(...).first()` |
| `register_project` | `self.db.query(...).first()`, `self.db.commit()`, `ProjectService.create_project(self.db, ...)` |
| `update_project_status` | `ProjectService.update_project(self.db, ...)` |
| `get_project_from_db_by_id` | `ProjectService.get_project_by_id(self.db, project_id)` |

**Location:** `app/modules/projects/projects_service.py`.

**Effect:** Every parse or parsing-status request blocks the single event-loop thread for the duration of the DB round-trip.

---

## 3. `ParseHelper.check_commit_status`: sync GitHub in async method

`check_commit_status` is `async` but:

- Calls `await self.project_manager.get_project_from_db_by_id(project_id)` (which is async but does sync DB inside).
- Then calls **sync** `self.github_service.get_repo(repo_name)` (GitHub API) with **no** `await` → blocks the event loop.
- Then **sync** `repo.get_branch(branch_name)` → blocks again.

**Location:** `app/modules/parsing/graph_construction/parsing_helper.py` (around 1910–1924).

**Effect:** Under concurrent parsing-status or parse traffic, GitHub latency blocks the loop and can delay or starve the request that should enqueue the parsing task.

---

## 4. `CodeProviderService.get_repo`: sync only

- `get_repo()` is synchronous (provider creation, `get_repository()`, `get_repo()`, etc.).
- Used from async paths (e.g. `ParseHelper.check_commit_status`, and any async code that calls it) without being offloaded to a thread.

**Location:** `app/modules/code_provider/code_provider_service.py`.

---

## 5. Impact on “parse never enqueued”

- The server **does** accept requests (we see parsing-status and get-branch-list for `cleder/awesome-python-testing`).
- If the event loop is blocked on sync DB + GitHub in status checks and branch list, a later **POST /parse** can be delayed or time out before it logs “Submitting parsing task” and calls `process_parsing.delay()`.
- So sync/async blocking can contribute to the parse request never (or rarely) being processed in time, even when the client does send it.

---

## 6. Code provider (GitHub) hot paths

| Endpoint | Issue | Fix |
|----------|--------|-----|
| **GET /github/get-branch-list** | Sync GitHub (list_branches + fallbacks) in async handler. | Run via `asyncio.to_thread(_fetch_branches_with_fallbacks_sync)`. |
| **GET /github/check-public-repo** | Sync `provider.get_repository()`. | Run via `asyncio.to_thread(_check_public_repo_sync)`. |

**Follow-up:** `get_user_repos` when not using GitHub App uses sync `list_user_repositories()`; consider to_thread.

---

## 7. Conversation hot paths

- **create_conversation** / **message_service** / **start_celery_task_and_wait**: already use `run_in_executor` for sync work.
- **ensure_unique_run_id** (sync Redis loop) and **redis_stream_generator** (sync consume_stream): potential blockers if called from async route; consider thread or async Redis.

---

## 8. Other GitHub call sites

- ParsingHelper clone/get_repo in Celery workers: OK (sync context). Intelligence tools and CodeChangesManager: if called from async path, consider to_thread for sync GitHub.

---

## 9. Fixes applied (see code)

1. **ParseHelper.check_commit_status**  
   - Run the sync GitHub work (`get_repo`, `get_branch`) in `asyncio.to_thread()` so it does not block the event loop.

2. **ProjectService**  
   - Run sync DB work in `asyncio.to_thread()` using a **new** session per call (session not shared across threads), so the event loop is not blocked and SQLAlchemy’s “session not thread-safe” constraint is respected.

3. **Optional follow-up**  
   - Use `get_async_db` and `AsyncSession` for parsing and parsing-status routes and implement async DB access there for a fully non-blocking path.
