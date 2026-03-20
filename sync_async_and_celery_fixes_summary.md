# Summary of Sync/Async and Celery Fixes

Handoff document for continuing work. All changes address **event-loop blocking** in production (parsing/conversation APIs failing under load) and **Celery task exception handling**.

---

## Problem Context

- **Production issue**: Parsing failed for repos (e.g. `cleder/awesome-python-testing`). Logs showed "will reparse" but no "Submitting parsing task" or Celery execution.
- **Root cause**: Sync DB and GitHub API calls blocked the FastAPI event loop. Under load, the server couldn't process parse requests in time.
- **Celery issue**: `ParsingService` raised `HTTPException` on errors; Celery cannot pickle it, causing `UnpickleableExceptionWrapper`.

---

## 1. `app/modules/projects/projects_service.py`

**Pattern**: All `async` methods that used `self.db.query()` directly were blocking the event loop. Each was refactored to run the sync DB logic in `asyncio.to_thread(_run)` with a dedicated `SessionLocal()` session created inside the thread.

**Methods fixed** (9 total):

| Method | Change |
|--------|--------|
| `get_project_name` | Wrapped `db.query(Project).filter(Project.id.in_(...))` in `asyncio.to_thread` with new `SessionLocal()` |
| `register_project` | Entire logic (check existing, update/create, commit) in `_run()` → `asyncio.to_thread` |
| `duplicate_project` | `ProjectService.create_project(db, project)` moved into `_run()` → `asyncio.to_thread` |
| `list_projects` | `get_projects_by_user_id(db, user_id)` in `_run()` → `asyncio.to_thread` |
| `update_project_status` | `ProjectService.update_project(db, ...)` in `_run()` → `asyncio.to_thread` |
| `get_project_from_db` | Extracted `_query_project_from_db` static method; runs in `_run()` → `asyncio.to_thread` |
| `get_global_project_from_db` | Query logic in `_run()` → `asyncio.to_thread` |
| `get_project_from_db_by_id` | Returns dict; query in `_run()` → `asyncio.to_thread` |
| `get_project_repo_details_from_db` | Query in `_run()` → `asyncio.to_thread` |
| `get_repo_and_branch_name` | `get_project_by_id(db, ...)` in `_run()` → `asyncio.to_thread` |
| `get_project_from_db_by_id_and_user_id` | Query in `_run()` → `asyncio.to_thread` |
| `delete_project` | Query + delete + commit in `_run()` → `asyncio.to_thread` |
| `get_demo_project_id` | Query in `_run()` → `asyncio.to_thread` |

**Imports added**: `asyncio`, `SessionLocal` (from `app.core.database`).

**Note**: `get_project_from_db` and `get_global_project_from_db` return **detached ORM objects** from closed sessions. Scalar attributes (`.id`, `.status`, `.repo_name`, etc.) work; lazy-loaded relationships would raise `DetachedInstanceError`. Consider returning dicts for robustness.

---

## 2. `app/modules/parsing/graph_construction/parsing_helper.py`

**Change**: `check_commit_status` was calling blocking GitHub API (`get_repo`, `get_branch`) directly. These are now run in a thread pool.

```python
# Before: blocking on event loop
_github, repo = self.github_service.get_repo(repo_name)
branch = repo.get_branch(branch_name)
latest_commit_id = branch.commit.sha

# After: offloaded to thread
def _fetch_latest_commit_sync() -> Optional[str]:
    _github, repo = self.github_service.get_repo(repo_name)
    branch = repo.get_branch(branch_name)
    return branch.commit.sha

latest_commit_id = await asyncio.to_thread(_fetch_latest_commit_sync)
```

**Import added**: `asyncio`.

---

## 3. `app/modules/code_provider/code_provider_controller.py`

**Changes**:

1. **`get_branch_list`**: GitHub branch fetch (including fallbacks) moved into sync helpers `_fetch_branches_from_provider_sync` and `_fetch_branches_with_fallbacks_sync`, executed via `asyncio.to_thread`. Cache check remains on main thread (fast Redis read).

2. **`check_public_repo`**: `provider.get_repository(repo_name)` moved into `_check_public_repo_sync` and run via `asyncio.to_thread`.

3. **`get_user_repos`**: Non-GitHub-App path (`provider.list_user_repositories()`) wrapped in `asyncio.to_thread`.

**Import added**: `asyncio`.

---

## 4. `app/modules/code_provider/github/github_service.py`

**Change**: `get_combined_user_repos` was doing sync `self.db.query(Project)...` in an async method. The DB query is now in `_query_project_list()` and executed via `asyncio.to_thread` with a dedicated `SessionLocal()`.

**Import added**: `SessionLocal` from `app.core.database`. (`asyncio` was already imported.)

---

## 5. `app/modules/parsing/graph_construction/parsing_controller.py`

**Change**: `fetch_parsing_status` was:
- Annotated with `db: AsyncSession` but receiving sync `Session` from the router
- Running blocking `db.execute(select(...))` on the event loop

The status query is now in `_query_status()` and run via `asyncio.to_thread` with a new `SessionLocal()`. Type hint updated to `db: Session`.

**Import**: `SessionLocal` imported inside the method (or at top if preferred).

---

## 6. `app/celery/tasks/parsing_tasks.py`

**Change**: `ParsingService` was created without `raise_library_exceptions=True`. On errors it raised `HTTPException`, which Celery cannot pickle, causing `UnpickleableExceptionWrapper`.

```python
# Before
parsing_service = ParsingService(self.db, user_id)

# After
parsing_service = ParsingService(
    self.db, user_id, raise_library_exceptions=True
)
```

With this flag, `ParsingService` raises `ParsingServiceError` instead of `HTTPException`, so Celery can handle failures correctly.

---

## 7. `app/modules/parsing/graph_construction/parsing_service.py`

**Change**: The `extracted_dir is None` branch always raised `HTTPException`. It now respects `_raise_library_exceptions` and raises `ParsingServiceError` when that flag is True (e.g. when called from Celery).

```python
if extracted_dir is None:
    if self._raise_library_exceptions:
        raise ParsingServiceError("Failed to set up project directory")
    raise HTTPException(...)
```

---

## Files Modified (Complete List)

| File | Purpose |
|------|---------|
| `app/modules/projects/projects_service.py` | Offload all sync DB calls to thread pool |
| `app/modules/parsing/graph_construction/parsing_helper.py` | Offload GitHub API in `check_commit_status` |
| `app/modules/code_provider/code_provider_controller.py` | Offload `get_branch_list`, `check_public_repo`, `get_user_repos` |
| `app/modules/code_provider/github/github_service.py` | Offload DB query in `get_combined_user_repos` |
| `app/modules/parsing/graph_construction/parsing_controller.py` | Offload DB query in `fetch_parsing_status` |
| `app/celery/tasks/parsing_tasks.py` | Use `raise_library_exceptions=True` for Celery |
| `app/modules/parsing/graph_construction/parsing_service.py` | Respect `raise_library_exceptions` for `extracted_dir is None` |

---

## Testing Notes

- **Local**: Low load; blocking was less visible. Changes should not break local behavior.
- **Production**: After deploy, parse requests should be processed under load; "Submitting parsing task" and Celery execution should appear in logs.
- **Celery failures**: Real exceptions (e.g. clone failure, GitHub API error) should now propagate instead of being wrapped in `HTTPException`. Check Celery worker logs for the actual traceback.

---

## Remaining Considerations

1. **Detached ORM objects**: `get_project_from_db` and `get_global_project_from_db` return detached objects. Safe for scalar access; consider returning dicts if relationship access is added.
2. **PostHogClient.send_event**: Sync call in parse flow; low impact but could be offloaded if needed.
3. **Redis cache reads**: `BranchCache.get_branches` is sync; kept on main thread as a fast operation. Monitor if it becomes a bottleneck.
4. **Conversation routing**: `redis_stream_generator`, `ensure_unique_run_id` use sync Redis; audited but not changed. Consider async Redis client if scaling further.
