# Potpie context-graph host (`app/modules/context_graph`)

**Where to edit**

| Goal | Location |
|------|----------|
| Domain logic, use cases, HTTP/MCP/CLI routes, Postgres/Neo4j adapters, job queue port | [`potpie/context-engine`](../../../../potpie/context-engine) |
| Potpie-only glue (Celery tasks, DB-scoped container build, auth-mounted FastAPI router, intelligence bundle helpers) | **this directory** |

Do **not** add portable business logic here—it belongs in the `context-engine` package so it stays testable and reusable.

**Tests**

All context-graph / context-engine unit tests run from the package:

```bash
cd potpie/context-engine
uv run pytest tests/unit/
```

There is **no** separate `tests/unit/context_graph/` tree in the repo root; that duplicated domain tests and has been removed.

**Files here**

| Module | Purpose |
|--------|---------|
| `wiring.py` | `PotpieContextEngineSettings`, `SqlalchemyPotResolution`, `UserScopedContextGraphPotResolution` (``context_graph_pots`` + members + repositories only), `build_container_for_session` / `build_container_for_user_session` |
| `context_graph_pot_model.py` | User-owned context pots (context scope independent of parsing projects) |
| `context_pot_routes.py` | `GET`/`POST` `/api/v2/context/pots` (API key); mounted from [`app/api/router.py`](../../api/router.py) |
| `celery_job_queue.py` | Celery implementation of `ContextGraphJobQueuePort` |
| `tasks.py` | Celery task wrappers calling `application.use_cases.context_graph_jobs` |
| `sync_enqueue.py` | Backfill enqueue helper for HTTP (`enqueue_backfill_with_container`) |
| `context_engine_http.py` | Mounts `create_context_router` at **`/api/v1/context`** (Firebase auth) + shared **`POTPIE_CONTEXT_GRAPH_MUTATIONS`** |
| (see [`app/api/router.py`](../../api/router.py)) | Same router also mounted at **`/api/v2/context`** with **`X-API-Key`** (`get_api_key_user`) for CLI / MCP |
| `bundle_renderer.py` | Prefetch / coverage snippets for intelligence agents |
| `code_provider_source_control.py` | Code-provider → `SourceControlPort` bridge |
| `models.py` | SQLAlchemy models re-exported from `app.core.models` |

**Env**

- `CONTEXT_GRAPH_JOB_QUEUE_BACKEND` — `celery` (default), `hatchet`, or `noop` (see `bootstrap.queue_factory` in context-engine).
- `CONTEXT_GRAPH_CELERY_QUEUE_MODULE` — override for the Celery adapter import path (default: `app.modules.context_graph.celery_job_queue`).
- **Hatchet (optional):** set `CONTEXT_GRAPH_JOB_QUEUE_BACKEND=hatchet` + **`HATCHET_CLIENT_*`** to publish jobs via Hatchet `event.push`. The in-repo worker entrypoint was removed with the server deprecation (#851/#858); consuming those events requires running your own Hatchet worker.
