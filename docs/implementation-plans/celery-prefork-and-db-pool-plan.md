# Celery Prefork & DB Pool — Final Implementation Plan

**Status:** Ready for implementation  
**Priority:** Critical — must complete before switching to prefork  
**Estimated time:** 1–2 hours

---

## Executive Summary

Switching from `--pool=solo --concurrency=1` to `--pool=prefork --concurrency=4` will allow 4 concurrent users per pod instead of 1. However, the current DB pool (10+10 per process) would allow up to ~240 connections with 5 processes, exceeding Postgres default `max_connections=100`. This plan fixes the pool first, adds fork-safe initialization, then switches to prefork.

---

## Findings Summary

| Finding | Impact |
|---------|--------|
| Pool 10+10 × 5 processes = 100+ connections | Exceeds Postgres default 100 → guaranteed outage under load |
| Inherited DB connections after fork | Stale connections in child processes |
| LiteLLM configured only at import (pre-fork) | May cause issues in forked children |
| Concurrency=1, pool=solo | Only 1 task at a time globally |

---

## Implementation Order

1. **database.py** — Env-driven pool + conservative defaults
2. **celery_app.py** — worker_process_init: dispose engine + LiteLLM
3. **start.sh** + **scripts/start.sh** — Prefork + concurrency=4
4. **compose.yaml** (optional) — Raise max_connections for future scaling

---

## Code Changes

### 1. `app/core/database.py`

**Change:** Make pool size env-driven with conservative defaults.

**Replace lines 13–21 (sync engine):**

```python
# Before
engine = create_engine(
    os.getenv("POSTGRES_SERVER"),
    pool_size=10,  # Initial number of connections in the pool
    max_overflow=10,  # Maximum number of connections beyond pool_size
    pool_timeout=30,  # Timeout in seconds for getting a connection from the pool
    pool_recycle=1800,  # Recycle connections every 30 minutes (to avoid stale connections)
    pool_pre_ping=True,  # Check the connection is alive before using it
    echo=False,  # Set to True for SQL query logging, False in production
)

# After
engine = create_engine(
    os.getenv("POSTGRES_SERVER"),
    pool_size=int(os.getenv("DB_POOL_SIZE", "4")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "6")),
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
    echo=False,
)
```

**Replace lines 36–44 (async engine):**

```python
# Before
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_size=10,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=False,  # Disabled: causes event loop issues in Celery workers
    echo=False,
)

# After
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_size=int(os.getenv("DB_POOL_SIZE", "4")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "6")),
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=False,  # Disabled: causes event loop issues in Celery workers
    echo=False,
)
```

**Result:** Max 10 connections per process. 5 processes × 10 = 50, safely under 100.

---

### 2. `app/celery/celery_app.py`

**Change:** Add `worker_process_init` handler to dispose DB pool and re-run LiteLLM config in each forked child.

**Add new handler after `log_worker_memory_config` (around line 416), before `@worker_process_shutdown.connect`:**

```python
@worker_process_init.connect
def reset_db_pool_and_litellm(**kwargs):
    """
    Run in each Celery worker child after fork.
    - Dispose inherited sync DB connections (stale after fork)
    - Re-run LiteLLM config for fork safety
    """
    try:
        from app.core.database import engine

        engine.dispose()
        logger.debug("Disposed sync engine pool in worker child")
    except Exception as e:
        logger.warning(f"Failed to dispose sync engine in worker child: {e}")

    try:
        configure_litellm_for_celery()
        logger.debug("Re-ran LiteLLM config in worker child")
    except Exception as e:
        logger.warning(f"Failed to configure LiteLLM in worker child: {e}")
```

---

### 3. `start.sh`

**Change:** Switch Celery worker to prefork with concurrency=4.

**Replace line 91:**

```bash
# Before
celery -A app.celery.celery_app worker --loglevel=debug -Q "${CELERY_QUEUE_NAME}_process_repository,${CELERY_QUEUE_NAME}_agent_tasks" -E --concurrency=1 --pool=solo &

# After
celery -A app.celery.celery_app worker --loglevel=debug -Q "${CELERY_QUEUE_NAME}_process_repository,${CELERY_QUEUE_NAME}_agent_tasks" -E --concurrency=4 --pool=prefork &
```

---

### 4. `scripts/start.sh`

**Change:** Same as start.sh.

**Replace line 87:**

```bash
# Before
celery -A app.celery.celery_app worker --loglevel=debug -Q "${CELERY_QUEUE_NAME}_process_repository,${CELERY_QUEUE_NAME}_agent_tasks" -E --concurrency=1 --pool=solo &

# After
celery -A app.celery.celery_app worker --loglevel=debug -Q "${CELERY_QUEUE_NAME}_process_repository,${CELERY_QUEUE_NAME}_agent_tasks" -E --concurrency=4 --pool=prefork &
```

---

### 5. (Optional) `compose.yaml` — Raise max_connections for future scaling

**Add to postgres service (for when you scale to 2+ pods):**

```yaml
postgres:
  image: postgres:latest
  container_name: potpie_postgres
  command: ["postgres", "-c", "max_connections=300"]
  environment:
    POSTGRES_USER: postgres
    # ... rest unchanged
```

**When to add:** Before scaling to 2+ pods, or proactively if you expect to scale soon.

---

## Environment Variables

Add to `.env` (optional; defaults are conservative):

```env
# DB pool (defaults: 4, 6 → max 10 per process)
# DB_POOL_SIZE=4
# DB_MAX_OVERFLOW=6

# When scaling to 2 pods (10 processes): use pool_size=3, max_overflow=5
# DB_POOL_SIZE=3
# DB_MAX_OVERFLOW=5
```

---

## Scaling Formula

```
Safe pool per process = (max_connections - 10 reserve) / total_processes

1 pod  (5 processes):  90/5  = 18 → pool_size=4, max_overflow=6  (max 10) ✅
2 pods (10 processes): 90/10 = 9  → pool_size=3, max_overflow=5  (max 8)  ✅
3 pods (15 processes): 90/15 = 6  → pool_size=2, max_overflow=4  (max 6)  ⚠️ tight
```

At 3 pods, raise `max_connections` to 300.

---

## Pre-Deploy Checklist

- [ ] Apply database.py changes (env-driven pool)
- [ ] Apply celery_app.py changes (worker_process_init)
- [ ] Apply start.sh and scripts/start.sh (prefork + concurrency=4)
- [ ] (Optional) Add max_connections to compose.yaml
- [ ] Run `ruff check` on modified files
- [ ] Test locally: agent chat (streaming + non-streaming), parsing, regenerate
- [ ] Monitor Postgres connection count after deploy

---

## Verification

1. **Connection count:** After deploy, run `SELECT count(*) FROM pg_stat_activity;` — should stay under 100 (or 300 if raised).
2. **Concurrent users:** 4 users can run agent tasks simultaneously without queueing.
3. **Logs:** No asyncpg, LiteLLM, or connection exhaustion errors in Celery worker logs.

---

## Rollback

If issues occur:

1. Revert start.sh / scripts/start.sh to `--concurrency=1 --pool=solo`
2. Optionally revert database.py pool to 10+10 (only if not hitting connection limits)
3. Keep worker_process_init (dispose + LiteLLM) — it's safe and improves fork hygiene

---

## Future Phases (Reference)

| Phase | Focus |
|-------|-------|
| **Phase 2** | Sync/async in API path (auth, GitHub, tunnel) |
| **Phase 3** | DB session leaks (tools, RepoMap) |
| **Phase 4** | Sync Redis in async paths |
| **Phase 5** | Queue splitting (agent vs parsing workers) |
| **Phase 6** | Chat history limit, streaming, tarball memory |

---

## Files Modified Summary

| File | Changes |
|------|---------|
| `app/core/database.py` | pool_size, max_overflow env-driven (4, 6) |
| `app/celery/celery_app.py` | worker_process_init: engine.dispose(), configure_litellm_for_celery() |
| `start.sh` | --concurrency=4 --pool=prefork |
| `scripts/start.sh` | --concurrency=4 --pool=prefork |
| `compose.yaml` | (optional) max_connections=300 |
