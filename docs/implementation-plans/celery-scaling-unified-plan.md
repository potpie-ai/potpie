# Celery Scaling — Unified Plan (Potpie + Potpie-Workflows)

**Document Status:** Ready for implementation  
**Scope:** Both apps always run together — shared Redis, shared Postgres  
**Source:** Merged from `celery-scaling-plan.docx` + implementation plan + workflows integration

---

## Executive Summary

| Item | Current | Target |
|------|---------|--------|
| **Core problem** | `concurrency=1 --pool=solo` = 1 task at a time globally | 4 concurrent users per potpie pod |
| **DB connections** | Pool 10+10 × many processes → exceeds Postgres 100 | Env-driven pool (4+6) + formula for both apps |
| **Event queues** | Both apps use `external-event` → collision risk | `potpie-external-event` vs `workflows-external-event` |
| **Architecture** | Parsing (potpie) + codegen (workflows) already separate | Keep separate; fix event queue naming in Phase 5 |

---

## Architecture: Potpie + Workflows Together

### Queue Ownership (Already Separate)

| Workload | App | Queue | Worker Entrypoint |
|----------|-----|-------|-------------------|
| **Parsing** (clone, parse, KG) | Potpie | `{prefix}_process_repository` | potpie start.sh |
| **Agent chat / regenerate** | Potpie | `{prefix}_agent_tasks` | potpie start.sh |
| **Codegen** (spec, plan, impl, PR) | Potpie-workflows | `codegen` | workflows start_celery_worker.py |
| **Workflow execution** | Potpie-workflows | `workflow_execution` | workflows start_celery_worker.py |
| **Events** (webhooks, custom) | Both | `external-event` (shared name) | Both event workers |

### Interdependence

- **Parsing** (potpie) → creates project, KG in DB
- **Codegen** (workflows) → reads project/spec from DB, runs agents, writes code
- Data flows via DB/API, not Celery queues. Queues are already correctly separated by responsibility.

### Shared Resources

- **Redis:** Same instance, DB 0 (both apps)
- **Postgres:** Same instance (both apps)
- **Event queue name:** Both use `external-event` → rename in Phase 5 to avoid collision

---

## Phase 1: Celery Concurrency Fix — THIS WEEK (1–2 hours)

**Priority:** Critical — must complete before switching to prefork

### Problem Statement

- `--concurrency=1 --pool=solo` = exactly one task at a time
- Users 2–5 queue behind User 1 regardless of Redis or pods
- Pool 10+10 per process × total processes = exceeds Postgres `max_connections=100`

### 1.1 Reduce DB Pool Size — `app/core/database.py`

**Do this FIRST.** Both apps share Postgres. Total processes:

| App | Processes (1 pod each) |
|-----|-------------------------|
| Potpie | 1 Gunicorn + 4 Celery workers = 5 |
| Workflows | 1 FastAPI + Celery worker (workflow+codegen) + optional event bus = 2–4 |
| **Total** | **7–9+** |

**Sync engine (lines 13–21):**

```python
# Before
pool_size=10,
max_overflow=10,

# After
pool_size=int(os.getenv("DB_POOL_SIZE", "4")),
max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "6")),
```

**Async engine (lines 36–44):** Same change.

**Result:** Max 10 connections per process. 9 processes × 10 = 90, under 100.

### 1.2 Add worker_process_init — `app/celery/celery_app.py`

Add after `log_worker_memory_config` (around line 416):

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

### 1.3 Switch to Prefork — `start.sh` + `scripts/start.sh`

```bash
# Before
celery ... --concurrency=1 --pool=solo &

# After
celery -A app.celery.celery_app worker --loglevel=debug \
  -Q "${CELERY_QUEUE_NAME}_process_repository,${CELERY_QUEUE_NAME}_agent_tasks" \
  -E --concurrency=4 --pool=prefork &
```

### 1.4 Pre-Deploy Checklist

- [ ] Reduce pool_size=4, max_overflow=6 in database.py (both engines)
- [ ] Add worker_process_init to celery_app.py
- [ ] Change start.sh and scripts/start.sh to prefork + concurrency=4
- [ ] Run `ruff check` on modified files
- [ ] Test: agent chat (streaming + non-streaming), parsing, regenerate, codegen
- [ ] Monitor: `SELECT count(*) FROM pg_stat_activity;` — stay under 100
- [ ] Monitor: Celery worker logs — no asyncpg, LiteLLM, connection errors

### 1.5 Rollback

- Revert start.sh to `--concurrency=1 --pool=solo`
- Keep worker_process_init — safe and improves fork hygiene

---

## Phase 2: Sync/Async in API Path — WEEKS 1–2

FastAPI runs an async event loop. Sync blocking calls stall ALL requests on that pod.

| File | Issue | Fix |
|------|-------|-----|
| auth_service.py ~L20 | requests.post() in login() — no timeout | Add timeout=30; wrap in asyncio.to_thread |
| auth_router.py ~L59 | requests.post(SLACK_WEBHOOK_URL) — no timeout | Add timeout=30 |
| github_service.py ~L83 | requests.get() in get_installation_for_repo() | Add timeout=30; offload to thread |
| github_service.py L1126,1171 | Sync redis.get/setex in async path | asyncio.to_thread or async Redis |
| tunnel_router.py L50,72 | is_workspace_online() sync Redis | Offload to thread |
| tunnel_service.py | Sync Redis in tunnel ops | asyncio.to_thread |
| linear_client.py | requests.post — no timeout | Add timeout=30 |
| parse_webhook_helper.py | requests.post — no timeout | Add timeout=30 |

---

## Phase 3: DB Session Leaks + Unbounded Queries — WEEKS 2–3

At concurrency=4, session leaks compound. Fix immediately after Phase 1 is stable.

### Session Leaks (High Priority)

| File | Location | Fix |
|------|----------|-----|
| parsing_repomap.py | ~L56 | ParseHelper(next(get_db())) never closed → try/finally |
| search_semantic_tool.py | L76 | db = next(get_db()) never closed → try/finally |
| search_bash_tool.py | L87 | Same pattern → try/finally |
| change_detection_tool.py | L842 | ChangeDetectionTool(next(get_db())) → try/finally |
| code_changes_manager.py | L3971,4043,4135+ | Multiple db = next(get_db()) → wrap each in try/finally |

### Unbounded Queries (Medium Priority)

| File | Issue | Fix |
|------|-------|-----|
| chat_history_service.py L33-37 | get_session_history() loads ALL messages | Add .limit(100) or sliding window |
| github_service.py L937-949 | Duplicate-repo query, no limit | Add .limit() or pagination |
| usage_service.py L18-31 | Date range query unbounded | Cap max range (e.g. 1 year) |

---

## Phase 4: Sync Redis in Async Paths — WEEKS 3–4

| File | Issue | Priority | Fix |
|------|-------|----------|-----|
| conversation_routing.py L171 | Sync redis.xread(block=5000) + time.sleep in async generator | HIGH | Async Redis or run in thread |
| github_service.py L1126,1171 | Sync redis.get/setex in async path | HIGH | asyncio.to_thread or async Redis |
| tunnel_service.py L459-466 | KEYS pattern + multiple GETs | MED | Replace KEYS with SCAN; add TTL |
| tunnel_service.py L211-212 | Tunnel record keys no TTL | MED | Add TTL (e.g. 24h) |

---

## Phase 5: Queue Splitting + Event Queue Naming — WHEN LOAD DEMANDS

### 5.1 Split Potpie Worker Deployments

Parsing (heavy, slow) and agent (latency-sensitive) share the same worker today. Under load, parsing floods starve agent tasks.

| Worker Type | Queue | Concurrency | Scaling Priority |
|-------------|-------|-------------|------------------|
| Agent / Chat | `{prefix}_agent_tasks` | 4–8 | Scale first — user-facing |
| Parsing | `{prefix}_process_repository` | 2–4 | Scale second — CPU/memory heavy |

**Deployment:**
- Agent worker: `-Q ${prefix}_agent_tasks --concurrency=4`
- Parsing worker: `-Q ${prefix}_process_repository --concurrency=2`

### 5.2 Rename Event Queues (Both Apps)

**Current:** Both use `external-event` → collision when both run workers.

**Target:**

| App | Current | New Queue Name |
|-----|---------|----------------|
| Potpie | `external-event` | `potpie-external-event` |
| Potpie-workflows | `external-event` | `workflows-external-event` |

**Potpie changes:**
- `app/celery/celery_app.py` task_routes: `"queue": "potpie-external-event"`
- `app/modules/event_bus/celery_bus.py` send_task: `queue="potpie-external-event"`
- `app/modules/event_bus/tasks/event_tasks.py` decorator: `queue="potpie-external-event"`
- `scripts/start_event_worker.sh`: `-Q potpie-external-event`

**Workflows changes:**
- `src/celery_config.py` task_routes: `"queue": "workflows-external-event"`
- `src/adapters/event_bus_subscriber_single_queue.py`: `queue="workflows-external-event"`, `--queues=workflows-external-event`

---

## Phase 6: Chat History, Streaming & Memory — AS NEEDED

| Area | File | Issue | Fix |
|------|------|-------|-----|
| Chat History | chat_history_service.py | Loads all messages into memory | Limit to last 50–100 (sliding window) |
| Streaming | conversation_routing.py | Sync Redis stream consumer blocks | Async Redis or thread-based consumer |
| Tarball Memory | parsing_helper.py | await resp.read() loads full tarball into RAM | Stream to temp file (aiofiles + chunked read) |

---

## DB Connection Scaling Formula (Both Apps)

```
total_processes = potpie_processes + workflows_processes
safe_pool_per_process = (max_connections - 10 reserve) / total_processes
```

**With max_connections=100:**

| Scenario | Potpie | Workflows | Total | Safe Pool/Process | Suggested |
|----------|--------|-----------|-------|-------------------|-----------|
| 1 pod each | 5 | 3 | 8 | 90/8 ≈ 11 | pool_size=4, max_overflow=6 (max 10) ✅ |
| 2 pods each | 10 | 6 | 16 | 90/16 ≈ 5 | pool_size=2, max_overflow=3 (max 5) ⚠️ |
| 3 pods each | 15 | 9 | 24 | 90/24 ≈ 3 | Raise max_connections to 300 |

**When scaling to 2+ pods each, add to compose.yaml:**

```yaml
postgres:
  command: ["postgres", "-c", "max_connections=300"]
```

**Env vars (.env):**

```env
DB_POOL_SIZE=4
DB_MAX_OVERFLOW=6
```

---

## Full Timeline Summary

| Phase | Focus | Timeline | Key Files |
|-------|-------|----------|-----------|
| **1 — CRITICAL** | Prefork + pool fix | This week (1–2 hrs) | database.py, celery_app.py, start.sh |
| **2** | Sync I/O in API path | Week 1–2 | auth, github, tunnel services |
| **3** | DB session leaks + query limits | Week 2–3 | tools, repomap, history service |
| **4** | Sync Redis in async paths | Week 3–4 | conversation_routing, github, tunnel |
| **5** | Queue splitting + event queue rename | When load demands | celery_app, celery_bus, event_tasks, workflows celery_config |
| **6** | History, streaming, memory | As needed | chat_history, conversation_routing, parsing_helper |

---

## Success Criteria

- [ ] Phase 1: 4 users can run agent tasks simultaneously without queueing
- [ ] Phase 1: pg_stat_activity count stays under 100
- [ ] Phase 1: No asyncpg, LiteLLM, or DB connection errors in worker logs
- [ ] Phase 2: No sync blocking in FastAPI paths under concurrent load
- [ ] Phase 3: No DB connection exhaustion during agent tasks with tools
- [ ] Phase 4: Redis calls non-blocking in all async routes
- [ ] Phase 5: Parsing flood does not delay agent/chat; event queues isolated
- [ ] Phase 6: Memory stable under long chat histories and large repo parsing

---

## Files Modified Summary

### Phase 1
| File | Changes |
|------|---------|
| app/core/database.py | pool_size, max_overflow env-driven (4, 6) |
| app/celery/celery_app.py | worker_process_init: engine.dispose(), configure_litellm_for_celery() |
| start.sh | --concurrency=4 --pool=prefork |
| scripts/start.sh | --concurrency=4 --pool=prefork |

### Phase 5 (Event Queue Rename)
| File | Changes |
|------|---------|
| app/celery/celery_app.py | task_routes: potpie-external-event |
| app/modules/event_bus/celery_bus.py | queue="potpie-external-event" |
| app/modules/event_bus/tasks/event_tasks.py | queue="potpie-external-event" |
| scripts/start_event_worker.sh | -Q potpie-external-event |
| potpie-workflows: src/celery_config.py | task_routes: workflows-external-event |
| potpie-workflows: event_bus_subscriber_single_queue.py | queue="workflows-external-event" |

---

## Related Documents

- [celery-prefork-and-db-pool-plan.md](./celery-prefork-and-db-pool-plan.md) — Phase 1 implementation details
- [celery-scaling-plan.docx](../celery-scaling-plan.docx) — Original Word plan (6 phases: prefork, sync/async, session leaks, Redis, queue splitting, history/streaming)
