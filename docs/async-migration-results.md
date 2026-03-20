# Async migration — benchmark results

Record before/after numbers for each step. Run the same commands and env (CONCURRENT, ROUNDS, BASE_URL) when comparing.

---

## Step 1: AsyncRedisStreamManager (native async Redis)

**Script:** `CONVERSATION_ID=<id> BASE_URL=http://localhost:8001 uv run python scripts/benchmark_redis_async.py`  
**Default:** CONCURRENT=5, ROUNDS=2

### Before (sync Redis on event loop)

| Endpoint            | p50 (ms) | p95 (ms) | p99 (ms) | Wall (round 1/2) | ok     |
|---------------------|----------|----------|----------|-------------------|--------|
| POST /message       | 14–15    | 24–31    | 25–35    | ~0.03s / ~0.01s   | 5/5    |
| GET /active-session | 5–6      | 7        | 7        | ~0.01s            | 5/5    |
| GET /task-status   | 4        | 5        | 5        | ~0.01s            | 5/5    |

*Effective req/s (POST): ~306–358.*

### After (native async Redis)

*Run the same benchmark on the branch with `AsyncRedisStreamManager` + `AsyncSessionService` and paste results here.*

Under higher concurrency (e.g. CONCURRENT=20, ROUNDS=3) or when Redis is slower, the “after” run should show better concurrency (wall time stays closer to 1× single-request time instead of stacking).

---

## Step 2a: Async UsageService

*When implemented: record before/after for `scripts/benchmark_usage_check.py` (or equivalent).*

---

## Later steps

*Add sections as each step is measured.*
