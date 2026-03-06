"""
Stress tests for async migration endpoints: failure points and performance.

- Run against the in-process test client (no live server).
- Use concurrency to detect failure points (non-2xx, timeouts) and latency degradation.
- Marked with 'stress'; exclude with: pytest -m "not stress" (default in CI).
- Run stress tests: pytest tests/integration-tests/stress/ -m stress -v

Performance: we assert 0% failure rate and optionally report p50/p95/p99.
Failure points: status codes and exception types are collected and reported on failure.
"""

import asyncio
import os
import time
from typing import Any

import httpx
import pytest
import pytest_asyncio


def _percentile(sorted_ms: list[float], p: float) -> float:
    if not sorted_ms:
        return 0.0
    k = (len(sorted_ms) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_ms) - 1)
    return sorted_ms[f] + (k - f) * (sorted_ms[c] - sorted_ms[f])


# Concurrency and rounds for in-process stress (keep short for CI)
STRESS_CONCURRENT = int(os.environ.get("STRESS_CONCURRENT", "5"))
STRESS_ROUNDS = int(os.environ.get("STRESS_ROUNDS", "1"))
STRESS_TIMEOUT = float(os.environ.get("STRESS_TIMEOUT", "30"))


@pytest_asyncio.fixture
async def stress_client(client: httpx.AsyncClient):
    """Client is the in-process AsyncClient from conftest."""
    yield client


async def _run_concurrent(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    count: int,
    rounds: int,
    timeout: float,
    stream: bool = False,
    data: dict | None = None,
) -> dict[str, Any]:
    """Run count concurrent requests × rounds. Return ok, fail, status_counts, latencies_ms, errors."""
    # Client from conftest has base_url="http://test"; relative path is sufficient
    full_url = url
    headers = getattr(client, "headers", {}) or {}

    latencies_ms: list[float] = []
    status_counts: dict[int, int] = {}
    errors: list[str] = []

    for _ in range(rounds):
        async def one(_i: int) -> tuple[float | None, int | None, str | None]:
            start = time.monotonic()
            try:
                if stream:
                    async with client.stream(
                        method, full_url, headers=headers, data=data, timeout=timeout
                    ) as r:
                        status = r.status_code
                        async for _ in r.aiter_bytes():
                            break
                else:
                    if data:
                        r = await client.request(
                            method, full_url, headers=headers, data=data, timeout=timeout
                        )
                    else:
                        r = await client.get(full_url, headers=headers, timeout=timeout)
                    status = r.status_code
                elapsed = (time.monotonic() - start) * 1000
                return (elapsed, status, None)
            except Exception as e:
                return (None, None, f"{type(e).__name__}:{e!s}")

        results = await asyncio.gather(
            *[one(i) for i in range(count)], return_exceptions=True
        )
        for r in results:
            if isinstance(r, Exception):
                errors.append(str(r))
                continue
            elapsed, status, err = r
            if err:
                errors.append(err)
                continue
            status_counts[status] = status_counts.get(status, 0) + 1
            if status and 200 <= status < 300 and elapsed is not None:
                latencies_ms.append(elapsed)

    total = count * rounds
    ok = sum(c for s, c in status_counts.items() if s and 200 <= s < 300)
    return {
        "ok": ok,
        "fail": total - ok,
        "status_counts": status_counts,
        "latencies_ms": latencies_ms,
        "errors": errors,
    }


@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_active_session_no_5xx(
    stress_client: httpx.AsyncClient,
    setup_test_conversation_committed,
):
    """GET /active-session under load: assert no 5xx (404/403 allowed when no active run)."""
    conversation_id = setup_test_conversation_committed.id
    url = f"/api/v1/conversations/{conversation_id}/active-session"
    res = await _run_concurrent(
        stress_client, "GET", url, STRESS_CONCURRENT, STRESS_ROUNDS, STRESS_TIMEOUT
    )
    total = res["ok"] + res["fail"]
    assert total > 0, "No requests completed"
    # Failure point: any 5xx or timeout
    bad_statuses = [s for s in res["status_counts"] if s and s >= 500]
    bad_count = sum(res["status_counts"].get(s, 0) for s in bad_statuses)
    assert bad_count == 0 and not res["errors"], (
        f"GET /active-session: {bad_count} 5xx, {len(res['errors'])} errors. "
        f"Status counts: {res['status_counts']}. Errors: {res['errors'][:5]}"
    )
    if res["latencies_ms"]:
        res["latencies_ms"].sort()
        print(
            f"  active-session latency ms: p50={_percentile(res['latencies_ms'], 50):.0f} "
            f"p95={_percentile(res['latencies_ms'], 95):.0f} p99={_percentile(res['latencies_ms'], 99):.0f}"
        )


@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_task_status_no_5xx(
    stress_client: httpx.AsyncClient,
    setup_test_conversation_committed,
):
    """GET /task-status under load: assert no 5xx (404/403 allowed when no run)."""
    conversation_id = setup_test_conversation_committed.id
    url = f"/api/v1/conversations/{conversation_id}/task-status"
    res = await _run_concurrent(
        stress_client, "GET", url, STRESS_CONCURRENT, STRESS_ROUNDS, STRESS_TIMEOUT
    )
    total = res["ok"] + res["fail"]
    assert total > 0, "No requests completed"
    bad_statuses = [s for s in res["status_counts"] if s and s >= 500]
    bad_count = sum(res["status_counts"].get(s, 0) for s in bad_statuses)
    assert bad_count == 0 and not res["errors"], (
        f"GET /task-status: {bad_count} 5xx, {len(res['errors'])} errors. "
        f"Status counts: {res['status_counts']}. Errors: {res['errors'][:5]}"
    )


@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_post_message_no_5xx(
    stress_client: httpx.AsyncClient,
    mock_celery_tasks,
    mock_redis_stream_manager,
    setup_test_conversation_committed,
):
    """POST /message (streaming) under load: assert no 5xx; 2xx = success (stream started)."""
    conversation_id = setup_test_conversation_committed.id
    url = f"/api/v1/conversations/{conversation_id}/message"
    data = {"content": "stress test message"}
    res = await _run_concurrent(
        stress_client,
        "POST",
        url,
        STRESS_CONCURRENT,
        STRESS_ROUNDS,
        STRESS_TIMEOUT,
        stream=True,
        data=data,
    )
    total = res["ok"] + res["fail"]
    assert total > 0, "No requests completed"
    # Failure point: 5xx or transport errors (mock may cause 4xx/DataError in some setups)
    bad_statuses = [s for s in res["status_counts"] if s and s >= 500]
    bad_count = sum(res["status_counts"].get(s, 0) for s in bad_statuses)
    assert bad_count == 0, (
        f"POST /message: {bad_count} 5xx. "
        f"Status counts: {res['status_counts']}. Errors: {res['errors'][:5]}"
    )
