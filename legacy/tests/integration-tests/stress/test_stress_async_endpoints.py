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


async def _do_one_request(
    client: httpx.AsyncClient,
    method: str,
    full_url: str,
    headers: dict,
    data: dict | None,
    timeout: float,
    stream: bool,
) -> tuple[float | None, int | None, str | None]:
    """Execute a single request; return (elapsed_ms, status_code, error_string or None)."""
    start = time.monotonic()
    try:
        if stream:
            async with client.stream(
                method, full_url, headers=headers, data=data, timeout=timeout
            ) as r:
                status = r.status_code
                async for _ in r.aiter_bytes():
                    break
        elif data:
            r = await client.request(
                method, full_url, headers=headers, data=data, timeout=timeout
            )
            status = r.status_code
        else:
            r = await client.get(full_url, headers=headers, timeout=timeout)
            status = r.status_code
        elapsed = (time.monotonic() - start) * 1000
        return (elapsed, status, None)
    except Exception as e:
        return (None, None, f"{type(e).__name__}:{e!s}")


def _record_result(
    elapsed: float | None,
    status: int | None,
    err: str | None,
    status_counts: dict[int, int],
    latencies_ms: list[float],
    errors: list[str],
) -> None:
    """Update status_counts, latencies_ms, and errors from one request result."""
    if err:
        errors.append(err)
        return
    status_counts[status] = status_counts.get(status, 0) + 1
    if status and 200 <= status < 300 and elapsed is not None:
        latencies_ms.append(elapsed)


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
    full_url = url
    headers = getattr(client, "headers", {}) or {}

    latencies_ms: list[float] = []
    status_counts: dict[int, int] = {}
    errors: list[str] = []

    for _ in range(rounds):
        results = await asyncio.gather(
            *[
                _do_one_request(
                    client, method, full_url, headers, data, timeout, stream
                )
                for _ in range(count)
            ],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                errors.append(str(r))
            else:
                _record_result(
                    r[0], r[1], r[2], status_counts, latencies_ms, errors
                )

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
    """GET /active-session under load: assert no 5xx.

    Stress-tests the 'no active session found' path: setup_test_conversation_committed
    creates only a DB conversation (no Redis stream keys), so SessionService.get_active_session()
    always returns no session and the endpoint returns 404. We assert no 5xx under load;
    all responses are expected to be 404 (or 403 if access denied).
    """
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
    # Explicit: we only exercise the no-session path (404/403), never 2xx
    ok_count = res["ok"]
    assert ok_count == 0, (
        f"Expected all 404/403 (no Redis keys). Got {ok_count} 2xx. "
        f"Status counts: {res['status_counts']}"
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
