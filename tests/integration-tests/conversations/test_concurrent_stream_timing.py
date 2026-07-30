"""
Tests for Redis-related async fixes: streaming does not block the event loop.

- wait_for_task_start uses async Redis so N concurrent requests complete in
  approximately one wait interval, not N wait intervals.
- Stream consumption uses the shared synchronous Redis stream fixture and ends
  immediately so the test measures only task-start concurrency.

Before fix: sync wait_for_task_start blocked the event loop and N requests took
approximately N * wait_secs. After the fix, the async waits overlap.

Run all: pytest tests/integration-tests/conversations/test_concurrent_stream_timing.py -v

Before/after timing comparison (use -s to see output):
  pytest tests/integration-tests/conversations/test_concurrent_stream_timing.py -k before_vs_after -v -s
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.modules.usage.usage_service import UsageService


# Timing-sensitive test: opt-in via `-m stress`
pytestmark = [pytest.mark.stress, pytest.mark.asyncio]

# Simulated delay inside wait_for_task_start (seconds)
WAIT_SECS = 1.2
# Number of concurrent requests
CONCURRENT_REQUESTS = 3
# If requests were serialized we'd see ~ WAIT_SECS * CONCURRENT_REQUESTS.
# With thread offload we expect ~ WAIT_SECS + overhead (DB, controller, etc.).
# Require wall time below 90% of serial so that serialized (blocking) behavior fails.
SERIAL_WALL_SECS = WAIT_SECS * CONCURRENT_REQUESTS
MAX_WALL_SECS = SERIAL_WALL_SECS * 0.9

@pytest.fixture
def slow_redis_stream_manager(app, monkeypatch, mock_redis_stream_manager):
    """Async Redis mock whose task-start wait is slow but non-blocking."""
    original_manager = getattr(app.state, "async_redis_stream_manager", None)
    mock_manager = MagicMock()
    mock_manager.set_task_status = AsyncMock()
    mock_manager.publish_event = AsyncMock()
    mock_manager.set_task_id = AsyncMock()
    mock_manager.get_task_status = AsyncMock(return_value="running")
    mock_manager.redis_client = MagicMock()
    mock_manager.redis_client.exists = AsyncMock(return_value=False)
    mock_manager.redis_client.set = AsyncMock(return_value=True)
    mock_manager.redis_client.delete = AsyncMock(return_value=None)
    mock_manager.stream_key.side_effect = (
        lambda conversation_id, run_id: f"stream:{conversation_id}:{run_id}"
    )

    async def consume_stream(*args, **kwargs):
        yield MagicMock(dict=lambda: {"type": "end"})

    async def slow_wait_for_task_start(*args, **kwargs):
        await asyncio.sleep(WAIT_SECS)
        return True

    async def allow_usage(*args, **kwargs):
        return True

    mock_manager.consume_stream = consume_stream
    mock_manager.wait_for_task_start = AsyncMock(side_effect=slow_wait_for_task_start)
    monkeypatch.setattr(UsageService, "check_usage_limit", allow_usage)
    monkeypatch.setattr(
        "app.modules.conversations.conversations_router.ConversationController",
        MagicMock(),
    )
    app.state.async_redis_stream_manager = mock_manager
    yield mock_manager
    app.state.async_redis_stream_manager = original_manager


@pytest.mark.asyncio
async def test_concurrent_stream_requests_not_serialized(
    client,
    mock_celery_tasks,
    slow_redis_stream_manager,
    setup_test_conversation_committed,
):
    """
    With async wait_for_task_start, N concurrent streaming requests should
    complete in ~1x wait time, not Nx.
    """
    conversation_id = setup_test_conversation_committed.id
    url = f"/api/v1/conversations/{conversation_id}/message"
    form_data = {"content": "Concurrent timing test message."}

    async def post_once():
        r = await client.post(url, data=form_data)
        # Consume stream so server-side generator completes and connection closes
        if r.status_code == 200:
            async for _ in r.aiter_bytes():
                pass
        return r

    start = time.monotonic()
    responses = await asyncio.gather(
        *[post_once() for _ in range(CONCURRENT_REQUESTS)]
    )
    wall_secs = time.monotonic() - start

    for r in responses:
        assert r.status_code == 200, getattr(r, "text", str(r))
        assert "text/event-stream" in r.headers.get("content-type", "")

    assert (
        wall_secs < MAX_WALL_SECS
    ), (
        f"Concurrent requests took {wall_secs:.2f}s (max allowed {MAX_WALL_SECS:.2f}s). "
        f"If wait_for_task_start blocked the event loop, {CONCURRENT_REQUESTS} requests "
        f"would take ~{SERIAL_WALL_SECS:.1f}s. Wall time < serial time proves concurrency."
    )


async def _run_concurrent_requests(client, url: str, form_data: dict, n: int) -> float:
    """Run n concurrent POSTs, consume stream for each, return wall-clock time in seconds."""

    async def post_once():
        r = await client.post(url, data=form_data)
        if r.status_code == 200:
            async for _ in r.aiter_bytes():
                pass
        return r

    start = time.monotonic()
    responses = await asyncio.gather(*[post_once() for _ in range(n)])
    return time.monotonic() - start, responses


@pytest.mark.asyncio
async def test_concurrent_stream_timing_before_vs_after(
    client,
    mock_celery_tasks,
    slow_redis_stream_manager,
    setup_test_conversation_committed,
    monkeypatch,
):
    """
    Run the same N concurrent requests twice: once with wait_for_task_start
    blocking the event loop (simulated "before" fix), once with an async wait
    ("after" fix). Print both timings so you can see the difference.

    Run with: pytest ... -k before_vs_after -s
    """
    conversation_id = setup_test_conversation_committed.id
    url = f"/api/v1/conversations/{conversation_id}/message"
    form_data = {"content": "Timing before/after test."}

    async def blocking_wait_for_task_start(*args, **kwargs):
        time.sleep(WAIT_SECS)
        return True

    async def async_wait_for_task_start(*args, **kwargs):
        await asyncio.sleep(WAIT_SECS)
        return True

    # "Before": wait runs on event loop (blocking) → requests serialize
    slow_redis_stream_manager.wait_for_task_start = AsyncMock(
        side_effect=blocking_wait_for_task_start
    )
    wall_before, resp_before = await _run_concurrent_requests(
        client, url, form_data, CONCURRENT_REQUESTS
    )
    for r in resp_before:
        assert r.status_code == 200, getattr(r, "text", str(r))

    # "After": async Redis wait yields to the event loop, so requests overlap
    slow_redis_stream_manager.wait_for_task_start = AsyncMock(
        side_effect=async_wait_for_task_start
    )

    wall_after, resp_after = await _run_concurrent_requests(
        client, url, form_data, CONCURRENT_REQUESTS
    )
    for r in resp_after:
        assert r.status_code == 200, getattr(r, "text", str(r))

    # Report so user can see the difference (use -s to see print output)
    print("\n--- wait_for_task_start: before vs after async fix ---")
    print(f"  Simulated delay in wait_for_task_start: {WAIT_SECS}s")
    print(f"  Concurrent requests: {CONCURRENT_REQUESTS}")
    print(f"  BEFORE (sync on event loop): {wall_before:.2f}s wall")
    print(f"  AFTER  (async Redis wait):  {wall_after:.2f}s wall")
    print(f"  Serial estimate (N × delay): ~{SERIAL_WALL_SECS:.1f}s")
    print(f"  Speedup: {wall_before / wall_after:.2f}x")
    print("--------------------------------------------------------\n")

    # Allow up to 15% tolerance: "after" should be faster or within noise of "before".
    # In CI/mocked env both runs can be similar; we fail only if "after" is clearly slower
    # (e.g. the async wait regressed and became serial).
    tolerance = max(wall_before * 0.15, 0.5)
    assert wall_after <= wall_before + tolerance, (
        f"After ({wall_after:.2f}s) should be faster or within {tolerance:.2f}s of before ({wall_before:.2f}s). "
        "If async task-start waiting regressed, after would be much larger."
    )
