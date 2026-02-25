"""
Test that wait_for_task_start runs off the event loop (thread), so concurrent
streaming requests are not serialized.

Before fix: sync wait_for_task_start blocked the event loop → N requests took ~ N * wait_secs.
After fix: wait_for_task_start runs in a thread → N requests take ~ wait_secs (concurrent).

Run with: pytest tests/integration-tests/conversations/test_concurrent_stream_timing.py -v

To see before/after timing comparison:
  pytest tests/integration-tests/conversations/test_concurrent_stream_timing.py -v -k "before_after" -s
"""
import asyncio
import time
from unittest.mock import MagicMock

import pytest

from app.modules.conversations.utils.redis_streaming import RedisStreamManager


# Simulated delay inside wait_for_task_start (seconds)
WAIT_SECS = 1.2
# Number of concurrent requests
CONCURRENT_REQUESTS = 3
# If requests were serialized we'd see ~ WAIT_SECS * CONCURRENT_REQUESTS.
# With thread offload we expect ~ WAIT_SECS + overhead (DB, controller, etc.).
# Assert wall time is below serial time (proves concurrency).
SERIAL_WALL_SECS = WAIT_SECS * CONCURRENT_REQUESTS
MAX_WALL_SECS = SERIAL_WALL_SECS * 0.85  # Must be clearly under serial

# Module where asyncio.to_thread is used (patch here to simulate "before" behavior)
CONVERSATION_ROUTING_MODULE = "app.modules.conversations.utils.conversation_routing"


@pytest.fixture
def slow_redis_stream_manager(monkeypatch):
    """RedisStreamManager mock whose wait_for_task_start sleeps WAIT_SECS then returns True."""
    mock_manager = MagicMock(spec=RedisStreamManager)
    mock_manager.redis_client = MagicMock()
    mock_manager.redis_client.exists.return_value = False
    mock_manager.stream_key.side_effect = (
        lambda conversation_id, run_id: f"stream:{conversation_id}:{run_id}"
    )
    # End stream immediately so response body consumption doesn't hang
    mock_manager.consume_stream.side_effect = lambda *a, **k: iter([{"type": "end"}])

    def slow_wait_for_task_start(*args, **kwargs):
        time.sleep(WAIT_SECS)
        return True

    mock_manager.wait_for_task_start.side_effect = slow_wait_for_task_start
    # Patch where RedisStreamManager is used (conversation_routing) so the mock is used
    monkeypatch.setattr(
        "app.modules.conversations.utils.conversation_routing.RedisStreamManager",
        lambda: mock_manager,
    )
    return mock_manager


@pytest.mark.asyncio
async def test_concurrent_stream_requests_not_serialized(
    client,
    mock_celery_tasks,
    slow_redis_stream_manager,
    setup_test_conversation_committed,
):
    """
    With wait_for_task_start offloaded to a thread, N concurrent streaming
    requests should complete in ~1x wait time, not Nx.
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
    blocking the event loop (simulated "before" fix), once with thread offload
    ("after" fix). Print both timings so you can see the difference.

    Run with: pytest ... -k before_vs_after -s
    """
    conversation_id = setup_test_conversation_committed.id
    url = f"/api/v1/conversations/{conversation_id}/message"
    form_data = {"content": "Timing before/after test."}

    # Simulate "before" fix: make to_thread run the callable on the event loop (blocking)
    async def _blocking_impl(f, *args, **kwargs):
        return f(*args, **kwargs)

    def blocking_to_thread(f, *args, **kwargs):
        return _blocking_impl(f, *args, **kwargs)

    # Save real to_thread before patching (same object is used by the module)
    real_to_thread = asyncio.to_thread

    # "Before": wait runs on event loop (blocking) → requests serialize
    monkeypatch.setattr(
        f"{CONVERSATION_ROUTING_MODULE}.asyncio.to_thread",
        blocking_to_thread,
    )
    wall_before, resp_before = await _run_concurrent_requests(
        client, url, form_data, CONCURRENT_REQUESTS
    )
    for r in resp_before:
        assert r.status_code == 200, getattr(r, "text", str(r))

    # "After": restore real to_thread so wait runs in thread pool → requests concurrent
    monkeypatch.setattr(
        f"{CONVERSATION_ROUTING_MODULE}.asyncio.to_thread",
        real_to_thread,
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
    print(f"  AFTER  (thread offload):    {wall_after:.2f}s wall")
    print(f"  Serial estimate (N × delay): ~{SERIAL_WALL_SECS:.1f}s")
    print(f"  Speedup: {wall_before / wall_after:.2f}x")
    print("--------------------------------------------------------\n")

    assert wall_after < wall_before, (
        f"After ({wall_after:.2f}s) should be faster than before ({wall_before:.2f}s)."
    )
