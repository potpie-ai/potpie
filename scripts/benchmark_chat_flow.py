#!/usr/bin/env python3
"""
Benchmark script for Step 2b (AsyncChatHistoryService) impact.

Measures latency for:
1. POST /conversations/{id}/message (streaming) — end-to-end including DB writes
   via AsyncChatHistoryService (add_message_chunk, flush_message_buffer, etc.)
2. POST /stop — stop_generation with save_partial_ai_message (async when available)

Usage:
  BASE_URL=http://localhost:8001 AUTH_HEADER="Bearer <token>" CONVERSATION_ID=<uuid> \\
    python scripts/benchmark_chat_flow.py
  CONCURRENT=5 ROUNDS=2 BASE_URL=... AUTH_HEADER=... CONVERSATION_ID=... python scripts/benchmark_chat_flow.py

Output: p50, p95, p99 latency (ms); requests/sec; total wall time per endpoint.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

import httpx

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8001").rstrip("/")
AUTH_HEADER = os.environ.get("AUTH_HEADER", "").strip() or None
CONVERSATION_ID = os.environ.get("CONVERSATION_ID", "").strip() or None
CONCURRENT = int(os.environ.get("CONCURRENT", "3"))
ROUNDS = int(os.environ.get("ROUNDS", "2"))
TIMEOUT = float(os.environ.get("TIMEOUT", "60"))


def _percentile(sorted_latencies_ms: list, p: float) -> float:
    if not sorted_latencies_ms:
        return 0.0
    k = (len(sorted_latencies_ms) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_latencies_ms) - 1)
    return sorted_latencies_ms[f] + (k - f) * (
        sorted_latencies_ms[c] - sorted_latencies_ms[f]
    )


async def _run_concurrent_post_message(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    count: int,
) -> tuple:
    """Run count concurrent POST /message (streaming). Returns (latencies_ms, wall_secs, ok_count)."""
    latencies_ms = []
    start = time.monotonic()
    ok = 0

    async def one_request(_i: int) -> tuple:
        req_start = time.monotonic()
        try:
            async with client.stream(
                "POST",
                url,
                headers=headers,
                data={"content": "benchmark ping"},
                timeout=TIMEOUT,
            ) as r:
                if r.status_code not in (200, 201):
                    return ((time.monotonic() - req_start) * 1000, 0)
                async for _ in r.aiter_bytes():
                    pass
            success = True
        except Exception:
            success = False
        elapsed = (time.monotonic() - req_start) * 1000
        return (elapsed, 1 if success else 0)

    tasks = [one_request(i) for i in range(count)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall_secs = time.monotonic() - start

    for r in results:
        if isinstance(r, Exception):
            continue
        lat, n = r
        latencies_ms.append(lat)
        ok += n

    latencies_ms.sort()
    return latencies_ms, wall_secs, ok


async def _run_concurrent_stop(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    count: int,
) -> tuple:
    """Run count concurrent POST /stop. Returns (latencies_ms, wall_secs, ok_count)."""
    latencies_ms = []
    start = time.monotonic()
    ok = 0

    async def one_request(_i: int) -> tuple:
        req_start = time.monotonic()
        try:
            r = await client.post(url, headers=headers, timeout=TIMEOUT)
            success = r.status_code in (200, 204)
        except Exception:
            success = False
        elapsed = (time.monotonic() - req_start) * 1000
        return (elapsed, 1 if success else 0)

    tasks = [one_request(i) for i in range(count)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall_secs = time.monotonic() - start

    for r in results:
        if isinstance(r, Exception):
            continue
        lat, n = r
        latencies_ms.append(lat)
        ok += n

    latencies_ms.sort()
    return latencies_ms, wall_secs, ok


async def main():
    if not AUTH_HEADER:
        print("Set AUTH_HEADER (e.g. Bearer <token>) to run benchmark.")
        return
    if not CONVERSATION_ID:
        print("Set CONVERSATION_ID to run benchmark.")
        return

    headers = {"Authorization": AUTH_HEADER}

    print("--- Chat flow benchmark (Step 2b: AsyncChatHistoryService) ---")
    print(
        f"BASE_URL={BASE_URL} CONVERSATION_ID={CONVERSATION_ID} CONCURRENT={CONCURRENT} ROUNDS={ROUNDS}"
    )

    async with httpx.AsyncClient() as client:
        # 1. POST /message (streaming)
        url = f"{BASE_URL}/conversations/{CONVERSATION_ID}/message"
        all_latencies = []
        total_ok = 0
        for round_no in range(ROUNDS):
            latencies, wall, ok = await _run_concurrent_post_message(
                client, url, headers, CONCURRENT
            )
            all_latencies.extend(latencies)
            total_ok += ok
            print(f"POST /message round {round_no + 1}: wall={wall:.2f}s, ok={ok}/{CONCURRENT}")
        all_latencies.sort()
        if all_latencies:
            print(
                f"  POST /message latency (ms): p50={_percentile(all_latencies, 50):.0f} "
                f"p95={_percentile(all_latencies, 95):.0f} p99={_percentile(all_latencies, 99):.0f}"
            )
        else:
            print("  No successful POST /message requests.")

        # 2. POST /stop (requires an active run; may 404 if none)
        stop_url = f"{BASE_URL}/conversations/{CONVERSATION_ID}/stop"
        all_stop_latencies = []
        stop_ok = 0
        for round_no in range(ROUNDS):
            latencies, wall, ok = await _run_concurrent_stop(
                client, stop_url, headers, CONCURRENT
            )
            all_stop_latencies.extend(latencies)
            stop_ok += ok
            print(f"POST /stop round {round_no + 1}: wall={wall:.2f}s, ok={ok}/{CONCURRENT}")
        all_stop_latencies.sort()
        if all_stop_latencies:
            print(
                f"  POST /stop latency (ms): p50={_percentile(all_stop_latencies, 50):.0f} "
                f"p95={_percentile(all_stop_latencies, 95):.0f} p99={_percentile(all_stop_latencies, 99):.0f}"
            )
        else:
            print("  No successful POST /stop requests (normal if no active stream).")

    print("Done. Compare with baseline (before async chat history) for impact.")


if __name__ == "__main__":
    asyncio.run(main())
