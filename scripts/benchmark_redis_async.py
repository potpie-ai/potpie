#!/usr/bin/env python3
"""
Benchmark script for Step 1 (AsyncRedisStreamManager) impact.

Measures latency and throughput for endpoints that now use native async Redis:
  1. N concurrent POST /message (streaming) — time to first byte and total
  2. N concurrent GET /active-session
  3. N concurrent GET /task-status

Usage:
  CONVERSATION_ID=<id> BASE_URL=http://localhost:8001 python scripts/benchmark_redis_async.py
  AUTH_HEADER="Bearer <token>" CONVERSATION_ID=... CONCURRENT=10 ROUNDS=3 python scripts/benchmark_redis_async.py

Output: p50, p95, p99 latency (ms); requests/sec; total wall time.
Use before/after deploying to compare impact of native async Redis.
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
CONVERSATION_ID = os.environ.get("CONVERSATION_ID", "")
AUTH_HEADER = os.environ.get("AUTH_HEADER", "").strip() or None
CONCURRENT = int(os.environ.get("CONCURRENT", "5"))
ROUNDS = int(os.environ.get("ROUNDS", "2"))
TIMEOUT = float(os.environ.get("TIMEOUT", "90"))


def _percentile(sorted_latencies_ms: list[float], p: float) -> float:
    if not sorted_latencies_ms:
        return 0.0
    k = (len(sorted_latencies_ms) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_latencies_ms) - 1)
    return sorted_latencies_ms[f] + (k - f) * (sorted_latencies_ms[c] - sorted_latencies_ms[f])


async def _run_concurrent(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict,
    count: int,
    stream: bool = False,
    data: dict | None = None,
) -> tuple[list[float], float]:
    """Run `count` concurrent requests. Returns (list of latencies_ms, wall_secs)."""
    latencies_ms: list[float] = []
    start = time.monotonic()

    async def one_request(i: int) -> float:
        req_start = time.monotonic()
        if stream:
            # Form body for POST /message (v1 uses Form)
            async with client.stream(
                method, url, headers=headers, data=data, timeout=TIMEOUT
            ) as r:
                async for _ in r.aiter_bytes():
                    break
        else:
            if data:
                await client.request(
                    method, url, headers=headers, data=data, timeout=TIMEOUT
                )
            else:
                await client.get(url, headers=headers, timeout=TIMEOUT)
        return (time.monotonic() - req_start) * 1000

    tasks = [one_request(i) for i in range(count)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall_secs = time.monotonic() - start
    for r in results:
        if isinstance(r, Exception):
            latencies_ms.append(-1.0)
        else:
            latencies_ms.append(r)
    return latencies_ms, wall_secs


async def main() -> None:
    if not CONVERSATION_ID:
        print("Set CONVERSATION_ID to an existing conversation id.")
        sys.exit(1)

    headers = {}
    if AUTH_HEADER:
        headers["Authorization"] = AUTH_HEADER

    async with httpx.AsyncClient() as client:
        print("--- Async Redis benchmark (conversation endpoints) ---")
        print(f"BASE_URL={BASE_URL} CONVERSATION_ID={CONVERSATION_ID} CONCURRENT={CONCURRENT} ROUNDS={ROUNDS}\n")

        # 1. POST /message (streaming) — time to first byte (v1 uses Form)
        url_post = f"{BASE_URL}/api/v1/conversations/{CONVERSATION_ID}/message"
        data_post = {"content": "Benchmark message"}  # Form data
        all_latencies: list[float] = []
        for r in range(ROUNDS):
            latencies, wall = await _run_concurrent(
                client, "POST", url_post, headers, CONCURRENT,
                stream=True, data=data_post,
            )
            valid = [x for x in latencies if x >= 0]
            all_latencies.extend(valid)
            print(f"POST /message round {r+1}: wall={wall:.2f}s, ok={len(valid)}/{CONCURRENT}")

        if all_latencies:
            all_latencies.sort()
            print(f"  POST /message latency (ms): p50={_percentile(all_latencies, 50):.0f} "
                  f"p95={_percentile(all_latencies, 95):.0f} p99={_percentile(all_latencies, 99):.0f}")
            n = len(all_latencies)
            total_wall = ROUNDS * (sum(all_latencies) / n) / 1000 if n else 0
            print(f"  Effective req/s (approx): {n / (total_wall or 1):.1f}\n")

        # 2. GET /active-session
        url_active = f"{BASE_URL}/api/v1/conversations/{CONVERSATION_ID}/active-session"
        all_latencies = []
        for r in range(ROUNDS):
            latencies, wall = await _run_concurrent(
                client, "GET", url_active, headers, CONCURRENT,
            )
            valid = [x for x in latencies if x >= 0]
            all_latencies.extend(valid)
            print(f"GET /active-session round {r+1}: wall={wall:.2f}s, ok={len(valid)}/{CONCURRENT}")

        if all_latencies:
            all_latencies.sort()
            print(f"  GET /active-session latency (ms): p50={_percentile(all_latencies, 50):.0f} "
                  f"p95={_percentile(all_latencies, 95):.0f} p99={_percentile(all_latencies, 99):.0f}\n")

        # 3. GET /task-status
        url_task = f"{BASE_URL}/api/v1/conversations/{CONVERSATION_ID}/task-status"
        all_latencies = []
        for r in range(ROUNDS):
            latencies, wall = await _run_concurrent(
                client, "GET", url_task, headers, CONCURRENT,
            )
            valid = [x for x in latencies if x >= 0]
            all_latencies.extend(valid)
            print(f"GET /task-status round {r+1}: wall={wall:.2f}s, ok={len(valid)}/{CONCURRENT}")

        if all_latencies:
            all_latencies.sort()
            print(f"  GET /task-status latency (ms): p50={_percentile(all_latencies, 50):.0f} "
                  f"p95={_percentile(all_latencies, 95):.0f} p99={_percentile(all_latencies, 99):.0f}")

    print("\nDone. Compare with baseline (before async Redis) for impact.")


if __name__ == "__main__":
    asyncio.run(main())
