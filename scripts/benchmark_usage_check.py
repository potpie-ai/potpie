#!/usr/bin/env python3
"""
Benchmark script for Step 2a (AsyncSession UsageService) impact.

Measures latency for POST /conversations, which triggers check_usage_limit
(now using AsyncSession for get_usage_data).

Usage:
  BASE_URL=http://localhost:8001 AUTH_HEADER="Bearer <token>" python scripts/benchmark_usage_check.py
  CONCURRENT=10 ROUNDS=3 BASE_URL=... AUTH_HEADER=... python scripts/benchmark_usage_check.py

Output: p50, p95, p99 latency (ms); requests/sec; total wall time.
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
CONCURRENT = int(os.environ.get("CONCURRENT", "5"))
ROUNDS = int(os.environ.get("ROUNDS", "2"))
TIMEOUT = float(os.environ.get("TIMEOUT", "30"))


def _percentile(sorted_latencies_ms: list, p: float) -> float:
    if not sorted_latencies_ms:
        return 0.0
    k = (len(sorted_latencies_ms) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_latencies_ms) - 1)
    return sorted_latencies_ms[f] + (k - f) * (
        sorted_latencies_ms[c] - sorted_latencies_ms[f]
    )


async def _run_concurrent(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    count: int,
) -> tuple:
    """Run count concurrent POST /conversations. Returns (latencies_ms, wall_secs, ok_count)."""
    latencies_ms = []
    start = time.monotonic()
    ok = 0

    async def one_request(_i: int) -> tuple:
        req_start = time.monotonic()
        try:
            r = await client.post(
                url,
                headers=headers,
                json={
                    "title": "Benchmark",
                    "project_ids": [],
                    "agent_ids": [],
                },
                timeout=TIMEOUT,
            )
            success = r.status_code in (200, 201)
        except Exception:
            success = False
        elapsed = (time.monotonic() - req_start) * 1000
        return (elapsed, 1 if success else 0)

    tasks = [one_request(i) for i in range(count)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall_secs = time.monotonic() - start

    for r in results:
        if isinstance(r, Exception):
            latencies_ms.append(0.0)
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

    url = f"{BASE_URL}/conversations"
    headers = {"Authorization": AUTH_HEADER, "Content-Type": "application/json"}

    print("--- Usage check benchmark (POST /conversations) ---")
    print(f"BASE_URL={BASE_URL} CONCURRENT={CONCURRENT} ROUNDS={ROUNDS}")

    all_latencies = []
    total_wall = 0.0
    async with httpx.AsyncClient() as client:
        for r in range(ROUNDS):
            latencies, wall, ok = await _run_concurrent(
                client, url, headers, CONCURRENT
            )
            all_latencies.extend(latencies)
            total_wall += wall
            print(f"  POST /conversations round {r + 1}: wall={wall:.2f}s, ok={ok}/{CONCURRENT}")

    all_latencies.sort()
    n = len(all_latencies)
    if n == 0:
        print("No successful requests.")
        return

    p50 = _percentile(all_latencies, 50)
    p95 = _percentile(all_latencies, 95)
    p99 = _percentile(all_latencies, 99)
    req_per_sec = n / total_wall if total_wall > 0 else 0

    print(f"  POST /conversations latency (ms): p50={p50:.0f} p95={p95:.0f} p99={p99:.0f}")
    print(f"  Effective req/s (approx): {req_per_sec:.1f}")
    print("Done. Compare with baseline (before async UsageService) for impact.")


if __name__ == "__main__":
    asyncio.run(main())
