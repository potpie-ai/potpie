#!/usr/bin/env python3
"""
Benchmark script for Step 3 (Async Redis for tunnel + GitHub/branch cache).

Measures latency and throughput for endpoints that now use native async Redis:
  1. N concurrent GET /tunnel/workspace/{workspace_id}
  2. N concurrent GET /tunnel/workspace/{workspace_id}/socket-status
  3. N concurrent GET /github/user-repos (triggers branch cache / repo listing)

Usage:
  AUTH_HEADER="Bearer <token>" BASE_URL=http://localhost:8001 python scripts/benchmark_tunnel_github.py
  WORKSPACE_ID=<16-char-hex> AUTH_HEADER="Bearer <token>" CONCURRENT=10 ROUNDS=2 python scripts/benchmark_tunnel_github.py

Output: p50, p95, p99 latency (ms); requests/sec; total wall time.
Use before/after deploying to compare impact of native async Redis (tunnel + cache).
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
WORKSPACE_ID = os.environ.get("WORKSPACE_ID", "0" * 16)
AUTH_HEADER = os.environ.get("AUTH_HEADER", "").strip() or None
CONCURRENT = int(os.environ.get("CONCURRENT", "5"))
ROUNDS = int(os.environ.get("ROUNDS", "2"))
TIMEOUT = float(os.environ.get("TIMEOUT", "30"))


def _percentile(sorted_latencies_ms: list[float], p: float) -> float:
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
    method: str,
    url: str,
    headers: dict,
    count: int,
) -> tuple[list[float], int, float]:
    """Run count concurrent requests. Returns (latencies_ms, ok_count, wall_secs)."""
    latencies_ms: list[float] = []
    start = time.monotonic()

    async def one_request(_i: int) -> float | None:
        req_start = time.monotonic()
        try:
            r = await client.get(url, headers=headers, timeout=TIMEOUT)
            if r.is_success:
                return (time.monotonic() - req_start) * 1000
        except Exception:
            pass
        return None

    tasks = [one_request(i) for i in range(count)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall_secs = time.monotonic() - start
    for r in results:
        if isinstance(r, (int, float)):
            latencies_ms.append(float(r))
        elif isinstance(r, Exception):
            pass
    ok = sum(1 for r in results if isinstance(r, (int, float)))
    return latencies_ms, ok, wall_secs


async def main() -> None:
    if not AUTH_HEADER:
        print("AUTH_HEADER is required (e.g. AUTH_HEADER='Bearer <firebase_token>')")
        sys.exit(1)

    headers = {"Authorization": AUTH_HEADER}
    print("--- Step 3 benchmark (tunnel + GitHub cache) ---")
    print(f"BASE_URL={BASE_URL} WORKSPACE_ID={WORKSPACE_ID} CONCURRENT={CONCURRENT} ROUNDS={ROUNDS}\n")

    async with httpx.AsyncClient() as client:
        # 1. GET /tunnel/workspace/{workspace_id}
        url1 = f"{BASE_URL}/tunnel/workspace/{WORKSPACE_ID}"
        all_lat1: list[float] = []
        for rnd in range(ROUNDS):
            lat, ok, wall = await _run_concurrent(
                client, "GET", url1, headers, CONCURRENT
            )
            all_lat1.extend(lat)
            print(f"GET /tunnel/workspace/{{id}} round {rnd + 1}: wall={wall:.2f}s, ok={ok}/{CONCURRENT}")
        all_lat1.sort()
        if all_lat1:
            print(
                f"  GET /tunnel/workspace latency (ms): p50={int(_percentile(all_lat1, 50))} "
                f"p95={int(_percentile(all_lat1, 95))} p99={int(_percentile(all_lat1, 99))}"
            )
            print(
                f"  Effective req/s (approx): {len(all_lat1) / sum(all_lat1) * 1000:.1f}"
            )
        print()

        # 2. GET /tunnel/workspace/{workspace_id}/socket-status
        url2 = f"{BASE_URL}/tunnel/workspace/{WORKSPACE_ID}/socket-status"
        all_lat2: list[float] = []
        for rnd in range(ROUNDS):
            lat, ok, wall = await _run_concurrent(
                client, "GET", url2, headers, CONCURRENT
            )
            all_lat2.extend(lat)
            print(f"GET /tunnel/workspace/{{id}}/socket-status round {rnd + 1}: wall={wall:.2f}s, ok={ok}/{CONCURRENT}")
        all_lat2.sort()
        if all_lat2:
            print(
                f"  GET /tunnel/workspace/{{id}}/socket-status latency (ms): p50={int(_percentile(all_lat2, 50))} "
                f"p95={int(_percentile(all_lat2, 95))} p99={int(_percentile(all_lat2, 99))}"
            )
        print()

        # 3. GET /github/user-repos
        url3 = f"{BASE_URL}/github/user-repos"
        all_lat3: list[float] = []
        for rnd in range(ROUNDS):
            lat, ok, wall = await _run_concurrent(
                client, "GET", url3, headers, CONCURRENT
            )
            all_lat3.extend(lat)
            print(f"GET /github/user-repos round {rnd + 1}: wall={wall:.2f}s, ok={ok}/{CONCURRENT}")
        all_lat3.sort()
        if all_lat3:
            print(
                f"  GET /github/user-repos latency (ms): p50={int(_percentile(all_lat3, 50))} "
                f"p95={int(_percentile(all_lat3, 95))} p99={int(_percentile(all_lat3, 99))}"
            )
            print(
                f"  Effective req/s (approx): {len(all_lat3) / sum(all_lat3) * 1000:.1f}"
            )

    print("\nDone. Compare with baseline (before async Redis tunnel/cache) for impact.")


if __name__ == "__main__":
    asyncio.run(main())
