#!/usr/bin/env python3
"""
Benchmark script for Step 2c (AsyncShareChatService, AsyncUserService) impact.

Measures latency for:
1. POST /conversations/share
2. GET /conversations/{id}/shared-emails
3. DELETE /conversations/{id}/access

Usage:
  BASE_URL=http://localhost:8001 AUTH_HEADER="Bearer <token>" CONVERSATION_ID=<uuid> \\
    python scripts/benchmark_share_access.py
  CONCURRENT=5 ROUNDS=2 BASE_URL=... AUTH_HEADER=... CONVERSATION_ID=... python scripts/benchmark_share_access.py

Output: p50, p95, p99 latency (ms); req/s per endpoint.
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
    method: str,
    url: str,
    headers: dict,
    count: int,
    **kwargs,
) -> tuple:
    latencies_ms = []
    start = time.monotonic()
    ok = 0

    async def one_request(_i: int) -> tuple:
        req_start = time.monotonic()
        try:
            if method == "POST":
                r = await client.post(url, headers=headers, timeout=TIMEOUT, **kwargs)
            elif method == "GET":
                r = await client.get(url, headers=headers, timeout=TIMEOUT)
            else:
                r = await client.request(method, url, headers=headers, timeout=TIMEOUT, **kwargs)
            success = r.status_code in (200, 201, 204)
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

    headers = {"Authorization": AUTH_HEADER, "Content-Type": "application/json"}

    print("--- Share/Access benchmark (Step 2c) ---")
    print(
        f"BASE_URL={BASE_URL} CONVERSATION_ID={CONVERSATION_ID} CONCURRENT={CONCURRENT} ROUNDS={ROUNDS}"
    )

    async with httpx.AsyncClient() as client:
        # 1. POST /conversations/share
        url = f"{BASE_URL}/conversations/share"
        all_lat = []
        total_wall = 0.0
        for r in range(ROUNDS):
            lat, wall, ok = await _run_concurrent(
                client, "POST", url, headers, CONCURRENT,
                json={"conversation_id": CONVERSATION_ID, "recipientEmails": [], "visibility": "PRIVATE"},
            )
            all_lat.extend(lat)
            total_wall += wall
            print(f"  POST /share round {r + 1}: wall={wall:.2f}s, ok={ok}/{CONCURRENT}")
        if all_lat:
            all_lat.sort()
            print(
                f"  POST /share latency (ms): p50={_percentile(all_lat, 50):.0f} "
                f"p95={_percentile(all_lat, 95):.0f} p99={_percentile(all_lat, 99):.0f}"
            )

        # 2. GET /conversations/{id}/shared-emails
        url = f"{BASE_URL}/conversations/{CONVERSATION_ID}/shared-emails"
        all_lat = []
        for r in range(ROUNDS):
            lat, wall, ok = await _run_concurrent(client, "GET", url, headers, CONCURRENT)
            all_lat.extend(lat)
            total_wall += wall
            print(f"  GET /shared-emails round {r + 1}: wall={wall:.2f}s, ok={ok}/{CONCURRENT}")
        if all_lat:
            all_lat.sort()
            print(
                f"  GET /shared-emails latency (ms): p50={_percentile(all_lat, 50):.0f} "
                f"p95={_percentile(all_lat, 95):.0f} p99={_percentile(all_lat, 99):.0f}"
            )

        # 3. DELETE /conversations/{id}/access (body: {"emails": []})
        url = f"{BASE_URL}/conversations/{CONVERSATION_ID}/access"
        all_lat = []
        for r in range(ROUNDS):
            lat, wall, ok = await _run_concurrent(
                client, "DELETE", url, headers, CONCURRENT,
                json={"emails": []},
            )
            all_lat.extend(lat)
            print(f"  DELETE /access round {r + 1}: wall={wall:.2f}s, ok={ok}/{CONCURRENT}")
        if all_lat:
            all_lat.sort()
            print(
                f"  DELETE /access latency (ms): p50={_percentile(all_lat, 50):.0f} "
                f"p95={_percentile(all_lat, 95):.0f} p99={_percentile(all_lat, 99):.0f}"
            )

    print("Done. Compare with baseline (before async share/access) for impact.")


if __name__ == "__main__":
    asyncio.run(main())
