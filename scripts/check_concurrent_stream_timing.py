#!/usr/bin/env python3
"""
Timing comparison script for Redis sync-offload fixes (concurrent streaming).

Sends N concurrent POSTs to the message endpoint and reports:
- Time to first byte (TTFB) per request
- Total wall-clock time per round
- Whether concurrency is evident (wall time ~ 1x single-request time, not Nx)

Usage:
  # With a running server and existing conversation:
  CONVERSATION_ID=your-convo-id BASE_URL=http://localhost:8001 python scripts/check_concurrent_stream_timing.py

  # Optional: auth header
  AUTH_HEADER="Bearer <token>" CONVERSATION_ID=... python scripts/check_concurrent_stream_timing.py

  # Override concurrency, rounds, and timeout
  CONCURRENT=5 ROUNDS=2 TIMEOUT=60 python scripts/check_concurrent_stream_timing.py

Timing comparison (before vs after Redis fixes):
  - BEFORE (sync Redis on event loop): N concurrent requests block each other;
    wall time ≈ N × time_to_first_byte (e.g. N × 2–30s). Event loop starved.
  - AFTER (Redis offloaded to thread pool): N requests run concurrently;
    wall time ≈ time for one request to start (e.g. ~2–5s), not N times that.
  To compare: run this script against OLD server, note "Wall time"; run against
  NEW server. After fixes, wall time should be much lower (concurrent, not serial).

Proper testing options:
  1) Automated (no real server): pytest proves concurrency with a simulated delay.
     uv run pytest tests/integration-tests/conversations/test_concurrent_stream_timing.py -v -s
     Use -k before_vs_after -s to see BEFORE vs AFTER wall times in one run.
  2) Against real server: run this script with VALIDATE=1 to get exit 0 only if
     wall time suggests concurrency (wall < CONCURRENT * 15s).
     CONVERSATION_ID=... VALIDATE=1 uv run python scripts/check_concurrent_stream_timing.py
  3) Before/after deploy: run script against old server, note avg wall time; run
     against new server. After fixes, avg wall should drop (often 3–10x).
"""

import asyncio
import os
import sys
import time
from pathlib import Path

import httpx

# Allow importing app when run from repo root (used if script is extended to use app code)
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8001").rstrip("/")
CONVERSATION_ID = os.environ.get("CONVERSATION_ID", "")
CONCURRENT = int(os.environ.get("CONCURRENT", "3"))
ROUNDS = int(os.environ.get("ROUNDS", "2"))  # Number of rounds for timing comparison
TIMEOUT = float(os.environ.get("TIMEOUT", "90"))
# If set, exit 0 only when wall time suggests concurrency (use for CI or manual pass/fail)
VALIDATE = os.environ.get("VALIDATE", "").strip().lower() in ("1", "true", "yes")
# Wall time above this (seconds) per concurrent request suggests serialization
SERIAL_THRESHOLD_PER_REQUEST = 15


async def post_and_measure(
    client: httpx.AsyncClient,
    conversation_id: str,
    index: int,
    auth_header: str | None,
) -> tuple[int, float, float]:
    """POST one message (stream), record TTFB and total time. Returns (index, ttfb_secs, total_secs)."""
    url = f"{BASE_URL}/api/v1/conversations/{conversation_id}/message"
    data = {"content": f"Timing check message #{index}."}
    headers = {}
    if auth_header:
        headers["Authorization"] = auth_header

    ttfb_secs: float | None = None
    start = time.monotonic()
    try:
        async with client.stream(
            "POST",
            url,
            data=data,
            headers=headers or None,
            timeout=TIMEOUT,
        ) as response:
            if response.status_code != 200:
                return (index, -1.0, time.monotonic() - start)
            first_byte = True
            async for _ in response.aiter_bytes():
                if first_byte:
                    ttfb_secs = time.monotonic() - start
                    first_byte = False
    except Exception as e:
        print(f"  Request #{index} error: {e}", file=sys.stderr)
    total_secs = time.monotonic() - start
    return (index, ttfb_secs if ttfb_secs is not None else -1.0, total_secs)


async def run_one_round(
    client: httpx.AsyncClient,
    conversation_id: str,
    auth_header: str | None,
    round_num: int,
) -> tuple[float, int]:
    """Run one round of CONCURRENT requests. Returns (wall_secs, ok_count)."""
    wall_start = time.monotonic()
    tasks = [
        post_and_measure(client, conversation_id, i, auth_header)
        for i in range(CONCURRENT)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall_secs = time.monotonic() - wall_start
    ok = 0
    for r in results:
        if isinstance(r, Exception):
            print(f"  Round {round_num} request failed: {r}")
            continue
        idx, ttfb, total = r
        ok += 1
        ttfb_str = f"{ttfb:.2f}s" if ttfb >= 0 else "N/A"
        print(f"  Round {round_num} Request #{idx}: TTFB={ttfb_str}, total={total:.2f}s")
    return wall_secs, ok


async def main() -> None:
    if not CONVERSATION_ID:
        print(
            "Set CONVERSATION_ID (and optionally BASE_URL, AUTH_HEADER).",
            file=sys.stderr,
        )
        print(
            "  Example: CONVERSATION_ID=your-id BASE_URL=http://localhost:8001 python scripts/check_concurrent_stream_timing.py",
            file=sys.stderr,
        )
        sys.exit(1)

    auth_header = os.environ.get("AUTH_HEADER", "").strip() or None
    print(f"BASE_URL={BASE_URL}")
    print(f"CONVERSATION_ID={CONVERSATION_ID}")
    print(f"CONCURRENT={CONCURRENT}  ROUNDS={ROUNDS}")
    print()

    wall_times: list[float] = []
    async with httpx.AsyncClient() as client:
        for r in range(1, ROUNDS + 1):
            wall_secs, ok = await run_one_round(
                client, CONVERSATION_ID, auth_header, r
            )
            wall_times.append(wall_secs)
            print(f"  Round {r} wall time (all {CONCURRENT} requests): {wall_secs:.2f}s")
            if r < ROUNDS:
                print()

    # Summary for timing comparison
    print()
    print("--- Timing comparison ---")
    if len(wall_times) >= 1:
        avg_wall = sum(wall_times) / len(wall_times)
        print(f"  Wall time: min={min(wall_times):.2f}s  avg={avg_wall:.2f}s  max={max(wall_times):.2f}s")
    serial_estimate = CONCURRENT * 10
    if wall_times and wall_times[0] < serial_estimate * 0.6:
        print(
            f"  -> Concurrency looks good (wall ~1x single-request time, not {CONCURRENT}x)."
        )
    else:
        print(
            f"  -> If event loop were blocked, wall would be ~{CONCURRENT * 30}s. "
            "Compare this run (after fixes) with a run against the old server."
        )
    print()
    print("  Before vs after: run this script against OLD server, note wall time;")
    print("  then run against NEW server. After Redis offload, wall time should drop.")

    # Optional: pass/fail for CI or manual validation
    if VALIDATE:
        if not wall_times:
            print()
            print("  VALIDATE: FAIL (no successful rounds)")
            sys.exit(1)
        avg_wall = sum(wall_times) / len(wall_times)
        max_acceptable = CONCURRENT * SERIAL_THRESHOLD_PER_REQUEST
        if avg_wall <= max_acceptable:
            print()
            print(f"  VALIDATE: PASS (avg wall {avg_wall:.2f}s <= {max_acceptable}s)")
            sys.exit(0)
        print()
        print(
            f"  VALIDATE: FAIL (avg wall {avg_wall:.2f}s > {max_acceptable}s; "
            "suggests event loop blocking or very slow server)"
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
