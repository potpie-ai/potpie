#!/usr/bin/env python3
"""
Stress test harness for async migration endpoints.

Runs multiple concurrency levels (ramp), records failures and latency,
and reports failure points (status codes, timeouts) plus performance (p50/p95/p99, req/s).

Usage:
  # Conversation endpoints (needs CONVERSATION_ID and AUTH_HEADER)
  CONVERSATION_ID=<id> AUTH_HEADER="Bearer <token>" BASE_URL=http://localhost:8001 \\
    python scripts/stress_harness.py --profile conversation

  # Tunnel + GitHub (needs AUTH_HEADER, optional WORKSPACE_ID)
  AUTH_HEADER="Bearer <token>" python scripts/stress_harness.py --profile tunnel_github

  # Custom concurrency levels and rounds (default: 4 to match Celery workers)
  CONCURRENCY=4,8 ROUNDS=3 python scripts/stress_harness.py --profile conversation

  # Output JSON for baseline comparison
  python scripts/stress_harness.py --profile conversation --output results.json

Exit code: 0 if all levels had 0% failure rate; 1 if any failures (so CI can fail on regression).
"""

import argparse
import asyncio
import json
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
CONVERSATION_ID = os.environ.get("CONVERSATION_ID", "")
WORKSPACE_ID = os.environ.get("WORKSPACE_ID", "0" * 16)
CONCURRENCY_LEVELS = [
    int(x.strip()) for x in os.environ.get("CONCURRENCY", "1").split(",")
]
ROUNDS = int(os.environ.get("ROUNDS", "1"))
# Default 120s: POST /message (stream) needs time for Celery to pick up and send first byte
TIMEOUT = float(os.environ.get("TIMEOUT", "120"))
WARMUP_REQUESTS = int(os.environ.get("WARMUP", "3"))


def _percentile(sorted_latencies_ms: list[float], p: float) -> float:
    if not sorted_latencies_ms:
        return 0.0
    k = (len(sorted_latencies_ms) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_latencies_ms) - 1)
    return sorted_latencies_ms[f] + (k - f) * (
        sorted_latencies_ms[c] - sorted_latencies_ms[f]
    )


async def _run_stress(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict,
    count: int,
    rounds: int,
    timeout: float,
    stream: bool = False,
    data: dict | None = None,
) -> dict:
    """
    Run `count` concurrent requests × `rounds`. Returns dict with:
    ok, fail, timeouts, status_counts, latencies_ms, wall_secs, errors (list of strings).
    """
    latencies_ms: list[float] = []
    status_counts: dict[int, int] = {}
    timeouts = 0
    errors: list[str] = []
    total_wall_secs = 0.0

    for _ in range(rounds):
        start = time.monotonic()

        async def one_request(_i: int) -> tuple[float | None, int | None, str | None]:
            req_start = time.monotonic()
            try:
                if stream:
                    async with client.stream(
                        method, url, headers=headers, data=data, timeout=timeout
                    ) as r:
                        status = r.status_code
                        async for _ in r.aiter_bytes():
                            break
                else:
                    if data:
                        r = await client.request(
                            method, url, headers=headers, data=data, timeout=timeout
                        )
                    else:
                        r = await client.get(url, headers=headers, timeout=timeout)
                    status = r.status_code
                elapsed = (time.monotonic() - req_start) * 1000
                return (elapsed, status, None)
            except httpx.TimeoutException as e:
                return (None, None, f"timeout:{e!s}")
            except Exception as e:
                return (None, None, f"error:{type(e).__name__}:{e!s}")

        tasks = [one_request(i) for i in range(count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_secs = time.monotonic() - start

        for r in results:
            if isinstance(r, Exception):
                timeouts += 1
                errors.append(str(r))
                continue
            elapsed, status, err = r
            if err:
                if "timeout" in err.lower():
                    timeouts += 1
                errors.append(err)
                continue
            status_counts[status] = status_counts.get(status, 0) + 1
            if status and 200 <= status < 300 and elapsed is not None:
                latencies_ms.append(elapsed)

        total_wall_secs += wall_secs

    total_requests = count * rounds
    ok_count = sum(c for s, c in status_counts.items() if s and 200 <= s < 300)
    return {
        "ok": ok_count,
        "fail": total_requests - ok_count,
        "timeouts": timeouts,
        "status_counts": status_counts,
        "latencies_ms": latencies_ms,
        "wall_secs": total_wall_secs,
        "total_requests": total_requests,
        "errors": errors[:20],
    }


async def _warmup(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict,
    n: int,
    timeout: float,
    data: dict | None = None,
) -> None:
    for _ in range(n):
        try:
            if data:
                await client.request(method, url, headers=headers, data=data, timeout=timeout)
            else:
                await client.get(url, headers=headers, timeout=timeout)
        except Exception:
            pass


def _profile_conversation() -> list[tuple[str, str, str, dict]]:
    if not CONVERSATION_ID:
        print("CONVERSATION_ID is required for profile=conversation")
        sys.exit(1)
    base = f"{BASE_URL}/api/v1/conversations/{CONVERSATION_ID}"
    return [
        ("POST /message (stream)", "POST", f"{base}/message", {"stream": True, "data": {"content": "stress"}}),
        ("GET /active-session", "GET", f"{base}/active-session", {}),
        ("GET /task-status", "GET", f"{base}/task-status", {}),
    ]


def _profile_tunnel_github() -> list[tuple[str, str, str, dict]]:
    return [
        ("GET /tunnel/workspace/{id}", "GET", f"{BASE_URL}/tunnel/workspace/{WORKSPACE_ID}", {}),
        ("GET /tunnel/workspace/{id}/socket-status", "GET", f"{BASE_URL}/tunnel/workspace/{WORKSPACE_ID}/socket-status", {}),
        ("GET /github/user-repos", "GET", f"{BASE_URL}/github/user-repos", {}),
    ]


async def run_stress_for_profile(profile: str, output_path: str | None) -> int:
    if not AUTH_HEADER and profile != "tunnel_github":
        print("AUTH_HEADER is required for most profiles (e.g. AUTH_HEADER='Bearer <token>')")
        return 1
    if profile == "tunnel_github" and not AUTH_HEADER:
        print("AUTH_HEADER is required for tunnel_github")
        return 1

    if profile == "conversation":
        endpoints = _profile_conversation()
    elif profile == "tunnel_github":
        endpoints = _profile_tunnel_github()
    else:
        print(f"Unknown profile: {profile}")
        return 1

    headers = {"Authorization": AUTH_HEADER} if AUTH_HEADER else {}
    all_results: dict[str, list[dict]] = {}
    exit_code = 0

    async with httpx.AsyncClient() as client:
        for name, method, url, opts in endpoints:
            stream = opts.get("stream", False)
            data = opts.get("data")
            print(f"\n--- {name} ---")
            all_results[name] = []

            for concurrency in CONCURRENCY_LEVELS:
                # Warm up
                await _warmup(
                    client, method, url, headers, WARMUP_REQUESTS, TIMEOUT, data
                )
                # Stress run
                res = await _run_stress(
                    client,
                    method,
                    url,
                    headers,
                    concurrency,
                    ROUNDS,
                    TIMEOUT,
                    stream=stream,
                    data=data,
                )
                all_results[name].append(
                    {"concurrency": concurrency, **res}
                )

                total = res["ok"] + res["fail"]
                fail_pct = (100.0 * res["fail"] / total) if total else 0
                if fail_pct > 0:
                    exit_code = 1

                print(f"  concurrency={concurrency}: ok={res['ok']} fail={res['fail']} timeouts={res['timeouts']} ({fail_pct:.1f}% fail)")
                if res["status_counts"]:
                    codes = ", ".join(f"{s}:{c}" for s, c in sorted(res["status_counts"].items()))
                    print(f"    status codes: {codes}")
                if res["errors"]:
                    for e in res["errors"][:5]:
                        print(f"    error: {e[:80]}")
                if res["latencies_ms"]:
                    res["latencies_ms"].sort()
                    print(
                        f"    latency (ms): p50={_percentile(res['latencies_ms'], 50):.0f} "
                        f"p95={_percentile(res['latencies_ms'], 95):.0f} "
                        f"p99={_percentile(res['latencies_ms'], 99):.0f}"
                    )
                    n = len(res["latencies_ms"])
                    total_sec = sum(res["latencies_ms"]) / 1000
                    if total_sec > 0:
                        print(f"    req/s (approx): {n / total_sec:.1f}")

    if output_path:
        out = {
            "profile": profile,
            "concurrency_levels": CONCURRENCY_LEVELS,
            "rounds": ROUNDS,
            "results": all_results,
        }
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {output_path}")

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress test async migration endpoints")
    parser.add_argument(
        "--profile",
        choices=["conversation", "tunnel_github"],
        default="conversation",
        help="Which set of endpoints to stress",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write JSON results to this file for baseline comparison",
    )
    args = parser.parse_args()
    return asyncio.run(run_stress_for_profile(args.profile, args.output))


if __name__ == "__main__":
    sys.exit(main())
