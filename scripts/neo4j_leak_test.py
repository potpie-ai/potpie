#!/usr/bin/env python3
"""
Neo4j leak test — real-time check that driver/service cleanup prevents connection and memory growth.

Run locally with Neo4j + app .env (and optional running app for API mode).

Modes:
  --direct   (default) Create/close a Neo4j driver in a loop (no app imports), sample process RSS
             and optional Neo4j connection count. Proves driver close() works; no running app.
  --api      Hit POST /api/v1/knowledge-graph/semantic-search in a loop. Tests the real app path
             (InferenceService created per request and closed in finally). Requires BASE_URL,
             AUTH_HEADER, PROJECT_ID. Use SERVER_PID to sample backend process memory.

Usage:
  # Direct mode (Neo4j + Postgres from .env; no server needed)
  uv run python scripts/neo4j_leak_test.py --direct --iterations 80

  # Direct mode, with Neo4j connection count (requires Neo4j 4.x / 5.x with listConnections)
  uv run python scripts/neo4j_leak_test.py --direct --iterations 80 --neo4j-connections

  # API mode – semantic-search (app must be running)
  BASE_URL=http://localhost:8001 AUTH_HEADER="Bearer <token>" PROJECT_ID=<uuid> \\
    uv run python scripts/neo4j_leak_test.py --api --profile semantic-search --iterations 50

  # API mode – parse (exercises ParsingService + CodeGraphService cleanup path)
  BASE_URL=http://localhost:8001 AUTH_HEADER="Bearer <token>" \\
    uv run python scripts/neo4j_leak_test.py --api --profile parse --iterations 30
  # Optional: REPO_NAME=owner/repo BRANCH_NAME=main (default: octocat/Hello-World, main)

  # Optional: sample another process (e.g. backend) memory in API mode
  SERVER_PID=12345 uv run python scripts/neo4j_leak_test.py --api --iterations 50

Exit code: 0 if leak check passed (growth within threshold), 1 if likely leak.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Load .env and ensure app is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

os.chdir(repo_root)

from dotenv import load_dotenv
load_dotenv(repo_root / ".env")


def get_rss_mb(pid: int | None = None) -> float | None:
    """Current process or given PID RSS in MiB. Returns None if unavailable."""
    try:
        import psutil
        p = psutil.Process(pid or os.getpid())
        return p.memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        if pid is not None:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # kB -> MiB
        else:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
    except Exception:
        pass
    return None


def get_neo4j_connection_count(uri: str, user: str, password: str) -> int | None:
    """Return number of Bolt connections seen by Neo4j server, or None if query not supported."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            with driver.session() as session:
                # Neo4j 4.x
                result = session.run("CALL dbms.listConnections()")
                count = len(list(result))
                return count
        except Exception:
            try:
                # Neo4j 5.x: SHOW CONNECTIONS (if available)
                with driver.session() as session:
                    result = session.run("SHOW CONNECTIONS YIELD connectionId RETURN count(*) AS c")
                    rec = result.single()
                    return rec["c"] if rec else None
            except Exception:
                return None
        finally:
            driver.close()
    except Exception:
        return None


def run_direct_mode(iterations: int, sample_neo4j_connections: bool) -> bool:
    """Create/close Neo4j driver in a loop (no app imports); check RSS and Neo4j connection count stabilize.
    This verifies that driver close() works. To test the full app request path (InferenceService/CodeGraphService
    cleanup), use --api mode with the app running."""
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not user or not password:
        print("ERROR: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD must be set in .env")
        return False

    rss_samples: list[float] = []
    conn_samples: list[int] = []

    print(f"Direct mode: {iterations} iterations (Neo4j driver create / run / close)")
    print("Sampling RSS and optionally Neo4j connection count...")

    for i in range(iterations):
        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            with driver.session() as session:
                session.run("RETURN 1 AS c")
        except Exception as e:
            if i == 0:
                print(f"ERROR: Could not connect to Neo4j at {uri}: {e}")
                print("Ensure Neo4j is running and NEO4J_URI/USERNAME/PASSWORD are correct.")
            return False
        finally:
            driver.close()

        rss = get_rss_mb()
        if rss is not None:
            rss_samples.append(rss)
        if sample_neo4j_connections:
            nc = get_neo4j_connection_count(uri, user, password)
            if nc is not None:
                conn_samples.append(nc)

        if (i + 1) % 20 == 0:
            rss_str = f"  RSS={rss:.1f} MiB" if rss is not None else "  RSS=N/A"
            conn_str = f"  Neo4j_connections={conn_samples[-1]}" if conn_samples else ""
            print(f"  iteration {i + 1}/{iterations}{rss_str}{conn_str}")

    # Leak check: trend over second half of run
    n = len(rss_samples)
    if n < 20:
        print("WARN: Too few RSS samples; run more iterations (e.g. 80)")
        return True
    half = n // 2
    first_half_avg = sum(rss_samples[:half]) / half
    second_half_avg = sum(rss_samples[half:]) / (n - half)
    rss_growth_mb = second_half_avg - first_half_avg

    rss_threshold_mb = 50.0
    conn_ok = True
    if conn_samples:
        conn_first = conn_samples[: len(conn_samples) // 2]
        conn_second = conn_samples[len(conn_samples) // 2 :]
        if conn_first and conn_second:
            avg_first = sum(conn_first) / len(conn_first)
            avg_second = sum(conn_second) / len(conn_second)
            if avg_second > avg_first + 5:
                conn_ok = False

    print("")
    print(f"RSS: first half avg = {first_half_avg:.1f} MiB, second half avg = {second_half_avg:.1f} MiB")
    print(f"RSS growth (second - first half avg) = {rss_growth_mb:.1f} MiB (threshold {rss_threshold_mb} MiB)")
    if conn_samples:
        print(f"Neo4j connections: sample range {min(conn_samples)}–{max(conn_samples)}")
        if not conn_ok:
            print("Neo4j connection count increased over run — possible leak.")

    if rss_growth_mb > rss_threshold_mb:
        print("FAIL: RSS growth exceeds threshold; possible memory leak.")
        return False
    if not conn_ok:
        print("FAIL: Neo4j connection count grew; possible connection leak.")
        return False
    print("PASS: No significant RSS or connection growth detected.")
    print("(Direct mode tests raw driver only. Use --api with app running to test app request-path cleanup.)")
    return True


def run_api_mode(
    base_url: str,
    auth_header: str,
    profile: str,
    iterations: int,
    server_pid: int | None,
    *,
    project_id: str = "",
    repo_name: str = "",
    branch_name: str = "main",
) -> bool:
    """Hit semantic-search or parse endpoint in a loop; optionally sample SERVER_PID RSS."""
    import httpx

    base = base_url.rstrip("/")
    headers = {"Authorization": auth_header, "Content-Type": "application/json"}

    if profile == "parse":
        url = f"{base}/api/v1/parse"
        # ParsingRequest: repo_name or repo_path required. ParsingService is created and closed in controller.
        repo = repo_name or os.getenv("REPO_NAME", "octocat/Hello-World")
        branch = branch_name or os.getenv("BRANCH_NAME", "main")
        payload = {"repo_name": repo, "branch_name": branch}
        print(f"API mode (parse): {iterations} requests to {url} (repo={repo}, branch={branch})")
        # Parse can be slow (Celery) or return quickly; use longer timeout
        timeout = 60.0
    else:
        url = f"{base}/api/v1/knowledge-graph/semantic-search"
        payload = {"query": "authentication", "project_id": project_id, "top_k": 5}
        print(f"API mode (semantic-search): {iterations} requests to {url}")
        timeout = 30.0

    rss_samples: list[float] = []
    errors = 0

    with httpx.Client(timeout=timeout) as client:
        for i in range(iterations):
            r = client.post(url, json=payload, headers=headers)
            if r.status_code not in (200, 201, 202):
                errors += 1
            if server_pid is not None:
                rss = get_rss_mb(server_pid)
                if rss is not None:
                    rss_samples.append(rss)
            if (i + 1) % 10 == 0:
                rss_str = f"  server RSS={rss_samples[-1]:.1f} MiB" if rss_samples else ""
                print(f"  request {i + 1}/{iterations}  errors={errors}{rss_str}")

    if errors:
        print(f"WARN: {errors}/{iterations} requests failed (check auth and params)")
    if not rss_samples:
        print("INFO: Set SERVER_PID to sample server process memory.")
        return True

    n = len(rss_samples)
    half = n // 2
    if half < 5:
        return True
    first_avg = sum(rss_samples[:half]) / half
    second_avg = sum(rss_samples[half:]) / (n - half)
    growth = second_avg - first_avg
    print(f"Server RSS growth (second - first half avg) = {growth:.1f} MiB")
    if growth > 50.0:
        print("FAIL: Server RSS growth suggests possible leak.")
        return False
    print("PASS: No significant server RSS growth.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Neo4j leak test: verify driver/service cleanup (direct or API mode)"
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        default=True,
        help="Direct mode: create/close services in-process (default)",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="API mode: hit an endpoint in a loop (requires BASE_URL, AUTH_HEADER)",
    )
    parser.add_argument(
        "--profile",
        choices=["semantic-search", "parse"],
        default="semantic-search",
        help="API profile: semantic-search (needs PROJECT_ID) or parse (needs REPO_NAME, optional BRANCH_NAME)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=80,
        help="Number of iterations (default 80)",
    )
    parser.add_argument(
        "--neo4j-connections",
        action="store_true",
        help="In direct mode, sample Neo4j connection count each iteration",
    )
    args = parser.parse_args()

    if args.api:
        args.direct = False

    if args.direct:
        ok = run_direct_mode(args.iterations, args.neo4j_connections)
    else:
        base_url = os.getenv("BASE_URL", "http://localhost:8001")
        auth = os.getenv("AUTH_HEADER", "").strip()
        server_pid_str = os.getenv("SERVER_PID")
        server_pid = int(server_pid_str) if server_pid_str else None
        if not auth:
            print("ERROR: In API mode set AUTH_HEADER (and optionally SERVER_PID)")
            return 1
        if args.profile == "semantic-search":
            project_id = os.getenv("PROJECT_ID", "")
            if not project_id:
                print("ERROR: For profile semantic-search set PROJECT_ID")
                return 1
            ok = run_api_mode(
                base_url, auth, args.profile, args.iterations, server_pid,
                project_id=project_id,
            )
        else:
            repo_name = os.getenv("REPO_NAME", "octocat/Hello-World")
            branch_name = os.getenv("BRANCH_NAME", "main")
            ok = run_api_mode(
                base_url, auth, args.profile, args.iterations, server_pid,
                repo_name=repo_name, branch_name=branch_name,
            )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
