#!/usr/bin/env python3
"""
Neo4j monitor — watch Neo4j connection count (and optional app RSS) in real time
while you trigger parsing or other activity from the UI.

Uses .env for NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD. Optional: SERVER_PID
to sample backend process memory.

Usage:
  # Default: sample every 3s, Neo4j connections only
  uv run python scripts/neo4j_monitor.py

  # Include backend process RSS (e.g. gunicorn master PID)
  SERVER_PID=$(pgrep -f "gunicorn.*potpie" | head -1) uv run python scripts/neo4j_monitor.py

  # Sample every 5 seconds
  uv run python scripts/neo4j_monitor.py --interval 5

  # One-shot (single sample then exit)
  uv run python scripts/neo4j_monitor.py --once

Press Ctrl+C to stop.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from dotenv import load_dotenv
load_dotenv(repo_root / ".env")


def get_rss_mb(pid: int | None = None) -> float | None:
    """RSS in MiB for process (or current if pid is None). Returns None if unavailable."""
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
                        return int(line.split()[1]) / 1024
        else:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
    except Exception:
        pass
    return None


def get_connection_count(driver) -> int | None:
    """Number of Bolt connections (Neo4j 4.x or 5.x). Uses existing driver."""
    try:
        with driver.session() as session:
            result = session.run("CALL dbms.listConnections()")
            return len(list(result))
    except Exception:
        try:
            with driver.session() as session:
                result = session.run(
                    "SHOW CONNECTIONS YIELD connectionId RETURN count(*) AS c"
                )
                rec = result.single()
                return int(rec["c"]) if rec and rec["c"] is not None else None
        except Exception:
            return None
    return None


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Monitor Neo4j connection count (and optional app RSS) in real time."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Seconds between samples (default: 3)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one sample and exit",
    )
    args = parser.parse_args()

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    server_pid_str = os.getenv("SERVER_PID")
    server_pid = int(server_pid_str) if server_pid_str and server_pid_str.isdigit() else None

    if not uri or not user or not password:
        print("ERROR: Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env", file=sys.stderr)
        return 1

    from neo4j import GraphDatabase
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
    except Exception as e:
        print(f"ERROR: Cannot connect to Neo4j at {uri}: {e}", file=sys.stderr)
        return 1

    try:
        if server_pid is not None:
            print("time                    connections  app_rss_mib")
            print("-" * 48)
        else:
            print("time                    connections")
            print("-" * 35)

        while True:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            conn = get_connection_count(driver)
            conn_str = str(conn) if conn is not None else "?"
            if server_pid is not None:
                rss = get_rss_mb(server_pid)
                rss_str = f"{rss:.1f}" if rss is not None else "N/A"
                print(f"{ts}  {conn_str:>10}     {rss_str:>10}")
            else:
                print(f"{ts}  {conn_str:>10}")

            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            driver.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
