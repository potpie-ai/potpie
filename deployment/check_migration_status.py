#!/usr/bin/env python3
"""
Check if the encrypt_user_auth_provider_tokens migration (20251217190000) has been applied.
Run with port-forward active and POSTGRES_SERVER set, e.g.:
  export POSTGRES_SERVER='postgresql://user:pass@localhost:5432/db'
  python deployment/check_migration_status.py

If POSTGRES_SERVER uses an in-cluster host (e.g. pgbouncer.pgb-ns.svc.cluster.local),
the script rewrites it to localhost so you can use port-forward:
  kubectl port-forward svc/pgbouncer -n pgb-ns 5432:5432 &
"""
import os
import re
import sys


def _rewrite_k8s_host_to_localhost(url: str) -> str:
    """If URL host is a Kubernetes internal hostname, rewrite to localhost for port-forward."""
    # Match postgresql://user:pass@HOST:port/db (host can be hostname or hostname:port)
    match = re.match(
        r"(postgresql(?:\+[a-z0-9]+)?://[^@]+@)([^:/]+)(:\d+)?(/.*)?",
        url,
        re.IGNORECASE,
    )
    if not match:
        return url
    prefix, host, port_part, path_part = match.groups()
    port_part = port_part or ":5432"
    path_part = path_part or ""
    # Kubernetes internal hostnames
    if ".svc.cluster.local" in host or host in ("pgbouncer", "postgres", "postgresql"):
        return f"{prefix}localhost{port_part}{path_part}"
    return url


def main():
    url = os.getenv("POSTGRES_SERVER")
    if not url:
        print("Error: POSTGRES_SERVER not set. Example:")
        print("  export POSTGRES_SERVER='postgresql://user:pass@localhost:5432/db'")
        sys.exit(1)

    # When running locally, in-cluster hostnames don't resolve — use localhost (port-forward)
    original_url = url
    url = _rewrite_k8s_host_to_localhost(url)
    if url != original_url:
        print("Note: Rewrote in-cluster DB host to localhost (use port-forward in another terminal):")
        print("  kubectl port-forward svc/pgbouncer -n pgb-ns 5432:5432 &")
        print()

    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("Error: sqlalchemy not installed. Run: uv sync (or pip install sqlalchemy)")
        sys.exit(1)

    engine = create_engine(url, pool_pre_ping=True)
    migration_rev = "20251217190000"  # encrypt_user_auth_provider_tokens

    with engine.connect() as conn:
        # Check alembic_version table
        try:
            row = conn.execute(
                text("SELECT version_num FROM alembic_version")
            ).fetchone()
        except Exception as e:
            print(f"Error reading alembic_version: {e}")
            sys.exit(1)

        if not row:
            print("alembic_version is empty — no migrations have been applied.")
            sys.exit(0)

        current = row[0]
        applied = current == migration_rev or _is_after(current, migration_rev)

        print("==========================================")
        print("Migration check: encrypt_user_auth_provider_tokens")
        print("==========================================")
        print(f"Current alembic version in DB: {current}")
        print(f"Target migration revision:     {migration_rev}")
        print("------------------------------------------")
        if applied:
            print("Result: YES — this migration has already been applied.")
        else:
            print("Result: NO — this migration has NOT been applied yet.")
        print("==========================================")

        # Optional: count rows in user_auth_providers with tokens
        try:
            r = conn.execute(text("""
                SELECT COUNT(*) FROM user_auth_providers
                WHERE access_token IS NOT NULL OR refresh_token IS NOT NULL
            """)).scalar()
            print(f"Rows in user_auth_providers with tokens: {r}")
        except Exception:
            pass

    return 0


def _is_after(current: str, target: str) -> bool:
    """Heuristic: if current is a different revision, we'd need the full chain.
    For this script we only care if 20251217190000 is applied; Alembic stores
    only the head revision, so if current == 20251217190000 it's applied.
    If we have a later revision (e.g. 202602...), then 20251217190000 is in the chain.
    """
    if current == target:
        return True
    # Later revisions are often larger timestamps
    try:
        return int(current) >= int(target)
    except (ValueError, TypeError):
        return current >= target


if __name__ == "__main__":
    sys.exit(main())
