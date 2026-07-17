from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Response
from sqlalchemy import text

from potpie_context_engine.adapters.inbound.http.api.v1.context.router import context_router
from potpie_context_engine.adapters.inbound.http.deps import get_container_or_503, get_db_optional

api_router = APIRouter()
api_router.include_router(context_router, prefix="/context", tags=["context"])


@api_router.get("/health")
def health() -> dict[str, str]:
    """Liveness — process is up. Unchanged contract (kept for scrapers)."""
    return {"status": "ok"}


def _check_postgres(db: Any) -> tuple[bool, str | None]:
    if db is None:
        return False, "no DATABASE_URL"
    try:
        db.execute(text("SELECT 1"))
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def _check_neo4j(container: Any) -> tuple[bool, str | None]:
    try:
        s = container.settings
        if not s.is_enabled():
            return False, "context graph disabled"
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            s.neo4j_uri(), auth=(s.neo4j_user(), s.neo4j_password())
        )
        try:
            driver.verify_connectivity()
            return True, None
        finally:
            driver.close()
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def _check_redis(container: Any) -> tuple[bool, str | None]:
    pub = getattr(container, "event_stream_publisher", None)
    client = getattr(pub, "_client", None)
    if client is None:
        return True, "not configured (NoOp)"  # optional dependency
    try:
        client.ping()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


@api_router.get("/ready")
def ready(
    response: Response,
    container: Any = Depends(get_container_or_503),
    db: Any = Depends(get_db_optional),
) -> dict[str, Any]:
    """Readiness — dependencies are reachable.

    Postgres + Neo4j are hard requirements; Redis is optional (NoOp
    fallback) so it never fails readiness. Probes run inside a span and
    emit an ``up``/``down`` gauge per dependency for Grafana.
    """
    obs = container.observability
    checks: dict[str, dict[str, Any]] = {}
    with obs.span("http.ready", kind="server"):
        for name, fn, hard in (
            ("postgres", lambda: _check_postgres(db), True),
            ("neo4j", lambda: _check_neo4j(container), True),
            ("redis", lambda: _check_redis(container), False),
        ):
            ok, detail = fn()
            checks[name] = {"ok": ok, "detail": detail, "required": hard}
            obs.gauge(
                "ce.dependency_up",
                1 if ok else 0,
                attributes={"dependency": name},
            )

    ready_flag = all(c["ok"] for c in checks.values() if c["required"])
    if not ready_flag:
        response.status_code = 503
    return {"ready": ready_flag, "checks": checks}
