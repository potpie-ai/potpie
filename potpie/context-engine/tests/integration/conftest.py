"""Shared fixtures for the live end-to-end integration tests.

These fixtures are **opt-in** — only the e2e tests that request them are
affected, so the existing (Neo4j-free) integration tests are untouched.

Environment is loaded from the repo-root ``.env`` (Neo4j vars only) without
overriding anything already set in the real environment. If Neo4j is not
reachable the dependent tests skip rather than fail.
"""

from __future__ import annotations

import asyncio
import socket
import uuid
from pathlib import Path
from urllib.parse import urlparse

import pytest

# Only these keys are pulled from .env — we deliberately do NOT load DB / LLM
# vars so the e2e suite stays deterministic (NullQueryAgent, NoOp telemetry).
_NEO4J_KEYS = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")


def _find_dotenv() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def _parse_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            out[key] = value
    return out


def _bolt_reachable(uri: str) -> bool:
    parsed = urlparse(uri)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 7687
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture()
def live_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set Neo4j env from real env or repo .env; skip if missing/unreachable."""
    import os

    dotenv = _parse_dotenv(_find_dotenv()) if _find_dotenv() else {}
    resolved: dict[str, str] = {}
    for key in _NEO4J_KEYS:
        value = os.environ.get(key) or dotenv.get(key)
        if value:
            monkeypatch.setenv(key, value)
            resolved[key] = value
    monkeypatch.setenv("CONTEXT_GRAPH_ENABLED", "1")

    if not all(resolved.get(k) for k in _NEO4J_KEYS):
        pytest.skip("Neo4j env (NEO4J_URI/USERNAME/PASSWORD) not configured")
    if not _bolt_reachable(resolved["NEO4J_URI"]):
        pytest.skip(f"Neo4j not reachable at {resolved['NEO4J_URI']}")
    return resolved


@pytest.fixture()
def settings(live_env):
    from potpie.context_engine.adapters.outbound.settings_env import EnvContextEngineSettings

    return EnvContextEngineSettings()


@pytest.fixture()
def pot_id() -> str:
    """A unique pot id per test, so live data never collides across tests."""
    return f"e2e_{uuid.uuid4().hex[:12]}"


@pytest.fixture()
def repo_name() -> str:
    return "acme/platform"


@pytest.fixture()
def container(settings, pot_id, repo_name):
    """A fully-wired container backed by live Neo4j, no LLM / DB / Redis.

    The pot resolves to a single GitHub repo via ``ExplicitPotResolution``.
    The pot's graph partition is reset on teardown.
    """
    from potpie.context_engine.bootstrap.ingestion_server import build_ingestion_server
    from potpie.context_engine.bootstrap.http_projects import ExplicitPotResolution

    c = build_ingestion_server(
        settings=settings,
        pots=ExplicitPotResolution({pot_id: repo_name}),
    )
    yield c
    try:
        asyncio.run(c.graph_writer.reset_pot(pot_id))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Real Postgres test database (created inside the configured instance)
# ---------------------------------------------------------------------------

import os  # noqa: E402

_DOTENV: dict[str, str] | None = None


def _dotenv() -> dict[str, str]:
    global _DOTENV
    if _DOTENV is None:
        path = _find_dotenv()
        _DOTENV = _parse_dotenv(path) if path else {}
    return _DOTENV


def _env_or_dotenv(*keys: str) -> str | None:
    d = _dotenv()
    for key in keys:
        value = os.environ.get(key) or d.get(key)
        if value:
            return value
    return None


def _pg_url_with_available_driver(url: str) -> str:
    """Pin the SQLAlchemy driver to one that's actually installed.

    The configured URL often uses the bare ``postgresql://`` scheme, which
    SQLAlchemy maps to psycopg2; many environments now ship psycopg v3 only.
    Rewrite to ``postgresql+psycopg://`` when psycopg2 is missing but psycopg
    is present so the DB fixture runs on either driver instead of erroring at
    ``create_engine``.
    """
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(url)
    if parsed.scheme in ("postgresql", "postgres"):
        try:
            import psycopg2  # noqa: F401
        except ModuleNotFoundError:
            try:
                import psycopg  # noqa: F401

                parsed = parsed._replace(scheme="postgresql+psycopg")
            except ModuleNotFoundError:
                pass
    return urlunparse(parsed)


class _TestDB:
    def __init__(self, url: str, sessionmaker, engine) -> None:
        self.url = url
        self.sessionmaker = sessionmaker
        self.engine = engine


@pytest.fixture(scope="session")
def pg_test_db():
    """Create a throwaway database in the configured Postgres instance.

    Same instance as ``POSTGRES_SERVER`` (or ``DATABASE_URL``), a brand-new
    database so tests never touch the app database. Schema is built from the
    context-engine ORM ``Base``; the database is dropped on teardown.
    """
    from urllib.parse import urlparse, urlunparse

    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    admin_url = _env_or_dotenv("DATABASE_URL", "POSTGRES_SERVER", "POSTGRES_URL")
    if not admin_url:
        pytest.skip("Postgres not configured (POSTGRES_SERVER / DATABASE_URL)")

    admin_url = _pg_url_with_available_driver(admin_url)
    parsed = urlparse(admin_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            pass
    except OSError:
        pytest.skip(f"Postgres not reachable at {host}:{port}")

    test_db = f"ctx_e2e_{uuid.uuid4().hex[:12]}"
    maintenance_url = urlunparse(parsed._replace(path="/postgres"))
    test_url = urlunparse(parsed._replace(path=f"/{test_db}"))

    admin = create_engine(maintenance_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{test_db}"'))

    engine = create_engine(test_url, pool_pre_ping=True)
    from potpie.context_engine.adapters.outbound.postgres.models import Base

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    yield _TestDB(url=test_url, sessionmaker=factory, engine=engine)

    engine.dispose()
    with admin.connect() as conn:
        conn.execute(
            text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = :db AND pid <> pg_backend_pid()"
            ),
            {"db": test_db},
        )
        conn.execute(text(f'DROP DATABASE IF EXISTS "{test_db}"'))
    admin.dispose()


# ---------------------------------------------------------------------------
# Real LLM
# ---------------------------------------------------------------------------

_LLM_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")
_LLM_MODEL_VARS = (
    # The read-side query-agent + answer-synthesis surfaces were removed when
    # the engine collapsed onto one evidence-envelope read contract; only the
    # reconciliation (ingestion) LLM surface remains.
    "CONTEXT_ENGINE_RECONCILIATION_MODEL",
)


@pytest.fixture()
def llm_env(monkeypatch: pytest.MonkeyPatch) -> str:
    """Load LLM keys + select the reconciliation (ingestion) model.

    Skips if no API key is configured. The read path is deterministic (one
    evidence-envelope contract, no LLM synthesis), so the only LLM surface left
    is reconciliation during ingestion.
    """
    have_key = False
    for key in _LLM_KEYS:
        value = _env_or_dotenv(key)
        if value:
            monkeypatch.setenv(key, value)
            have_key = True
    if not have_key:
        pytest.skip("No LLM API key configured")

    model = (
        _env_or_dotenv("CONTEXT_ENGINE_RECONCILIATION_MODEL")
        or "openai-responses:gpt-5.4-mini"
    )
    for var in _LLM_MODEL_VARS:
        monkeypatch.setenv(var, model)
    # Bound LLM time so a slow/hung run can't stall the suite.
    monkeypatch.setenv("CONTEXT_ENGINE_AGENT_RUN_TIMEOUT_SECS", "180")
    monkeypatch.setenv("CONTEXT_ENGINE_QUERY_AGENT_TIMEOUT_SECS", "90")
    return model


@pytest.fixture()
def pipeline_container(
    monkeypatch: pytest.MonkeyPatch, settings, llm_env, pg_test_db, pot_id, repo_name
):
    """Container with a real reconciliation agent + real Postgres + Neo4j.

    ``build_ingestion_server`` attaches the context graph + read tools to the agent
    and wires the real query agent / answer synthesizer from the LLM env. The
    Celery job queue is replaced with a NoOp so batches are driven in-process.
    """
    monkeypatch.setenv("DATABASE_URL", pg_test_db.url)

    from potpie.context_engine.bootstrap.ingestion_server import build_ingestion_server
    from potpie.context_engine.bootstrap.http_projects import ExplicitPotResolution
    from potpie.context_engine.domain.ports.context_graph_job_queue import NoOpContextGraphJobQueue
    from potpie.context_engine.adapters.outbound.reconciliation.pydantic_deep_agent import (
        PydanticDeepReconciliationAgent,
    )

    agent = PydanticDeepReconciliationAgent()  # model resolved from llm_env
    c = build_ingestion_server(
        settings=settings,
        pots=ExplicitPotResolution({pot_id: repo_name}),
        reconciliation_agent=agent,
        jobs=NoOpContextGraphJobQueue(),
    )
    yield c
    try:
        asyncio.run(c.graph_writer.reset_pot(pot_id))
    except Exception:
        pass


@pytest.fixture()
def db_container(settings, pg_test_db, pot_id, repo_name):
    """Container with a real (constructed-but-not-invoked) agent + real Postgres.

    Lets the Postgres round-trip tests admit ``agent_reconciliation`` events
    (which require an agent to be *present*) without making any LLM calls, so
    they run even when no API key is configured.
    """
    from potpie.context_engine.bootstrap.ingestion_server import build_ingestion_server
    from potpie.context_engine.bootstrap.http_projects import ExplicitPotResolution
    from potpie.context_engine.domain.ports.context_graph_job_queue import NoOpContextGraphJobQueue
    from potpie.context_engine.adapters.outbound.reconciliation.pydantic_deep_agent import (
        PydanticDeepReconciliationAgent,
    )

    c = build_ingestion_server(
        settings=settings,
        pots=ExplicitPotResolution({pot_id: repo_name}),
        reconciliation_agent=PydanticDeepReconciliationAgent(),
        jobs=NoOpContextGraphJobQueue(),
    )
    yield c
    try:
        asyncio.run(c.graph_writer.reset_pot(pot_id))
    except Exception:
        pass
