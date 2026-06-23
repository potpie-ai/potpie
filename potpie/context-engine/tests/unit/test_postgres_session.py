"""Postgres session URL resolution."""

from __future__ import annotations

import pytest

from adapters.outbound.postgres.session import database_url

_DB_KEYS = (
    "DATABASE_URL",
    "POSTGRES_URL",
    "CONTEXT_ENGINE_DATABASE_URL",
    "POSTGRES_SERVER",
)


def _clear_db_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _DB_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_database_url_postgres_server(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("POSTGRES_SERVER", "postgresql://u:p@localhost:5432/db")
    assert database_url() == "postgresql://u:p@localhost:5432/db"


def test_database_url_explicit_over_postgres_server(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("POSTGRES_SERVER", "postgresql://legacy:5432/x")
    monkeypatch.setenv("DATABASE_URL", "postgresql://primary:5432/y")
    assert database_url() == "postgresql://primary:5432/y"
