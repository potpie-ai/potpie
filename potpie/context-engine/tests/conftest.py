"""Shared pytest fixtures for the context-engine test suite."""

from __future__ import annotations

import pytest


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _default_in_process_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep engine tests deterministic unless they explicitly select daemon mode."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")
