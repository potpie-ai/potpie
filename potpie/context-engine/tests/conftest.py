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


@pytest.fixture(autouse=True)
def _isolated_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Never let a test resolve the *developer's* ``~/.potpie``.

    Every home-rooted default (pot store, resource store, ``falkordb_lite``'s db
    file) falls back to ``Path.home() / ".potpie"`` when ``CONTEXT_ENGINE_HOME``
    is unset. A test that builds a real backend without passing a path therefore
    opens the live graph, spawns a server against it, and mutates it — which is
    how the conformance suite came to create an ``appendonlydir`` in a real home
    mid-run. Pin the variable suite-wide so that is structurally impossible; the
    handful of tests that assert the *unset* default delete it themselves.
    """
    monkeypatch.setenv(
        "CONTEXT_ENGINE_HOME", str(tmp_path_factory.mktemp("potpie-home"))
    )
