"""Shared pytest fixtures for root Potpie CLI, daemon, and runtime tests."""

from __future__ import annotations

import logging
import shutil
import socket
import tempfile
import webbrowser
from collections.abc import Callable, Generator
from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie.daemon.runtime.context import ServiceEndpoints, ShellContext
from potpie.runtime import build_product_shell


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture()
def short_socket_dir() -> Generator[Path, None, None]:
    path = Path(tempfile.mkdtemp(prefix="potpie-d-", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def wait_for_condition(
    condition: Callable[[], bool],
    *,
    timeout_s: float = 2.5,
    interval_s: float = 0.05,
    error_message: str = "condition was not met before timeout",
) -> None:
    import asyncio

    remaining = timeout_s
    while remaining > 0:
        if condition():
            return
        await asyncio.sleep(interval_s)
        remaining -= interval_s
    raise TimeoutError(error_message)


@pytest.fixture()
def daemon_ctx(tmp_path: Path) -> ShellContext:
    return ShellContext(
        config={},
        data_dir=tmp_path,
        logger=logging.getLogger("test"),
        endpoints=ServiceEndpoints(),
    )


@pytest.fixture()
def root_test_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a root product HostShell with deterministic in-memory engine state."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return build_product_shell(backend=InMemoryGraphBackend())


@pytest.fixture(autouse=True)
def _default_in_process_cli_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep existing CLI unit tests on the direct host unless they opt into daemon mode."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")
    monkeypatch.setenv("POTPIE_RUNTIME_MODE", "in-process")
    monkeypatch.setenv("POTPIE_GRAPH_BACKEND", "in_memory")
    monkeypatch.setenv("POTPIE_HOME", str(tmp_path / "potpie-runtime"))


@pytest.fixture(autouse=True)
def _reset_cli_state():
    """Reset process-wide injected CLI state after each test."""
    yield
    try:
        from potpie.cli.commands import _common

        _common._state["store"] = None
        _common._state["host"] = None
        _common._state["runtime"] = None
        _common._state["engine_view"] = None
        _common._state["json"] = False
        _common._state["verbose"] = False
        from potpie.runtime import reset_runtime

        reset_runtime()
    except Exception:
        logging.getLogger(__name__).debug(
            "failed to reset CLI test state", exc_info=True
        )


@pytest.fixture(autouse=True)
def _reset_product_analytics_state():
    """Keep product analytics globals isolated between tests."""
    _reset_product_analytics_globals()

    yield

    _reset_product_analytics_globals()


def _reset_product_analytics_globals() -> None:
    from potpie.cli.telemetry import product_analytics

    product_analytics._flush_product_analytics_dispatcher()
    product_analytics._dispatcher = product_analytics._ProductAnalyticsDispatcher()
    product_analytics._sink = product_analytics.NoOpProductAnalyticsSink()


@pytest.fixture(autouse=True)
def _no_real_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never open a real browser during tests."""
    monkeypatch.setattr(webbrowser, "open", lambda *args, **kwargs: False)
