"""Shared pytest fixtures for root Potpie CLI tests."""

from __future__ import annotations

import logging
import shutil
import socket
import tempfile
import webbrowser
from collections.abc import Callable, Generator
from pathlib import Path

import pytest

from potpie.daemon.runtime.context import ServiceEndpoints, ShellContext


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


@pytest.fixture(autouse=True)
def _default_in_process_cli_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI unit tests on the direct host unless they opt into daemon mode."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")


@pytest.fixture(autouse=True)
def _isolated_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Never let a test resolve the *developer's* ``~/.potpie``.

    These tests run the CLI in-process, so an unset ``CONTEXT_ENGINE_HOME``
    lands every home-rooted default — pot store, resource store, the
    ``falkordb_lite`` db file — on the live home and mutates it for real. Pin it
    suite-wide; the few tests that assert the *unset* default delete it
    themselves.
    """
    monkeypatch.setenv(
        "CONTEXT_ENGINE_HOME", str(tmp_path_factory.mktemp("potpie-home"))
    )


@pytest.fixture(autouse=True)
def _reset_cli_state():
    """Reset process-wide injected CLI state after each test."""
    yield
    try:
        from potpie.cli.commands import _common

        _common._state["store"] = None
        _common._state["host"] = None
        _common._state["json"] = False
        _common._state["verbose"] = False
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
    """Never open a real browser from CLI authentication tests."""
    monkeypatch.setattr(webbrowser, "open", lambda *args, **kwargs: False)
