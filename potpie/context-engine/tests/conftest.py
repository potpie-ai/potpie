"""Shared pytest fixtures for the context-engine test suite."""

from __future__ import annotations

import logging
import shutil
import socket
import tempfile
import webbrowser
from collections.abc import Callable, Generator
from pathlib import Path

import pytest

from host.daemon_runtime.context import ServiceEndpoints, ShellContext


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
    """Keep existing CLI unit tests on the direct host unless they opt into daemon mode."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")


@pytest.fixture(autouse=True)
def _reset_cli_state():
    """Reset process-wide injected CLI state after each test.

    ``commands/_common`` caches CLI flags, host, and store in module state; clear
    them so fakes injected by one test never leak into the next.
    """
    yield
    try:
        from adapters.inbound.cli.commands import _common

        _common._state["store"] = None
        _common._state["host"] = None
        _common._state["json"] = False
        _common._state["verbose"] = False
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _reset_product_analytics_state():
    """Keep product analytics globals isolated between tests.

    The CLI uses one module-global background dispatcher per process. In the test
    process, earlier CLI/setup tests can leave queued analytics payloads behind;
    reset the dispatcher and sink so dispatcher tests only observe their own
    events.
    """

    _reset_product_analytics_globals()

    yield

    _reset_product_analytics_globals()


def _reset_product_analytics_globals() -> None:
    from adapters.inbound.cli.telemetry import product_analytics

    product_analytics._flush_product_analytics_dispatcher()
    product_analytics._dispatcher = product_analytics._ProductAnalyticsDispatcher()
    product_analytics._sink = product_analytics.NoOpProductAnalyticsSink()


@pytest.fixture(autouse=True)
def _no_real_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never open a real browser during tests.

    The CLI auth flows (``potpie login``, ``potpie auth github login``, Linear
    OAuth) call ``webbrowser.open(...)`` on the device-flow / sign-in URL. Under
    test that would pop a real browser tab (e.g. github.com/login/device). Patch
    the ``webbrowser`` module's ``open`` to a no-op for every test. Tests that
    need to assert the opened URL override this with their own
    ``monkeypatch.setattr(webbrowser, "open", ...)``, which wins for that test.
    """
    monkeypatch.setattr(webbrowser, "open", lambda *args, **kwargs: False)
