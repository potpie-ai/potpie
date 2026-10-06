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

# tests/docs is Node (docs-check.mjs). Do not collect it as pytest.
collect_ignore = ["docs"]


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


@pytest.fixture(autouse=True)
def _default_in_process_cli_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI unit tests on the direct host unless they opt into daemon mode."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")


@pytest.fixture(autouse=True)
def _isolated_config_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Never let a test read or write the *developer's* ``~/.config/potpie``.

    The telemetry identity file and the telemetry spool live there. A suite
    that appended to the live spool would leave test events for the
    developer's next real command to ship with real keys. Per test, so each
    test starts from a fresh install identity and an empty spool. A sibling of
    ``tmp_path`` rather than inside it, since tests list their own
    ``tmp_path``.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg-config")))


@pytest.fixture(autouse=True)
def _no_real_telemetry_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a developer's real telemetry keys reach a test.

    In the ``dev`` environment the CLI merges the project ``.env`` into the
    process environment, but only for keys that are missing. Pinning these to
    empty keeps a real Sentry DSN or analytics key out of every in-process CLI
    run; tests that exercise an enabled sink set their own placeholder values.
    """
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "")
    monkeypatch.setenv("POTPIE_POSTHOG_API_KEY", "")


@pytest.fixture(autouse=True)
def _reset_cli_state():
    """Reset process-wide injected CLI state after each test."""
    yield
    try:
        from potpie.cli.commands import _common

        runner = _common._state.get("engine_runner")
        manager = _common._state.get("engine_manager")
        if runner is not None and manager is not None:
            runner.run(manager.shutdown())
        if runner is not None:
            runner.close()
        _common._state["store"] = None
        _common._state["runtime"] = None
        _common._state["json"] = False
        _common._state["verbose"] = False
        _common._state["engine_runner"] = None
        _common._state["engine_manager"] = None
        _common._state["engine_runtime"] = None
        _common._state["engine_remote_host"] = None
        _common._state["engine_remote_home"] = None
        _common._state["pot_service"] = None
        _common._state["pot_service_runtime"] = None
        _common._state["root_product_services"] = {}
        _common._state["root_product_services_runtime"] = None
        _common._state["root_runtime"] = None
        _common._state["root_runtime_source"] = None
    except Exception:
        logging.getLogger(__name__).debug(
            "failed to reset CLI test state", exc_info=True
        )


@pytest.fixture(autouse=True)
def _reset_sentry_runtime_state():
    """Leave no Sentry client or metric recorder behind a test.

    The metrics runtime is process-global and configures once. A test that
    initialises a *real* client (a DSN pointed at ``example.invalid``) would
    otherwise keep it for every later test, and its queue would drain at
    interpreter exit after pytest has closed its streams. Close any real
    client without waiting and reset the runtime so each test starts
    unconfigured.
    """
    yield
    import types

    from potpie.cli.telemetry import sentry_runtime as cli_sentry_runtime
    from potpie_context_engine.bootstrap import sentry_metrics_runtime

    sdk = sentry_metrics_runtime._sentry_sdk
    if isinstance(sdk, types.ModuleType) and hasattr(sdk, "get_client"):
        try:
            sdk.get_client().close(timeout=0)
        except Exception:  # noqa: BLE001 - teardown must not fail the test
            pass
    sentry_metrics_runtime._configured = False
    sentry_metrics_runtime._enabled = False
    sentry_metrics_runtime._sentry_sdk = None
    cli_sentry_runtime.disable_cli_sentry()


@pytest.fixture(autouse=True)
def _reset_product_analytics_state():
    """Keep product analytics globals isolated between tests."""
    _reset_product_analytics_globals()
    yield
    _reset_product_analytics_globals()


def _reset_product_analytics_globals() -> None:
    from potpie.cli.telemetry import product_analytics, spool
    from potpie_context_engine.bootstrap import sentry_metrics_runtime

    product_analytics._sink = product_analytics.NoOpProductAnalyticsSink()
    sentry_metrics_runtime.set_metric_recorder(None)
    spool._appended = False
    spool.launch_after_append(False)


@pytest.fixture(autouse=True, scope="session")
def _never_launch_a_real_flusher():
    """No test may start a detached telemetry flusher.

    The spool registers an exit-time launch the first time a process appends,
    and the daemon's analytics launch one after every append. In-process CLI
    tests append into their per-test ``XDG_CONFIG_HOME``; by the time the
    interpreter exits that env is restored, so an exit-time launch would ship
    the developer's real spool. Off for the whole session; tests that observe
    a launch stub ``_launch`` themselves.
    """
    from potpie.cli.telemetry import spool

    spool.auto_spawn_enabled = False
    yield


@pytest.fixture(autouse=True)
def _stub_flusher_launch(monkeypatch: pytest.MonkeyPatch):
    from potpie.cli.telemetry import spool

    monkeypatch.setattr(spool, "_launch", lambda: None)


@pytest.fixture(autouse=True)
def _no_real_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never open a real browser from CLI authentication tests."""
    monkeypatch.setattr(webbrowser, "open", lambda *args, **kwargs: False)
