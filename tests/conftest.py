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
def _isolated_config_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Never let a test read or write the *developer's* ``~/.config/potpie``.

    The telemetry identity file lives there, and since it now also remembers
    which once-only activation events went out, a suite that touched the live
    file would both mutate it and — on the second run — see the marker already
    set and stop emitting the events the tests assert. Per test, not per
    session: each test starts from a fresh install identity. A sibling of
    ``tmp_path`` rather than inside it, since tests list their own ``tmp_path``.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg-config")))


@pytest.fixture(autouse=True)
def _isolated_harness_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Never let a test write into the *developer's* ``~/.claude`` & friends.

    ``CONTEXT_ENGINE_HOME`` above pins Potpie's own state, and deliberately does
    not pin this: skills install where the harness reads them, which is the real
    home directory. So a sandboxed suite still installed eleven skill files into
    the live ``~/.claude``, ``~/.cursor``, ``~/.agents`` and
    ``~/.config/opencode``, overwriting whatever versions were there — which
    made running these tests on a machine you also work on unsafe.
    """
    monkeypatch.setenv(
        "POTPIE_HARNESS_HOME", str(tmp_path_factory.mktemp("potpie-harness-home"))
    )


@pytest.fixture(autouse=True)
def _isolated_embedding_env() -> Generator[None, None, None]:
    """Never let one test's ``setup`` choose the next test's embedder.

    ``_apply_setup_embedding_env`` writes ``CONTEXT_ENGINE_EMBEDDER`` and
    ``CONTEXT_ENGINE_EMBEDDING_MODEL`` straight into ``os.environ`` so the
    components built later in the same process see the choice. In a CLI process
    that is right and ends at exit; in a suite it is a permanent write, and
    ``monkeypatch.delenv(..., raising=False)`` does not undo it because a
    variable that was absent records nothing to restore.

    The cost was a real, order-dependent failure: one test set the model to a
    name that does not exist, a later one set the mode to
    ``sentence-transformers``, and the first test after both to run a genuine
    ``setup`` tried to download that name from HuggingFace mid-suite.

    ``CONTEXT_ENGINE_HOST_MODE`` — the third variable written that way — is
    already pinned per test by ``_default_in_process_cli_host`` above.
    """
    import os

    names = ("CONTEXT_ENGINE_EMBEDDER", "CONTEXT_ENGINE_EMBEDDING_MODEL")
    before = {name: os.environ.get(name) for name in names}
    yield
    for name, value in before.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


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
def _reset_sentry_runtime_state():
    """Leave no Sentry client behind a test.

    The metrics runtime is process-global and configures once. A test that runs
    the CLI in-process with a DSN set therefore left a *real* client — with a
    real transport pointed at ``example.invalid`` — for every later test in the
    session, and those tests' command metrics queued in it. The CLI no longer
    flushes inside each command, so the queue used to drain at interpreter exit,
    after pytest had closed the capture streams: six "Logging error" blocks of
    urllib3 retry noise per run. Close any real client without waiting and
    reset the runtime so each test starts unconfigured.
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
    from potpie.cli.telemetry import product_analytics

    product_analytics._flush_product_analytics_dispatcher()
    product_analytics._dispatcher = product_analytics._ProductAnalyticsDispatcher()
    product_analytics._sink = product_analytics.NoOpProductAnalyticsSink()


@pytest.fixture(autouse=True)
def _no_real_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never open a real browser from CLI authentication tests."""
    monkeypatch.setattr(webbrowser, "open", lambda *args, **kwargs: False)
