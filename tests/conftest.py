"""Shared pytest fixtures for root Potpie CLI tests."""

from __future__ import annotations

import logging
import webbrowser

import pytest


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _default_in_process_cli_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI unit tests on the direct host unless they opt into daemon mode."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")


@pytest.fixture(autouse=True)
def _reset_cli_state():
    """Reset process-wide injected CLI state after each test.

    ``commands/_common`` caches CLI flags, host, and store in module state; clear
    them so fakes injected by one test never leak into the next.
    """
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
    from potpie.cli.telemetry import product_analytics

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
