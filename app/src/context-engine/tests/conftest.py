"""Shared pytest fixtures for the context-engine test suite."""

from __future__ import annotations

import webbrowser

import pytest


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
