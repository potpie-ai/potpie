"""Small fixtures for auth commands that run without environment loading."""

from __future__ import annotations

import pytest

from potpie.cli.auth import auth_commands


@pytest.fixture
def auth_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        auth_commands, "ensure_runtime_environment_loaded", lambda: None
    )
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))


@pytest.fixture
def auth_json_cli(auth_cli: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
