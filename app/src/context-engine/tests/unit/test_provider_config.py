"""Tests for CLI OAuth provider configuration."""

from __future__ import annotations

import pytest

from adapters.inbound.cli.provider_config import (
    DEFAULT_LINEAR_CLI_CLIENT_ID,
    get_client_id,
)


def test_get_client_id_uses_builtin_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LINEAR_CLIENT_ID", raising=False)
    assert get_client_id("linear") == DEFAULT_LINEAR_CLI_CLIENT_ID


def test_get_client_id_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_CLIENT_ID", "override-client-id")
    assert get_client_id("linear") == "override-client-id"
