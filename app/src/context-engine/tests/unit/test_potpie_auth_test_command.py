"""Unit tests for Potpie auth diagnostic commands."""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import main as cli_main
from adapters.inbound.cli.auth.firebase_session import FirebaseSession

runner = CliRunner()


class _FakeClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.base_url = args[0]
        self.headers = kwargs["auth_headers"]

    def list_context_pots(self) -> list[dict[str, Any]]:
        assert self.headers == {"Authorization": "Bearer id-token"}
        return [
            {
                "id": "pot-1",
                "slug": "demo-pot",
                "primary_repo_name": "org/repo",
            }
        ]


def test_auth_test_pots_lists_context_pots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "get_potpie_firebase_refresh_token", lambda: "refresh")
    monkeypatch.setattr(cli_main, "get_potpie_firebase_api_key", lambda: "firebase-key")
    monkeypatch.setattr(
        cli_main,
        "refresh_id_token",
        lambda token, *, firebase_api_key=None: FirebaseSession(
            id_token="id-token",
            refresh_token="new-refresh",
            expires_at=123.0,
        ),
    )
    updated: list[str] = []
    monkeypatch.setattr(cli_main, "update_potpie_firebase_refresh_token", updated.append)
    monkeypatch.setattr(cli_main, "resolve_potpie_api_base_url", lambda: "http://api")
    monkeypatch.setattr(cli_main, "PotpieContextApiClient", _FakeClient)

    result = runner.invoke(cli_main.app, ["--json", "auth", "test", "pots"])

    assert result.exit_code == 0, result.stdout
    assert '"ok": true' in result.stdout
    assert '"auth_type": "firebase_session"' in result.stdout
    assert '"slug": "demo-pot"' in result.stdout
    assert updated == ["new-refresh"]
