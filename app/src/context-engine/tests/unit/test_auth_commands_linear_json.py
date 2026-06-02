"""JSON output behavior for Linear auth commands."""

from __future__ import annotations

import json

import pytest

from adapters.inbound.cli import auth_commands


def test_linear_refresh_emits_single_json_document(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _provider: {
            "authenticated": True,
            "expires_at": 1,
            "login": "nihit",
            "email": "nihit@example.com",
            "site_name": "Acme",
            "auth_type": "oauth",
        },
    )
    monkeypatch.setattr(auth_commands, "token_needs_refresh", lambda _expires_at: True)
    monkeypatch.setattr(auth_commands, "_try_refresh_linear_session", lambda: True)

    auth_commands._run_linear_oauth_flow()

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["ok"] is True
    assert payload["provider"] == "linear"
    assert payload.get("refreshed") is True
