"""Unit tests for Atlassian auth_commands helpers."""

from __future__ import annotations

import pytest

from adapters.inbound.cli import auth_commands


def test_handle_already_connected_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    printed: list[dict] = []
    monkeypatch.setattr(
        auth_commands,
        "print_plain_line",
        lambda *args, **kwargs: printed.append(kwargs.get("json_payload") or {}),
    )
    auth_commands._handle_already_connected(
        "jira",
        {
            "auth_type": "api_token",
            "site_url": "https://team.atlassian.net",
            "site_name": "team",
        },
    )
    assert printed[-1].get("already_connected") is True


def test_run_product_use_result_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    blobs: list[dict] = []
    monkeypatch.setattr(
        auth_commands,
        "print_json_blob",
        lambda payload, **kwargs: blobs.append(payload),
    )

    auth_commands._run_product_use_result(
        {
            "product": "jira",
            "workspace_key": "ENG",
            "workspace_name": "Engineering",
            "items": [{"key": "ENG-1"}],
        },
        product_label="Jira",
    )

    assert blobs[0]["workspace_key"] == "ENG"
