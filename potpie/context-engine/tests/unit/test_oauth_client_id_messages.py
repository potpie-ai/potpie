"""Tests for OAuth client ID user-facing error messages."""

from __future__ import annotations

import pytest

from adapters.outbound.cli_auth import oauth_client_id_messages as msgs


def test_missing_linear_message_for_end_users(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(msgs, "looks_like_local_development", lambda: False)
    message = msgs.missing_linear_client_id_message()
    assert "report this" in message.lower()
    assert "github.com/potpie-ai/potpie/issues" in message


def test_missing_github_message_for_end_users(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(msgs, "looks_like_local_development", lambda: False)
    message = msgs.missing_github_client_id_message()
    assert "report this" in message.lower()
    assert "github.com/potpie-ai/potpie/issues" in message


def test_missing_linear_message_for_local_dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(msgs, "looks_like_local_development", lambda: True)
    message = msgs.missing_linear_client_id_message()
    assert "LINEAR_CLIENT_ID" in message
    assert ".env.template" in message


def test_missing_github_message_for_local_dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(msgs, "looks_like_local_development", lambda: True)
    message = msgs.missing_github_client_id_message()
    assert "POTPIE_GITHUB_CLIENT_ID" in message
    assert ".env.template" in message
