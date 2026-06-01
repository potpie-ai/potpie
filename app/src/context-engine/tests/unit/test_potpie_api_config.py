"""Unit tests for Potpie CLI auth config resolution."""

from __future__ import annotations

from adapters.inbound.cli import potpie_api_config as config
from adapters.inbound.cli.auth.firebase_session import FirebaseSession


def test_resolve_api_base_url_uses_cli_default_env(monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_CLI_API_BASE_URL", "https://stage-api.potpie.ai/")
    monkeypatch.delenv("POTPIE_API_URL", raising=False)
    monkeypatch.delenv("POTPIE_BASE_URL", raising=False)
    monkeypatch.setattr(config, "get_stored_api_base_url", lambda: "")

    assert config.resolve_potpie_api_base_url() == "https://stage-api.potpie.ai"


def test_resolve_auth_config_prefers_api_key(monkeypatch) -> None:
    monkeypatch.delenv("POTPIE_API_KEY", raising=False)
    monkeypatch.setattr(config, "get_stored_api_key", lambda: "sk-existing")
    monkeypatch.setattr(config, "get_potpie_firebase_refresh_token", lambda: "refresh")

    auth = config.resolve_potpie_auth_config()
    assert auth.mode == "api_key"
    assert auth.headers == {"X-API-Key": "sk-existing"}


def test_resolve_auth_config_uses_firebase_refresh_token(monkeypatch) -> None:
    monkeypatch.delenv("POTPIE_API_KEY", raising=False)
    monkeypatch.setattr(config, "get_stored_api_key", lambda: "")
    monkeypatch.setattr(config, "get_potpie_firebase_refresh_token", lambda: "refresh")
    monkeypatch.setattr(config, "get_potpie_firebase_api_key", lambda: "firebase-key")
    refresh_calls: list[tuple[str, str | None]] = []

    def _refresh_id_token(
        token: str, *, firebase_api_key: str | None = None
    ) -> FirebaseSession:
        refresh_calls.append((token, firebase_api_key))
        return FirebaseSession(
            id_token="id-token",
            refresh_token="new-refresh",
            expires_at=123.0,
        )

    monkeypatch.setattr(
        config,
        "refresh_id_token",
        _refresh_id_token,
    )
    updated: list[str] = []
    monkeypatch.setattr(config, "update_potpie_firebase_refresh_token", updated.append)

    auth = config.resolve_potpie_auth_config()
    assert auth.mode == "firebase_session"
    assert auth.headers == {"Authorization": "Bearer id-token"}
    assert refresh_calls == [("refresh", "firebase-key")]
    assert updated == ["new-refresh"]
