"""Unit tests for Potpie CLI auth config resolution."""

from __future__ import annotations

from adapters.outbound.cli_auth import potpie_api_config as config
from adapters.outbound.cli_auth.firebase_session import FirebaseSession


def test_resolve_api_base_url_uses_canonical_env(monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_ENVIRONMENT", "test")
    monkeypatch.setenv("POTPIE_API_URL", "https://stage-api.potpie.ai/")
    monkeypatch.setattr(config, "get_stored_api_base_url", lambda: "")

    assert config.resolve_potpie_api_base_url() == "https://stage-api.potpie.ai"


def test_resolve_api_base_url_uses_code_default_and_ignores_old_aliases(
    monkeypatch,
) -> None:
    monkeypatch.setenv("POTPIE_ENVIRONMENT", "test")
    for key in (
        "POTPIE_API_URL",
        "POTPIE_BASE_URL",
        "POTPIE_CLI_API_BASE_URL",
        "POTPIE_CLI_BASE_URL",
        "POTPIE_PORT",
        "POTPIE_API_PORT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config, "get_stored_api_base_url", lambda: "")

    assert config.resolve_potpie_api_base_url() == "http://localhost:8001"

    monkeypatch.setenv("POTPIE_CLI_API_BASE_URL", "https://old-api.potpie.ai")
    monkeypatch.setenv("POTPIE_API_PORT", "8123")
    assert config.resolve_potpie_api_base_url() == "http://localhost:8001"


def test_resolve_api_base_url_uses_stored_url_before_code_default(monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_ENVIRONMENT", "test")
    monkeypatch.delenv("POTPIE_API_URL", raising=False)
    monkeypatch.setattr(
        config, "get_stored_api_base_url", lambda: "https://stored-api.potpie.ai/"
    )

    assert config.resolve_potpie_api_base_url() == "https://stored-api.potpie.ai"


def test_resolve_api_base_url_env_wins_over_stored_url(monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_ENVIRONMENT", "test")
    monkeypatch.setenv("POTPIE_API_URL", "https://runtime-api.potpie.ai/")
    monkeypatch.setattr(config, "get_stored_api_base_url", lambda: "")

    assert config.resolve_potpie_api_base_url() == "https://runtime-api.potpie.ai"


def test_resolve_potpie_api_key_errors_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("POTPIE_API_KEY", raising=False)
    monkeypatch.setattr(config, "get_stored_api_key", lambda: "")

    try:
        config.resolve_potpie_api_key()
    except ValueError as exc:
        assert "Potpie API key missing" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_resolve_auth_config_prefers_env_api_key(monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_API_KEY", "sk-env")
    monkeypatch.setattr(config, "get_potpie_auth_type", lambda: "potpie")
    monkeypatch.setattr(config, "get_potpie_firebase_refresh_token", lambda: "refresh")
    monkeypatch.setattr(config, "get_stored_api_key", lambda: "sk-stored")

    auth = config.resolve_potpie_auth_config()

    assert auth.mode == "api_key"
    assert auth.headers == {"X-API-Key": "sk-env"}


def test_resolve_auth_config_prefers_firebase_session_metadata(monkeypatch) -> None:
    monkeypatch.delenv("POTPIE_API_KEY", raising=False)
    monkeypatch.setattr(config, "get_potpie_auth_type", lambda: "potpie")
    monkeypatch.setattr(config, "get_stored_api_key", lambda: "sk-existing")
    monkeypatch.setattr(config, "get_potpie_firebase_refresh_token", lambda: "refresh")
    monkeypatch.setattr(config, "get_potpie_firebase_id_token", lambda: "")
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

    monkeypatch.setattr(config, "refresh_id_token", _refresh_id_token)
    updated: list[str] = []
    monkeypatch.setattr(config, "update_potpie_firebase_refresh_token", updated.append)
    cached_id_tokens: list[str] = []
    monkeypatch.setattr(
        config, "store_potpie_firebase_id_token", cached_id_tokens.append
    )

    auth = config.resolve_potpie_auth_config()

    assert auth.mode == "potpie"
    assert auth.headers == {"Authorization": "Bearer id-token"}
    assert refresh_calls == [("refresh", "firebase-key")]
    assert updated == ["new-refresh"]
    assert cached_id_tokens == ["id-token"]


def test_resolve_auth_config_uses_cached_valid_firebase_id_token(monkeypatch) -> None:
    monkeypatch.delenv("POTPIE_API_KEY", raising=False)
    monkeypatch.setattr(config, "get_potpie_auth_type", lambda: "potpie")
    monkeypatch.setattr(config, "get_stored_api_key", lambda: "sk-existing")
    monkeypatch.setattr(config, "get_potpie_firebase_refresh_token", lambda: "refresh")
    monkeypatch.setattr(
        config, "get_potpie_firebase_id_token", lambda: "cached-id-token"
    )
    monkeypatch.setattr(
        config, "id_token_expires_at", lambda token: config.time.time() + 60
    )
    refresh_calls: list[str] = []
    monkeypatch.setattr(
        config,
        "refresh_id_token",
        lambda token, *, firebase_api_key=None: refresh_calls.append(token),
    )

    auth = config.resolve_potpie_auth_config()

    assert auth.mode == "potpie"
    assert auth.headers == {"Authorization": "Bearer cached-id-token"}
    assert refresh_calls == []


def test_resolve_auth_config_falls_back_to_stored_api_key(monkeypatch) -> None:
    monkeypatch.delenv("POTPIE_API_KEY", raising=False)
    monkeypatch.setattr(config, "get_potpie_auth_type", lambda: "")
    monkeypatch.setattr(config, "get_potpie_firebase_refresh_token", lambda: "")
    monkeypatch.setattr(config, "get_stored_api_key", lambda: "sk-existing")

    auth = config.resolve_potpie_auth_config()

    assert auth.mode == "api_key"
    assert auth.headers == {"X-API-Key": "sk-existing"}


def test_resolve_auth_config_uses_refresh_token_without_metadata(monkeypatch) -> None:
    monkeypatch.delenv("POTPIE_API_KEY", raising=False)
    monkeypatch.setattr(config, "get_potpie_auth_type", lambda: "")
    monkeypatch.setattr(config, "get_stored_api_key", lambda: "")
    monkeypatch.setattr(config, "get_potpie_firebase_refresh_token", lambda: "refresh")
    monkeypatch.setattr(config, "get_potpie_firebase_id_token", lambda: "")
    monkeypatch.setattr(config, "get_potpie_firebase_api_key", lambda: "")
    monkeypatch.setattr(
        config,
        "refresh_id_token",
        lambda token, *, firebase_api_key=None: FirebaseSession(
            id_token="id-token",
            refresh_token="new-refresh",
            expires_at=123.0,
        ),
    )
    updated: list[str] = []
    monkeypatch.setattr(config, "update_potpie_firebase_refresh_token", updated.append)
    cached_id_tokens: list[str] = []
    monkeypatch.setattr(
        config, "store_potpie_firebase_id_token", cached_id_tokens.append
    )

    auth = config.resolve_potpie_auth_config()

    assert auth.mode == "potpie"
    assert auth.headers == {"Authorization": "Bearer id-token"}
    assert updated == ["new-refresh"]
    assert cached_id_tokens == ["id-token"]


def test_resolve_auth_config_errors_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("POTPIE_API_KEY", raising=False)
    monkeypatch.setattr(config, "get_potpie_auth_type", lambda: "")
    monkeypatch.setattr(config, "get_stored_api_key", lambda: "")
    monkeypatch.setattr(config, "get_potpie_firebase_refresh_token", lambda: "")

    try:
        config.resolve_potpie_auth_config()
    except ValueError as exc:
        assert "Potpie auth missing" in str(exc)
    else:
        raise AssertionError("expected ValueError")
