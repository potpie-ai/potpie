"""Unit tests for Potpie CLI browser auth helpers."""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import main as cli_main
from adapters.inbound.cli.auth import potpie as potpie_auth
from adapters.inbound.cli.auth.firebase_session import FirebaseSession
from adapters.inbound.cli.credentials_store import (
    clear_potpie_auth,
    get_potpie_firebase_api_key,
    get_potpie_firebase_refresh_token,
    get_stored_api_key,
    read_credentials,
    store_potpie_api_key,
    store_potpie_firebase_refresh_token,
)


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    import adapters.inbound.cli.credentials_store as cs

    store: dict[tuple[str, str], str] = {}

    def _set_password(service: str, username: str, password: str) -> None:
        store[(service, username)] = password

    def _get_password(service: str, username: str) -> str | None:
        return store.get((service, username))

    def _delete_password(service: str, username: str) -> None:
        store.pop((service, username), None)

    monkeypatch.setattr(cs.keyring, "set_password", _set_password)
    monkeypatch.setattr(cs.keyring, "get_password", _get_password)
    monkeypatch.setattr(cs.keyring, "delete_password", _delete_password)
    return store


def test_build_cli_success_url() -> None:
    url = potpie_auth.build_cli_success_url(provider="github")
    assert url.endswith("/cli-success?provider=github")
    assert url.startswith("http")


def test_resolve_potpie_ui_url_uses_cli_default_env(monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_CLI_UI_BASE_URL", "https://stage.potpie.ai/")
    monkeypatch.delenv("POTPIE_UI_URL", raising=False)

    assert potpie_auth.resolve_potpie_ui_url() == "https://stage.potpie.ai"


def test_parse_custom_token_body_accepts_jwt_like_token() -> None:
    result = potpie_auth._parse_custom_token_body(
        json.dumps(
            {
                "custom_token": "header.payload.signature",
                "firebase_api_key": "firebase-key",
                "state": "expected-state",
            }
        ).encode(),
        expected_state="expected-state",
    )
    assert result.custom_token == "header.payload.signature"
    assert result.firebase_api_key == "firebase-key"


def test_parse_custom_token_body_rejects_invalid() -> None:
    with pytest.raises(potpie_auth.PotpieCliAuthError):
        potpie_auth._parse_custom_token_body(
            b'{"custom_token":"not-a-jwt","state":"expected-state"}',
            expected_state="expected-state",
        )


def _start_callback_server(
    host: str,
    port: int,
    *,
    path: str = "/callback-random",
    state: str = "expected-state",
    allowed_origin: str = "https://potpie.example.com",
) -> tuple[potpie_auth._OneShotCallbackServer, threading.Thread]:
    server = potpie_auth._OneShotCallbackServer(
        (host, port),
        expected_state=state,
        expected_path=path,
        allowed_origin=allowed_origin,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_callback_server(
    server: potpie_auth._OneShotCallbackServer,
    thread: threading.Thread,
) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


def test_callback_server_allows_cors_preflight() -> None:
    host = "127.0.0.1"
    port = potpie_auth.pick_callback_port()
    server, thread = _start_callback_server(host, port)
    try:
        response = httpx.options(
            f"http://{host}:{port}/callback-random",
            headers={
                "Origin": "https://potpie.example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type, authorization",
            },
            timeout=5.0,
        )
        assert response.status_code == 204
        assert (
            response.headers.get("access-control-allow-origin")
            == "https://potpie.example.com"
        )
        assert response.headers.get("access-control-allow-private-network") == "true"
    finally:
        _stop_callback_server(server, thread)


def test_wait_for_cli_callback_accepts_post() -> None:
    host = "127.0.0.1"
    port = potpie_auth.pick_callback_port()
    path = "/callback-random"
    state = "expected-state"

    def _post_later() -> None:
        time.sleep(0.15)
        httpx.post(
            f"http://{host}:{port}{path}",
            json={
                "custom_token": "header.payload.signature",
                "firebase_api_key": "firebase-key",
                "state": state,
            },
            headers={"Origin": "https://potpie.example.com"},
            timeout=5.0,
        )

    threading.Thread(target=_post_later, daemon=True).start()
    result = potpie_auth.wait_for_cli_callback(
        host=host,
        port=port,
        path=path,
        state=state,
        allowed_origin="https://potpie.example.com",
        timeout_seconds=5.0,
    )
    assert result.custom_token == "header.payload.signature"
    assert result.firebase_api_key == "firebase-key"


def test_callback_server_rejects_wrong_state() -> None:
    host = "127.0.0.1"
    port = potpie_auth.pick_callback_port()
    server, thread = _start_callback_server(host, port)
    try:
        response = httpx.post(
            f"http://{host}:{port}/callback-random",
            json={
                "custom_token": "header.payload.signature",
                "firebase_api_key": "firebase-key",
                "state": "wrong-state",
            },
            timeout=5.0,
        )
        assert response.status_code == 400
        assert not server.received.is_set()
    finally:
        _stop_callback_server(server, thread)


def test_callback_server_rejects_missing_state() -> None:
    host = "127.0.0.1"
    port = potpie_auth.pick_callback_port()
    server, thread = _start_callback_server(host, port)
    try:
        response = httpx.post(
            f"http://{host}:{port}/callback-random",
            json={
                "custom_token": "header.payload.signature",
                "firebase_api_key": "firebase-key",
            },
            timeout=5.0,
        )
        assert response.status_code == 400
        assert not server.received.is_set()
    finally:
        _stop_callback_server(server, thread)


def test_callback_server_rejects_wrong_path() -> None:
    host = "127.0.0.1"
    port = potpie_auth.pick_callback_port()
    server, thread = _start_callback_server(host, port)
    try:
        response = httpx.post(
            f"http://{host}:{port}/wrong-path",
            json={
                "custom_token": "header.payload.signature",
                "firebase_api_key": "firebase-key",
                "state": "expected-state",
            },
            timeout=5.0,
        )
        assert response.status_code == 404
        assert not server.received.is_set()
    finally:
        _stop_callback_server(server, thread)


def test_store_potpie_api_key_metadata_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path, fake_keyring: dict
) -> None:
    import adapters.inbound.cli.credentials_store as cs

    monkeypatch.setattr(cs, "config_dir", lambda: tmp_path / "potpie")
    cs.write_credentials(api_key="legacy-file-key", api_base_url="http://localhost:8000")
    cs.write_provider_credentials(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "access_token": "gh-token",
            "token_type": "bearer",
            "scopes": ["repo"],
            "account": {"login": "octo"},
            "created_at": "2020-01-01T00:00:00+00:00",
            "updated_at": "2020-01-01T00:00:00+00:00",
            "expires_at": None,
            "metadata": {},
            "token_storage": "keychain",
        },
    )

    store_potpie_api_key("sk-chain-key", created_at="2026-05-29T12:00:00+00:00")
    assert fake_keyring[("potpie", "potpie_api_key")] == "sk-chain-key"
    assert get_stored_api_key() == "sk-chain-key"

    data = read_credentials()
    assert data.get("api_key") == "legacy-file-key"
    assert data["api_base_url"] == "http://localhost:8000"
    potpie_meta = data["integrations"]["potpie"]
    assert potpie_meta["auth_type"] == "api_key"
    assert potpie_meta["token_storage"] == "keychain"
    assert "sk-chain-key" not in json.dumps(data)

    clear_potpie_auth(clear_api_key=True)
    assert get_stored_api_key() == "legacy-file-key"
    assert "potpie" not in (read_credentials().get("integrations") or {})


def test_store_potpie_firebase_refresh_token_metadata_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path, fake_keyring: dict
) -> None:
    import adapters.inbound.cli.credentials_store as cs

    monkeypatch.setattr(cs, "config_dir", lambda: tmp_path / "potpie")
    cs.write_credentials(api_key="legacy-file-key", api_base_url="http://localhost:8000")

    store_potpie_firebase_refresh_token(
        "refresh-token",
        created_at="2026-05-29T12:00:00+00:00",
        firebase_api_key="firebase-key",
    )
    assert fake_keyring[("potpie", "potpie_firebase_refresh_token")] == "refresh-token"
    assert fake_keyring[("potpie", "potpie_firebase_api_key")] == "firebase-key"
    assert get_potpie_firebase_refresh_token() == "refresh-token"
    assert get_potpie_firebase_api_key() == "firebase-key"

    data = read_credentials()
    assert data.get("api_key") == "legacy-file-key"
    potpie_meta = data["integrations"]["potpie"]
    assert potpie_meta["auth_type"] == "firebase_session"
    assert potpie_meta["token_storage"] == "keychain"
    assert "firebase_api_key" not in potpie_meta
    assert "refresh-token" not in json.dumps(data)
    assert "firebase-key" not in json.dumps(data)


def test_clear_potpie_auth_preserves_api_key_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path, fake_keyring: dict
) -> None:
    import adapters.inbound.cli.credentials_store as cs

    monkeypatch.setattr(cs, "config_dir", lambda: tmp_path / "potpie")
    store_potpie_api_key("sk-chain-key", created_at="2026-05-29T12:00:00+00:00")
    store_potpie_firebase_refresh_token(
        "refresh-token", created_at="2026-05-29T12:01:00+00:00"
    )

    clear_potpie_auth()
    assert get_stored_api_key() == "sk-chain-key"
    assert get_potpie_firebase_refresh_token() == ""


def test_clear_provider_credentials_github_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path, fake_keyring: dict
) -> None:
    import adapters.inbound.cli.credentials_store as cs

    monkeypatch.setattr(cs, "config_dir", lambda: tmp_path / "potpie")
    store_potpie_firebase_refresh_token(
        "refresh-token", created_at="2026-05-29T12:00:00+00:00"
    )
    cs.write_provider_credentials(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "access_token": "gh-token",
            "token_storage": "keychain",
        },
    )
    cs.clear_provider_credentials("github")
    assert get_potpie_firebase_refresh_token() == "refresh-token"
    assert cs.get_provider_credentials("github") == {}


_runner = CliRunner()


def test_potpie_login_command_stores_firebase_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path, fake_keyring: dict
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        cli_main,
        "run_browser_login_flow",
        lambda: potpie_auth.CliCallbackResult(
            custom_token="header.payload.signature",
            firebase_api_key="firebase-key",
        ),
    )
    monkeypatch.setattr(
        cli_main,
        "exchange_custom_token",
        lambda token, *, firebase_api_key=None: FirebaseSession(
            id_token="id-token",
            refresh_token="refresh-token",
            expires_at=9999999999.0,
        ),
    )
    result = _runner.invoke(cli_main.app, ["login"])
    assert result.exit_code == 0, result.stdout
    assert get_potpie_firebase_refresh_token() == "refresh-token"
    assert "Logged in to Potpie successfully" in result.stdout

    alias = _runner.invoke(cli_main.app, ["auth", "potpie-login"])
    assert alias.exit_code == 0, alias.stdout


def test_potpie_logout_command_clears_firebase_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path, fake_keyring: dict
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    store_potpie_firebase_refresh_token(
        "refresh-token", created_at="2026-05-29T12:00:00+00:00"
    )
    result = _runner.invoke(cli_main.app, ["logout"])
    assert result.exit_code == 0, result.stdout
    assert get_potpie_firebase_refresh_token() == ""
    assert "Logged out of Potpie" in result.stdout
