"""Linear auth command, storage, profile, and provider configuration cases."""

from __future__ import annotations
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner
from potpie.cli.auth import auth_commands
from potpie.cli import main as cli_main
from unittest.mock import MagicMock, patch
import json
from potpie.auth.adapters import credentials_store as cs
from potpie.auth.adapters.integration_profile import (
    build_linear_integration_record,
    fetch_linear_viewer,
)
from potpie.auth.adapters.http import AuthHttpError
from potpie.auth.adapters import provider_config
from potpie.auth.adapters.integration_verify import (
    _verify_linear,
    verify_integration_access,
)
from potpie.auth.adapters.provider_config import (
    DEFAULT_CALLBACK_PORT,
    DEFAULT_FALLBACK_CALLBACK_PORTS,
    LINEAR_TOKEN_URL,
    authorization_url,
    get_callback_host,
    get_callback_path,
    get_callback_port,
    get_callback_port_candidates,
    get_client_id,
    get_client_secret,
    get_redirect_uri,
    get_redirect_uri_for_port,
    get_scopes,
    token_url,
)
# --- test_auth_commands.py ---

runner = CliRunner()


def test_auth_status_routes_to_integration_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[bool] = []

    def _integration_status(*, verify: bool = False) -> None:
        called.append(verify)

    monkeypatch.setattr(
        auth_commands,
        "integration_status",
        _integration_status,
    )

    result = runner.invoke(cli_main.app, ["auth", "status"])
    assert result.exit_code == 0, result.stdout
    assert called == [False]

    result = runner.invoke(cli_main.app, ["auth", "status", "--verify"])
    assert result.exit_code == 0, result.stdout
    assert called == [False, True]


def test_auth_status_json_keeps_provider_errors_per_row(
    auth_json_cli: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _status(provider: str) -> dict[str, object]:
        if provider == "linear":
            raise cs.ProviderCredentialError("keychain locked")
        return {
            "provider": provider,
            "authenticated": False,
            "auth_type": "oauth",
        }

    monkeypatch.setattr(auth_commands, "get_integration_status", _status)

    result = runner.invoke(auth_commands.auth_app, ["status"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    linear = next(row for row in payload["integrations"] if row["provider"] == "linear")
    assert linear["authenticated"] is False
    assert linear["status_error"] == "keychain locked"


def test_token_is_expired_invalid_expires_at() -> None:
    assert auth_commands._token_is_expired("not-a-number") is False
    assert auth_commands._token_is_expired(0.0) is True


def test_auth_logout_not_authenticated(
    auth_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auth_commands,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )

    result = runner.invoke(auth_commands.auth_app, ["logout", "linear"])

    assert result.exit_code == 1
    assert captured
    assert "not authenticated" in captured[0][0]


# --- test_auth_commands_cli.py ---


def test_auth_status_verify_token_error(
    auth_json_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": True,
            "auth_type": "oauth",
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _p: (_ for _ in ()).throw(RuntimeError("refresh broke")),
    )
    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])
    assert result.exit_code == 0
    assert '"verified": false' in result.stdout
    assert "refresh broke" in result.stdout


def test_wait_for_callback_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    from potpie.auth.adapters.callback_server import (
        OAuthCallbackResult,
    )

    monkeypatch.setattr(
        auth_commands,
        "wait_for_oauth_callback",
        lambda **kwargs: OAuthCallbackResult(code="c", state="s"),
    )
    result = auth_commands._wait_for_callback(
        host="localhost", port=8080, path="/callback"
    )
    assert result.code == "c"


def test_linear_login_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[tuple[bool, bool]] = []
    monkeypatch.setattr(
        auth_commands,
        "_run_linear_oauth_flow",
        lambda force=False, add=False: called.append((force, add)),
    )
    runner.invoke(auth_commands.linear_app, ["login", "--force"])
    assert called == [(True, False)]


# --- test_auth_commands_helpers.py ---


def test_build_linear_authorization_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(auth_commands, "get_scopes", lambda _p: "read")
    url = auth_commands._build_linear_authorization_url(
        redirect_uri="http://localhost:8080/callback",
        state="state-1",
        code_challenge="challenge",
    )
    assert "client_id=client-id" in url
    assert "code_challenge=challenge" in url
    assert "linear.app/oauth/authorize" in url


def test_token_is_expired_helper() -> None:
    import time as time_module

    assert auth_commands._token_is_expired(time_module.time() + 3600) is False
    assert auth_commands._token_is_expired(time_module.time() - 10) is True
    assert auth_commands._token_is_expired(None) is False


def test_try_refresh_linear_session_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _p: {"access_token": "tok", "expires_at": 9999999999.0},
    )
    assert auth_commands._try_refresh_linear_session() is True


def test_try_refresh_linear_session_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _p: (_ for _ in ()).throw(RuntimeError("refresh failed")),
    )
    assert auth_commands._try_refresh_linear_session() is False


def test_handle_already_connected_json(
    auth_json_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    printed: list[dict] = []
    monkeypatch.setattr(
        auth_commands,
        "print_plain_line",
        lambda *args, **kwargs: printed.append(kwargs.get("json_payload") or {}),
    )
    auth_commands._handle_already_connected(
        "linear",
        {"auth_type": "oauth", "expires_at": 1.0, "cloud_id": "c1"},
    )
    assert printed[-1].get("already_connected") is True


# --- test_auth_commands_status.py ---


def test_auth_status_includes_github_authenticated(
    auth_cli: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "github",
            "login": "octocat",
            "email": "a@b.com",
            "auth_type": "oauth",
        },
    )
    result = runner.invoke(auth_commands.auth_app, ["status"])

    assert result.exit_code == 0
    assert "github: authenticated" in result.stdout
    assert "login=octocat" in result.stdout


def test_linear_ls_lists_workspaces(
    auth_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "fetch_linear_workspaces",
        lambda limit=50: [
            {
                "key": "potpie-ai-cli",
                "name": "Potpie AI CLI",
                "type": "workspace",
                "active": True,
            },
        ],
    )

    result = runner.invoke(auth_commands.auth_app, ["linear", "ls"])

    assert result.exit_code == 0
    assert "Potpie AI CLI" in result.stdout
    assert "potpie linear login --add" in result.stdout


def test_linear_select_fetches_issues(
    auth_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "run_linear_use_flow",
        lambda org_key=None, team_key=None, limit=10: {
            "product": "linear",
            "workspace_key": "potpie-ai-cli",
            "workspace_name": "Potpie AI CLI",
            "team_key": "ENG",
            "team_name": "Engineering",
            "items": [
                {
                    "identifier": "ENG-1",
                    "title": "Fix login",
                    "status": "In Progress",
                    "url": "https://linear.app/issue/ENG-1",
                }
            ],
        },
    )

    result = runner.invoke(
        auth_commands.auth_app,
        ["linear", "select", "--org", "potpie-ai-cli", "--key", "ENG"],
    )

    assert result.exit_code == 0
    assert "ENG-1" in result.stdout
    assert "Fix login" in result.stdout
    assert "Potpie AI CLI" in result.stdout


def test_auth_status_verify_linear(
    auth_json_cli: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": True,
            "auth_type": "oauth",
            "expires_at": 9999999999.0,
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _provider: {"access_token": "tok", "expires_at": 9999999999.0},
    )
    monkeypatch.setattr(
        auth_commands,
        "verify_integration_access",
        lambda _provider, _creds: (True, "ok (Ada)"),
    )

    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])

    assert result.exit_code == 0, result.stdout
    assert '"verified": true' in result.stdout


def test_auth_status_verify_linear_refresh_error_stays_in_row(
    auth_json_cli: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linear_status_calls = 0

    def _status(provider: str) -> dict[str, object]:
        nonlocal linear_status_calls
        if provider != "linear":
            return {
                "provider": provider,
                "authenticated": False,
                "auth_type": "oauth",
            }
        linear_status_calls += 1
        if linear_status_calls == 2:
            raise cs.ProviderCredentialError("keychain locked")
        return {
            "provider": provider,
            "authenticated": True,
            "auth_type": "oauth",
            "expires_at": 9999999999.0,
        }

    monkeypatch.setattr(auth_commands, "get_integration_status", _status)
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _provider: {"access_token": "tok", "expires_at": 9999999999.0},
    )
    monkeypatch.setattr(
        auth_commands,
        "verify_integration_access",
        lambda _provider, _creds: (True, "ok (Ada)"),
    )

    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    linear = next(row for row in payload["integrations"] if row["provider"] == "linear")
    assert linear["verified"] is False
    assert linear["verify_message"] == "keychain locked"
    assert linear_status_calls == 2


def test_auth_status_human_verify_failed(
    auth_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "linear",
            "login": "Ada",
            "email": "ada@example.com",
            "site_name": "Acme",
            "expires_at": 12345.0,
            "token_storage": "file",
            "auth_type": "oauth",
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _p: {"access_token": "tok"},
    )
    monkeypatch.setattr(
        auth_commands,
        "verify_integration_access",
        lambda _p, _c: (False, "Linear API request failed"),
    )

    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])

    assert result.exit_code == 0
    assert "verify failed" in result.stdout
    assert "Linear API request failed" in result.stdout.replace("\n", "")
    assert "expires_at=" in result.stdout


def test_auth_status_human_verify_ok(
    auth_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "linear",
            "login": "Ada",
            "auth_type": "oauth",
            "expires_at": 9999999999.0,
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _p: {"access_token": "tok"},
    )
    monkeypatch.setattr(
        auth_commands,
        "verify_integration_access",
        lambda _p, _c: (True, "ok (Ada)"),
    )

    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])

    assert result.exit_code == 0
    assert "verify=ok (Ada)" in result.stdout


# --- test_credentials_store_integrations.py ---


def _linear_viewer_stub(**org_overrides: object) -> dict[str, object]:
    organization = {
        "id": "org-1",
        "name": "Potpie AI CLI",
        "url_key": "potpie-ai-cli",
        **org_overrides,
    }
    return {
        "account": {"name": "Ada", "email": "ada@example.com"},
        "organization": organization,
    }


def test_linear_integration_tokens_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "potpie.auth.adapters.integration_profile.fetch_linear_viewer",
        lambda _token: _linear_viewer_stub(),
    )
    cs.save_integration_tokens(
        "linear",
        {
            "access_token": "access",
            "refresh_token": "refresh",
            "scope": "read",
            "expires_at": 9999999999.0,
        },
    )
    tokens = cs.get_integration_tokens("linear")
    assert tokens["access_token"] == "access"
    assert tokens["refresh_token"] == "refresh"

    cs.clear_integration_tokens("linear")
    assert cs.get_integration_tokens("linear") == {}


def test_list_integration_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "potpie.auth.adapters.integration_profile.fetch_linear_viewer",
        lambda _token: _linear_viewer_stub(),
    )
    cs.save_integration_tokens(
        "linear",
        {"access_token": "tok", "expires_at": 9999999999.0},
    )
    providers = cs.list_integration_providers()
    assert "linear" in providers


def test_clear_integration_tokens_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "potpie.auth.adapters.integration_profile.fetch_linear_viewer",
        lambda _token: _linear_viewer_stub(),
    )
    cs.save_integration_tokens(
        "linear",
        {"access_token": "a", "refresh_token": "r", "expires_at": 9999999999.0},
    )
    cs.clear_integration_tokens("linear")
    assert cs.get_integration_tokens("linear") == {}
    status = cs.get_integration_status("linear")
    assert status["authenticated"] is False


def test_linear_status_includes_org_and_scope_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "potpie.auth.adapters.integration_profile.fetch_linear_viewer",
        lambda _token: _linear_viewer_stub(name="Potpie", url_key="potpie"),
    )
    cs.save_integration_tokens(
        "linear",
        {
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_at": 9999999999.0,
            "scope": "read,write",
            "account": {"name": "Ada", "email": "ada@example.com"},
            "organization": {"name": "Potpie"},
        },
    )
    status = cs.get_integration_status("linear")
    assert status["authenticated"] is True
    assert status["site_name"] == "Potpie"
    assert status["scope"] in (["read", "write"], "read,write")


# --- test_integration_profile.py ---


def test_build_linear_integration_record_includes_account_and_scopes() -> None:
    tokens = {
        "access_token": "lin-token",
        "refresh_token": "lin-refresh",
        "scope": "read,write",
        "expires_at": 999.0,
        "token_type": "Bearer",
    }
    viewer = {
        "account": {"id": "u1", "name": "Ada", "email": "ada@example.com"},
        "organization": {"id": "o1", "name": "Acme"},
    }
    with patch(
        "potpie.auth.adapters.integration_profile.fetch_linear_viewer",
        return_value=viewer,
    ):
        record = build_linear_integration_record(tokens)
    assert record["account"]["email"] == "ada@example.com"
    assert record["organization"]["name"] == "Acme"
    assert record["scopes"] == ["read", "write"]
    assert record["metadata"]["auth_flow"] == "pkce"
    assert record["created_at"]
    assert record["updated_at"]


def test_build_linear_integration_record_preserves_account_on_refresh() -> None:
    prior = {
        "account": {"id": "u1", "name": "Ada", "email": "ada@example.com"},
        "created_at": "2020-01-01T00:00:00+00:00",
    }
    tokens = {"access_token": "new-token", "expires_at": 123.0}
    with patch(
        "potpie.auth.adapters.integration_profile.fetch_linear_viewer",
        return_value={},
    ):
        record = build_linear_integration_record(tokens, existing=prior)
    assert record["account"]["email"] == "ada@example.com"
    assert record["created_at"] == "2020-01-01T00:00:00+00:00"


def test_save_integration_tokens_writes_account_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cs, "_store_file_secret", lambda *a, **k: None)
    viewer = {
        "account": {"id": "u1", "name": "Ada", "email": "ada@example.com"},
        "organization": {"id": "org-acme", "name": "Acme", "url_key": "acme"},
    }
    with patch(
        "potpie.auth.adapters.integration_profile.fetch_linear_viewer",
        return_value=viewer,
    ):
        cs.save_integration_tokens(
            "linear",
            {"access_token": "tok", "scope": "read", "expires_at": 1.0},
        )
    payload = cs.read_credentials()
    linear = payload["integrations"]["linear"]
    assert linear["account"]["email"] == "ada@example.com"
    assert linear["organization"]["name"] == "Acme"
    assert linear["organizations"]["org-acme"]["name"] == "Acme"


def test_get_integration_status_linear_requires_file_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cs, "_load_file_secret", lambda *_a, **_k: "")
    cs._write_payload(
        {
            "integrations": {
                "linear": {
                    "provider": "linear",
                    "auth_type": "oauth",
                    "account": {"email": "ada@example.com", "name": "Ada"},
                    "organization": {"name": "Acme"},
                }
            }
        }
    )
    status = cs.get_integration_status("linear")
    assert status["authenticated"] is False
    assert status["auth_type"] == "oauth"


def test_fetch_linear_viewer_non_json_body_returns_empty() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.side_effect = ValueError("not json")
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    assert fetch_linear_viewer("token", http=client) == {}


def test_fetch_linear_viewer_non_200_returns_empty() -> None:
    response = MagicMock()
    response.status_code = 500
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    assert fetch_linear_viewer("token", http=client) == {}


def test_fetch_linear_viewer_empty_token_returns_empty() -> None:
    assert fetch_linear_viewer("   ") == {}


def test_fetch_linear_viewer_non_dict_viewer_returns_empty() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"data": {"viewer": "not-a-dict"}}
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    assert fetch_linear_viewer("token", http=client) == {}


def test_build_linear_integration_record_normalizes_scope_list() -> None:
    record = build_linear_integration_record(
        {
            "access_token": "a",
            "scope": ["read", "write"],
            "account": {"name": "Ada"},
        }
    )
    assert record["scopes"] == ["read", "write"]


def test_fetch_linear_viewer_parses_graphql_response() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": {
            "viewer": {
                "id": "v1",
                "name": "Nihit",
                "email": "n@example.com",
                "organization": {"id": "org1", "name": "Potpie"},
            }
        }
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    profile = fetch_linear_viewer("token", http=client)
    assert profile["account"]["name"] == "Nihit"
    assert profile["organization"]["name"] == "Potpie"


# --- test_integration_verify.py ---


def test_verify_linear_transport_error_returns_false() -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.side_effect = AuthHttpError("offline")
    ok, message = _verify_linear("token", http=client)
    assert ok is False
    assert message == "Linear API request failed"


def test_verify_linear_non_json_body_returns_false() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.side_effect = ValueError("not json")
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    ok, message = _verify_linear("token", http=client)
    assert ok is False
    assert "non-JSON" in message


def test_verify_integration_access_linear_no_token() -> None:
    ok, message = verify_integration_access("linear", {})
    assert ok is False
    assert message == "not authenticated"


def test_verify_linear_success_without_org() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": {"viewer": {"id": "v1", "name": "Ada", "email": "a@example.com"}}
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    ok, message = _verify_linear("token", http=client)
    assert ok is True
    assert "Ada" in message


def test_verify_integration_access_linear_expired() -> None:
    ok, message = verify_integration_access(
        "linear",
        {"access_token": "tok", "expires_at": 1.0},
    )
    assert ok is False
    assert "expired" in message


def test_verify_linear_success() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": {
            "viewer": {
                "id": "v1",
                "name": "Ada",
                "email": "a@example.com",
                "organization": {"name": "Acme"},
            }
        }
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    ok, message = _verify_linear("token", http=client)
    assert ok is True
    assert "Ada" in message
    assert "Acme" in message


def test_verify_integration_access_linear_ignores_invalid_expires_at() -> None:
    with patch(
        "potpie.auth.adapters.integration_verify._verify_linear",
        return_value=(True, "ok (Ada)"),
    ) as verify:
        ok, message = verify_integration_access(
            "linear",
            {"access_token": "tok", "expires_at": "not-a-number"},
        )
    verify.assert_called_once_with("tok", http=None)
    assert ok is True
    assert message == "ok (Ada)"


def test_verify_linear_graphql_errors() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"errors": [{"message": "bad token"}]}
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    ok, message = _verify_linear("token", http=client)
    assert ok is False
    assert "rejected" in message


def test_verify_linear_non_200_status() -> None:
    response = MagicMock()
    response.status_code = 401
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    ok, message = _verify_linear("token", http=client)
    assert ok is False
    assert "HTTP 401" in message


# --- test_provider_config.py ---


def test_get_client_id_uses_configured_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LINEAR_CLIENT_ID", raising=False)
    expected = "configured-linear-client-id"
    if hasattr(provider_config, "PACKAGE_LINEAR_CLIENT_ID"):
        monkeypatch.setattr(provider_config, "PACKAGE_LINEAR_CLIENT_ID", expected)
    else:
        monkeypatch.setattr(
            provider_config,
            "load_runtime_settings",
            lambda: SimpleNamespace(linear_client_id=expected),
        )
    assert get_client_id("linear") == expected


def test_get_client_id_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_CLIENT_ID", "override-client-id")
    assert get_client_id("linear") == "override-client-id"


def test_redirect_and_callback_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", raising=False)
    monkeypatch.setenv(
        "POTPIE_CLI_OAUTH_REDIRECT_URI",
        "http://127.0.0.1:9001/custom/callback",
    )
    assert get_redirect_uri() == "http://127.0.0.1:9001/custom/callback"
    assert get_callback_host() == "127.0.0.1"
    assert get_callback_path() == "/custom/callback"
    assert get_callback_port() == 9001
    assert get_callback_port_candidates() == (9001,)


def test_default_callback_port_is_five_digit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POTPIE_CLI_OAUTH_REDIRECT_URI", raising=False)
    monkeypatch.delenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", raising=False)
    assert DEFAULT_CALLBACK_PORT == 28757
    assert DEFAULT_FALLBACK_CALLBACK_PORTS == (28763,)
    assert get_callback_port() == DEFAULT_CALLBACK_PORT
    assert get_callback_port_candidates() == (
        DEFAULT_CALLBACK_PORT,
        *DEFAULT_FALLBACK_CALLBACK_PORTS,
    )


def test_get_callback_port_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", raising=False)
    monkeypatch.setenv(
        "POTPIE_CLI_OAUTH_REDIRECT_URI", "http://localhost:8080/callback"
    )
    monkeypatch.setenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", "9999")
    assert get_callback_port() == 9999
    assert get_callback_port_candidates() == (9999,)


def test_get_redirect_uri_for_port_rewrites_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "POTPIE_CLI_OAUTH_REDIRECT_URI",
        "http://127.0.0.1:9001/custom/callback",
    )
    assert get_redirect_uri_for_port(9002) == "http://127.0.0.1:9002/custom/callback"


def test_linear_oauth_urls() -> None:
    assert authorization_url("linear") == "https://linear.app/oauth/authorize"
    assert token_url("linear") == LINEAR_TOKEN_URL


def test_get_scopes_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LINEAR_OAUTH_SCOPE", raising=False)
    assert get_scopes("linear") == "read"
    monkeypatch.setenv("LINEAR_OAUTH_SCOPE", "read,write")
    assert get_scopes("linear") == "read,write"


def test_invalid_redirect_uri_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", raising=False)
    monkeypatch.setenv("POTPIE_CLI_OAUTH_REDIRECT_URI", "https://example.com/cb")
    with pytest.raises(ValueError, match="localhost"):
        get_callback_port()


def test_get_client_secret_linear_returns_empty() -> None:
    assert get_client_secret("linear") == ""


def test_get_callback_port_invalid_env_falls_back_to_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", raising=False)
    monkeypatch.setenv(
        "POTPIE_CLI_OAUTH_REDIRECT_URI", "http://localhost:8080/callback"
    )
    monkeypatch.setenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", "not-a-port")
    assert get_callback_port() == 8080


def test_get_callback_port_out_of_range_falls_back_to_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "POTPIE_CLI_OAUTH_REDIRECT_URI", "http://localhost:8080/callback"
    )
    monkeypatch.setenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", "70000")
    assert get_callback_port() == 8080
    monkeypatch.setenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", "0")
    assert get_callback_port() == 8080


def test_redirect_uri_hostname_not_localhost_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "POTPIE_CLI_OAUTH_REDIRECT_URI",
        "http://evil.example.com:8080/callback",
    )
    with pytest.raises(ValueError, match="localhost"):
        get_callback_host()


def test_unsupported_oauth_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported OAuth provider"):
        authorization_url("github")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported OAuth provider"):
        token_url("github")  # type: ignore[arg-type]
