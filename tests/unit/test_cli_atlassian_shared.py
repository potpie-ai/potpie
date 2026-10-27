"""Atlassian auth command, storage, profile, and provider configuration cases."""

from __future__ import annotations
import pytest
from tests._auth_cli_fixtures import (
    auth_cli as auth_cli,
    auth_json_cli as auth_json_cli,
)
from typer.testing import CliRunner
from potpie.cli.auth import auth_commands
from potpie.cli.commands._common import set_store
from tests._auth_fakes import InMemoryCredentialStore
from potpie.auth.adapters import credentials_store as cs
from potpie.auth.adapters.integration_profile import (
    build_atlassian_integration_record,
    build_product_integration_record,
)
from potpie.cli.auth.atlassian_auth import (
    AtlassianAuthErrorKind,
    AtlassianVerifyResult,
)
from potpie.auth.adapters.integration_verify import (
    _verify_atlassian_product,
    _verify_message_for_kind,
    verify_integration_access,
)
from potpie.auth.adapters.provider_config import (
    ATLASSIAN_API_GATEWAY,
    atlassian_confluence_gateway_url,
    atlassian_jira_gateway_url,
)
# --- test_auth_commands.py ---

runner = CliRunner()


def test_auth_logout_not_authenticated_still_clears_stale_state(
    auth_cli: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    cleared: list[str] = []
    store = InMemoryCredentialStore()
    store.clear_integration_tokens = lambda provider: cleared.append(provider)  # type: ignore[method-assign]
    set_store(store)

    result = runner.invoke(auth_commands.auth_app, ["logout", "jira"])

    assert result.exit_code == 0
    assert cleared == ["jira"]
    assert "stale" in result.stdout.lower()


def test_auth_logout_wiki_alias(
    auth_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": True},
    )
    cleared: list[str] = []
    store = InMemoryCredentialStore()
    store.clear_integration_tokens = lambda provider: cleared.append(provider)  # type: ignore[method-assign]
    set_store(store)

    result = runner.invoke(auth_commands.auth_app, ["logout", "wiki"])

    assert result.exit_code == 0
    assert cleared == ["confluence"]


def test_auth_logout_jira(auth_json_cli: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _provider: {"authenticated": True},
    )
    cleared: list[str] = []
    store = InMemoryCredentialStore()
    store.clear_integration_tokens = lambda provider: cleared.append(provider)  # type: ignore[method-assign]
    set_store(store)

    result = runner.invoke(auth_commands.auth_app, ["logout", "jira"])

    assert result.exit_code == 0, result.stdout
    assert cleared == ["jira"]
    assert '"ok": true' in result.stdout


# --- test_auth_commands_cli.py ---


def test_print_wiki_row_pages_and_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(
        auth_commands,
        "_print_remote_line",
        lambda message: lines.append(message),
    )

    auth_commands._print_wiki_row(
        {
            "title": "Page [bold]",
            "space": "DOCS",
            "status": "current",
            "created": "2024-01-01",
            "created_by": "Ada",
            "updated": "2024-02-01",
            "updated_by": "Bob",
            "excerpt": "Hello",
            "url": "https://x.com/p",
        },
        pages=True,
    )
    assert any("Page" in line for line in lines)
    assert any("DOCS" in line for line in lines)


def test_run_product_use_result_human(
    auth_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines: list[str] = []
    monkeypatch.setattr(
        auth_commands, "print_plain_line", lambda m, **k: lines.append(m)
    )
    monkeypatch.setattr(auth_commands, "_print_remote_line", lambda m: lines.append(m))

    auth_commands._run_product_use_result(
        {
            "product": "confluence",
            "workspace_key": "DOCS",
            "workspace_name": "Documentation",
            "items": [{"title": "Intro", "space": "DOCS", "url": "https://x.com"}],
        },
        product_label="Confluence",
    )
    assert any("Documentation" in line for line in lines)
    assert any("Intro" in line for line in lines)


def test_auth_status_verify_token_error(
    auth_json_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": True,
            "auth_type": "api_token",
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "get_integration_tokens",
        lambda _p: (_ for _ in ()).throw(RuntimeError("token load broke")),
    )
    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])
    assert result.exit_code == 0
    assert '"verified": false' in result.stdout
    assert "token load broke" in result.stdout


# --- test_auth_commands_helpers.py ---


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
        "jira",
        {
            "auth_type": "api_token",
            "site_url": "https://team.atlassian.net",
            "site_name": "team",
        },
    )
    assert printed[-1].get("already_connected") is True


def test_run_product_use_result_json(
    auth_json_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    assert blobs[0]["provider"] == "jira"


def test_run_product_use_result_json_maps_wiki_to_confluence_provider(
    auth_json_cli: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs: list[dict] = []
    monkeypatch.setattr(
        auth_commands,
        "print_json_blob",
        lambda payload, **kwargs: blobs.append(payload),
    )

    auth_commands._run_product_use_result(
        {
            "product": "wiki",
            "workspace_key": "DOCS",
            "workspace_name": "Docs",
            "items": [],
        },
        product_label="Confluence",
    )

    assert blobs[0]["product"] == "wiki"
    assert blobs[0]["provider"] == "confluence"


# --- test_auth_commands_status.py ---


def test_jira_login_help() -> None:
    result = runner.invoke(auth_commands.auth_app, ["jira", "login", "--help"])
    assert result.exit_code == 0
    assert "api-token" in result.stdout.lower() or "token" in result.stdout.lower()


def test_auth_status_human_verify_failed(
    auth_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "jira",
            "login": "Ada",
            "email": "ada@example.com",
            "site_name": "Team",
            "site_url": "https://team.atlassian.net",
            "expires_at": 12345.0,
            "cloud_id": "cloud-1",
            "token_storage": "file",
            "auth_type": "api_token",
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "get_integration_tokens",
        lambda _p: {"api_token": "tok"},
    )
    monkeypatch.setattr(
        auth_commands,
        "verify_integration_access",
        lambda _p, _c: (False, "gateway down"),
    )

    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])

    # Normalize whitespace: the long status line soft-wraps at the console width.
    out = " ".join(result.stdout.split())
    assert result.exit_code == 0
    assert "verify failed" in out
    assert "gateway down" in out
    assert "expires_at=" in out
    assert "cloud_id=" in out


def test_auth_status_human_verify_ok(
    auth_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "jira",
            "login": "Ada",
            "email": "ada@example.com",
            "auth_type": "api_token",
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "get_integration_tokens",
        lambda _p: {
            "api_token": "tok",
            "email": "ada@example.com",
            "site_url": "https://x.net",
        },
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


def test_jira_credentials_roundtrip() -> None:
    cs.save_jira_credentials(
        {
            "email": "user@example.com",
            "api_token": "jira-token",
            "site_url": "https://team.atlassian.net",
            "cloud_id": "cloud-1",
        }
    )
    creds = cs.get_jira_credentials()
    assert creds["api_token"] == "jira-token"
    assert creds["email"] == "user@example.com"

    cs.clear_jira_credentials()
    assert cs.get_jira_credentials() == {}
    assert cs._JIRA_TOKEN_SECRET not in cs._read_integration_secrets_file()


def test_list_integration_providers() -> None:
    cs.save_jira_credentials(
        {
            "email": "u@example.com",
            "api_token": "tok",
            "site_url": "https://x.atlassian.net",
            "cloud_id": "c1",
        }
    )
    providers = cs.list_integration_providers()
    assert "jira" in providers


def test_get_atlassian_credentials_prefers_jira() -> None:
    cs.save_jira_credentials(
        {
            "email": "u@example.com",
            "api_token": "tok",
            "site_url": "https://x.atlassian.net",
            "cloud_id": "c1",
        }
    )
    creds = cs.get_atlassian_credentials()
    assert creds["api_token"] == "tok"


def test_get_integration_status_jira_unauthenticated() -> None:
    status = cs.get_integration_status("jira")
    assert status["authenticated"] is False


def test_confluence_credentials_roundtrip() -> None:
    cs.save_confluence_credentials(
        {
            "email": "wiki@example.com",
            "api_token": "wiki-token",
            "site_url": "https://team.atlassian.net",
            "cloud_id": "cloud-2",
        }
    )
    creds = cs.get_confluence_credentials()
    assert creds["api_token"] == "wiki-token"
    cs.clear_confluence_credentials()
    assert cs.get_confluence_credentials() == {}


def test_get_integration_tokens_jira() -> None:
    cs.save_jira_credentials(
        {
            "email": "u@example.com",
            "api_token": "tok",
            "site_url": "https://x.atlassian.net",
            "cloud_id": "c1",
        }
    )
    tokens = cs.get_integration_tokens("jira")
    assert tokens["auth_type"] == "api_token"
    assert tokens["api_token"] == "tok"


def test_get_integration_status_jira_authenticated() -> None:
    cs.save_jira_credentials(
        {
            "email": "u@example.com",
            "api_token": "tok",
            "site_url": "https://x.atlassian.net",
            "cloud_id": "c1",
            "site_name": "X",
        }
    )
    status = cs.get_integration_status("jira")
    assert status["authenticated"] is True
    assert status["email"] == "u@example.com"
    assert status["site_url"] == "https://x.atlassian.net"


def test_save_atlassian_credentials_legacy() -> None:
    cs.save_atlassian_credentials(
        {
            "email": "legacy@example.com",
            "api_token": "legacy-tok",
            "site_url": "https://legacy.atlassian.net",
            "cloud_id": "legacy-cloud",
        }
    )
    creds = cs.get_atlassian_credentials()
    assert creds["email"] == "legacy@example.com"
    cs.clear_atlassian_credentials()
    assert cs.get_atlassian_credentials() == {}


def test_clear_atlassian_credentials() -> None:
    cs.save_jira_credentials(
        {
            "email": "u@example.com",
            "api_token": "tok",
            "site_url": "https://x.atlassian.net",
            "cloud_id": "c1",
        }
    )
    cs.clear_atlassian_credentials()
    assert cs.get_jira_credentials() == {}
    assert cs.get_confluence_credentials() == {}


def test_get_integration_status_confluence_authenticated() -> None:
    cs.save_confluence_credentials(
        {
            "email": "wiki@example.com",
            "api_token": "wiki-tok",
            "site_url": "https://team.atlassian.net",
            "cloud_id": "c2",
        }
    )
    status = cs.get_integration_status("confluence")
    assert status["authenticated"] is True
    assert status["email"] == "wiki@example.com"


def test_save_jira_credentials_requires_token() -> None:
    with pytest.raises(cs.ProviderCredentialError, match="API token is required"):
        cs.save_jira_credentials({"email": "u@example.com", "api_token": "  "})


def test_clear_integration_tokens_confluence() -> None:
    cs.save_confluence_credentials(
        {
            "email": "wiki@example.com",
            "api_token": "wiki-tok",
            "site_url": "https://team.atlassian.net",
            "cloud_id": "c2",
        }
    )
    cs.clear_integration_tokens("confluence")
    assert cs.get_confluence_credentials() == {}


def test_clear_integration_tokens_jira() -> None:
    cs.save_jira_credentials(
        {
            "email": "u@example.com",
            "api_token": "tok",
            "site_url": "https://x.atlassian.net",
            "cloud_id": "c1",
        }
    )
    cs.clear_integration_tokens("jira")
    assert cs.get_jira_credentials() == {}


# --- test_integration_profile.py ---


def test_build_atlassian_integration_record_nested_and_flat() -> None:
    record = build_atlassian_integration_record(
        {
            "email": "user@example.com",
            "cloud_id": "cid",
            "site_url": "https://acme.atlassian.net",
            "site_name": "acme",
        }
    )
    assert record["account"]["email"] == "user@example.com"
    assert record["site"]["site_url"] == "https://acme.atlassian.net"
    assert record["email"] == "user@example.com"
    assert record["site_url"] == "https://acme.atlassian.net"


def test_build_product_integration_record_sets_provider() -> None:
    record = build_product_integration_record(
        "jira",
        {
            "email": "user@example.com",
            "cloud_id": "cid",
            "site_url": "https://acme.atlassian.net",
            "site_name": "acme",
        },
    )
    assert record["provider"] == "jira"
    assert record["account"]["email"] == "user@example.com"


def test_get_integration_status_reads_legacy_atlassian_flat_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cs, "_load_file_secret", lambda *a, **k: "legacy-token")
    monkeypatch.setattr(cs, "_store_file_secret", lambda *a, **k: None)
    cs._write_payload(
        {
            "integrations": {
                "atlassian": {
                    "provider": "atlassian",
                    "auth_type": "api_token",
                    "email": "legacy@example.com",
                    "site_url": "https://legacy.atlassian.net",
                    "site_name": "legacy",
                    "cloud_id": "cloud-1",
                }
            }
        }
    )
    status = cs.get_integration_status("atlassian")
    assert status["authenticated"] is True
    assert status["email"] == "legacy@example.com"
    assert status["site_url"] == "https://legacy.atlassian.net"


def test_clear_jira_credentials_preserves_shared_legacy_for_confluence() -> None:
    """Product logout must not remove legacy storage still used by the other product."""
    cs.save_atlassian_credentials(
        {
            "email": "shared@example.com",
            "api_token": "shared-tok",
            "site_url": "https://team.atlassian.net",
            "cloud_id": "cloud-shared",
        }
    )
    cs.save_confluence_credentials(
        {
            "email": "wiki@example.com",
            "api_token": "wiki-tok",
            "site_url": "https://team.atlassian.net",
            "cloud_id": "c2",
        }
    )
    cs.save_jira_credentials(
        {
            "email": "jira@example.com",
            "api_token": "jira-tok",
            "site_url": "https://team.atlassian.net",
            "cloud_id": "c1",
        }
    )

    cs.clear_jira_credentials()

    assert cs.get_confluence_credentials().get("api_token") == "wiki-tok"
    integrations = cs.read_credentials().get("integrations", {})
    assert "atlassian" in integrations
    assert "confluence" in integrations
    assert "jira" not in integrations
    assert cs._JIRA_TOKEN_SECRET not in cs._read_integration_secrets_file()
    secrets = cs._read_integration_secrets_file()
    assert cs._ATLASSIAN_LEGACY_TOKEN_SECRET in secrets


def test_clear_atlassian_credentials_removes_shared_legacy() -> None:
    cs.save_atlassian_credentials(
        {
            "email": "legacy@example.com",
            "api_token": "legacy-token",
            "site_url": "https://legacy.atlassian.net",
            "cloud_id": "cloud-1",
        }
    )
    assert cs.get_jira_credentials().get("api_token") == "legacy-token"

    cs.clear_atlassian_credentials()

    assert cs.get_jira_credentials() == {}
    assert "atlassian" not in cs.read_credentials().get("integrations", {})


def test_atlassian_account_from_entry_email_only() -> None:
    from potpie.auth.adapters.integration_profile import (
        atlassian_account_from_entry,
    )

    assert atlassian_account_from_entry({"email": "u@example.com"}) == {
        "email": "u@example.com"
    }


# --- test_integration_verify.py ---


def test_verify_message_for_kind() -> None:
    assert "invalid" in _verify_message_for_kind(
        "jira", AtlassianAuthErrorKind.INVALID_CREDENTIALS
    )
    assert "scope" in _verify_message_for_kind(
        "confluence", AtlassianAuthErrorKind.INSUFFICIENT_SCOPES
    )


def test_verify_atlassian_product_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "potpie.auth.adapters.integration_verify.fetch_cloud_id_for_site",
        lambda _url: "cloud-1",
    )
    monkeypatch.setattr(
        "potpie.auth.adapters.integration_verify.verify_gateway_product",
        lambda *args, **kwargs: AtlassianVerifyResult(
            ok=True,
            display_name="Ada",
            error_kind=None,
        ),
    )
    ok, message = _verify_atlassian_product(
        "jira",
        {
            "email": "ada@example.com",
            "api_token": "tok",
            "site_url": "https://team.atlassian.net",
            "site_name": "team",
        },
    )
    assert ok is True
    assert "Ada" in message


def test_verify_integration_access_jira(monkeypatch) -> None:
    monkeypatch.setattr(
        "potpie.auth.adapters.integration_verify._verify_atlassian_product",
        lambda _p, _c: (True, "ok (Ada @ team)"),
    )
    ok, message = verify_integration_access(
        "jira",
        {"email": "a@example.com", "api_token": "t", "site_url": "https://x.net"},
    )
    assert ok is True
    assert "Ada" in message


# --- test_provider_config.py ---


def test_atlassian_gateway_urls() -> None:
    assert atlassian_jira_gateway_url("abc") == (f"{ATLASSIAN_API_GATEWAY}/ex/jira/abc")
    assert atlassian_confluence_gateway_url("abc") == (
        f"{ATLASSIAN_API_GATEWAY}/ex/confluence/abc"
    )
