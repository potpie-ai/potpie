"""Integration credential paths in credentials_store."""

from __future__ import annotations

import pytest

from adapters.inbound.cli import credentials_store as cs


@pytest.fixture
def keychain(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    stored: dict[tuple[str, str], str] = {}

    def set_password(service: str, username: str, password: str) -> None:
        stored[(service, username)] = password

    def get_password(service: str, username: str) -> str | None:
        return stored.get((service, username))

    def delete_password(service: str, username: str) -> None:
        stored.pop((service, username), None)

    monkeypatch.setattr(cs.keyring, "set_password", set_password)
    monkeypatch.setattr(cs.keyring, "get_password", get_password)
    monkeypatch.setattr(cs.keyring, "delete_password", delete_password)
    return stored


def test_jira_credentials_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
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
    assert cs._JIRA_TOKEN_SECRET not in {k[1] for k in keychain}


def test_list_integration_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
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


def test_get_atlassian_credentials_prefers_jira(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
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


def test_get_integration_status_jira_unauthenticated(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    status = cs.get_integration_status("jira")
    assert status["authenticated"] is False


def test_confluence_credentials_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
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


def test_get_integration_tokens_jira(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
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


def test_get_integration_status_jira_authenticated(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
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


def test_save_atlassian_credentials_legacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
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


def test_clear_atlassian_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
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


def test_get_integration_status_confluence_authenticated(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
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


def test_save_jira_credentials_requires_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    with pytest.raises(cs.ProviderCredentialError, match="API token is required"):
        cs.save_jira_credentials({"email": "u@example.com", "api_token": "  "})


def test_clear_integration_tokens_confluence(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
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


def test_readable_credentials_path_prefers_legacy_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    cfg = tmp_path / "xdg"
    legacy_cfg = tmp_path / "legacy-xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg))
    monkeypatch.setattr(cs, "legacy_config_dir", lambda: legacy_cfg)
    legacy_cfg.mkdir(parents=True)
    legacy_file = legacy_cfg / "credentials.json"
    legacy_file.write_text('{"api_key": "from-legacy"}', encoding="utf-8")

    assert cs.readable_credentials_path() == legacy_file
    assert cs.read_credentials().get("api_key") == "from-legacy"


def test_read_credentials_invalid_json_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    path = tmp_path / "credentials.json"
    path.write_text("{not json", encoding="utf-8")
    assert cs.read_credentials() == {}


def test_get_integration_status_unknown_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    with pytest.raises(ValueError, match="Unknown integration provider"):
        cs.get_integration_status("github")


def test_store_secure_secret_generic_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def set_password(service: str, username: str, password: str) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(cs.keyring, "set_password", set_password)
    with pytest.raises(cs.CredentialStoreError, match="keychain"):
        cs.store_secure_secret("name", "secret")


def test_load_secure_secret_generic_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def get_password(service: str, username: str) -> str | None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(cs.keyring, "get_password", get_password)
    with pytest.raises(cs.CredentialStoreError, match="keychain"):
        cs.load_secure_secret("name")


def test_clear_integration_tokens_jira(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
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
