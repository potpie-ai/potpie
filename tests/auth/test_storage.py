"""Canonical credential storage, isolation, and authentication selection."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from potpie.auth.adapters import credentials_store as cs
from potpie.auth.adapters import potpie_api_config as config
from potpie.auth.adapters.credentials import FileCredentialStore
from potpie.auth.adapters.firebase_session import FirebaseSession
from tests._auth_fakes import InMemoryCredentialStore

CREATED = "2026-10-27T00:00:00Z"
API_KEY = "sk-storage-contract"


def _github(token="github-token"):  # noqa: S107 - fixture credential.
    return {
        "provider": "github",
        "provider_host": "github.com",
        "access_token": token,
        "account": {"login": "octocat", "id": 1, "email": "octo@example.com"},
        "scopes": ["repo"],
        "created_at": CREATED,
        "updated_at": CREATED,
        "metadata": {"auth_flow": "device"},
    }


def test_files_are_private_and_published_atomically(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    original = cs.os.replace
    publications = []

    def replace(source, destination):
        path = Path(source)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        publications.append((Path(destination).name, Path(destination).exists()))
        original(source, destination)

    monkeypatch.setattr(cs.os, "replace", replace)
    cs.store_potpie_api_key(API_KEY, created_at=CREATED)
    cs.write_api_base_url("https://api.example.com/")

    assert cs.config_dir() == tmp_path / "xdg" / "potpie"
    for path in (cs.credentials_path(), cs.integration_secrets_path()):
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert ("credentials.json", False) in publications
    assert ("integration_secrets.json", False) in publications
    metadata = cs.read_credentials()
    assert API_KEY not in json.dumps(metadata)
    assert "api_key" not in metadata
    assert metadata["api_base_url"] == "https://api.example.com"
    assert metadata["integrations"]["potpie"] == {
        "auth_type": "api_key",
        "token_storage": "file",
        "created_at": CREATED,
    }
    assert cs.get_stored_api_key() == API_KEY
    assert cs.get_stored_api_base_url() == "https://api.example.com"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://new.example/", "https://new.example"),
        (None, "https://old.example"),
        ("", ""),
    ],
    ids=["replace", "preserve", "clear"],
)
def test_base_url_updates_preserve_the_api_key(value, expected):
    cs.store_potpie_api_key(API_KEY, created_at=CREATED)
    cs.write_api_base_url("https://old.example")
    cs.write_api_base_url(value)
    assert cs.get_stored_api_base_url() == expected
    assert cs.get_stored_api_key() == API_KEY


@pytest.mark.parametrize(
    "clear_api_key", [False, True], ids=["keep-api-key", "clear-api-key"]
)
def test_session_rotation_and_logout_keep_secrets_out_of_metadata(clear_api_key):
    cs.store_potpie_api_key(API_KEY, created_at=CREATED)
    cs.write_api_base_url("https://api.example.com")
    cs.store_potpie_firebase_refresh_token(
        "refresh", created_at=CREATED, firebase_api_key="firebase-key"
    )
    metadata = cs.read_credentials()
    cs.store_potpie_firebase_id_token("id-token")
    cs.update_potpie_firebase_refresh_token("rotated-refresh")
    assert cs.read_credentials() == metadata
    assert cs.get_potpie_firebase_refresh_token() == "rotated-refresh"
    assert cs.get_potpie_firebase_id_token() == "id-token"
    assert cs.get_potpie_firebase_api_key() == "firebase-key"
    assert cs.get_potpie_auth_type() == "potpie"
    for secret in (API_KEY, "rotated-refresh", "id-token", "firebase-key"):
        assert secret not in json.dumps(metadata)

    cs.clear_potpie_auth(clear_api_key=clear_api_key)
    assert cs.get_potpie_firebase_refresh_token() == ""
    assert cs.get_potpie_firebase_id_token() == ""
    assert cs.get_potpie_firebase_api_key() == ""
    assert cs.get_stored_api_key() == ("" if clear_api_key else API_KEY)
    assert cs.get_stored_api_base_url() == "https://api.example.com"
    assert cs.get_integration_metadata("potpie") == {}


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: cs.store_potpie_api_key("invalid", created_at=CREATED),
            "must start with sk-",
        ),
        (
            lambda: cs.store_potpie_firebase_refresh_token(" ", created_at=CREATED),
            "refresh token is required",
        ),
        (lambda: cs.store_potpie_firebase_id_token(" "), "ID token is required"),
        (
            lambda: cs.update_potpie_firebase_refresh_token(" "),
            "refresh token is required",
        ),
        (lambda: cs.store_secure_secret(" ", "token"), "secret name must be non-empty"),
        (lambda: cs.get_integration_metadata(" "), "provider must be non-empty"),
    ],
    ids=["api-key", "refresh", "id-token", "rotation", "secret-name", "provider-name"],
)
def test_invalid_credentials_fail_before_writing(operation, message):
    with pytest.raises((cs.CredentialStoreError, ValueError), match=message):
        operation()
    assert not cs.credentials_path().exists()
    assert not cs.integration_secrets_path().exists()


def test_secret_round_trip_missing_reads_and_repeated_deletion():
    assert cs.load_secure_secret("example") == ""
    cs.delete_secure_secret("example")
    cs.store_secure_secret("example", "secret")
    assert cs.load_secure_secret("example") == "secret"
    assert json.loads(cs.integration_secrets_path().read_text())["example"] == "secret"
    cs.delete_secure_secret("example")
    cs.delete_secure_secret("example")
    assert cs.load_secure_secret("example") == ""


@pytest.mark.parametrize("action", ["store", "delete", "default-label"])
def test_secret_write_errors_preserve_the_operation_and_label(monkeypatch, action):
    if action == "delete":
        cs.store_secure_secret("example", "secret")

    def fail(_secrets):
        raise OSError("permission denied")

    monkeypatch.setattr(cs, "_write_integration_secrets_file", fail)
    label = {} if action == "default-label" else {"label": "Example token"}
    with pytest.raises(cs.CredentialStoreError) as caught:
        if action == "delete":
            cs.delete_secure_secret("example", **label)
        else:
            cs.store_secure_secret("example", "new-secret", **label)
    assert (
        "remove" in str(caught.value)
        if action == "delete"
        else "store" in str(caught.value)
    )
    assert "local credentials file" in str(caught.value)
    if label:
        assert "Example token" in str(caught.value)


@pytest.mark.parametrize("with_api_key", [False, True])
def test_metadata_round_trip_and_removal_preserve_other_credentials(with_api_key):
    if with_api_key:
        cs.store_potpie_api_key(API_KEY, created_at=CREATED)
    cs.write_integration_metadata(
        "Example", {"auth_type": "oauth", "token_storage": "file"}
    )
    expected = {"auth_type": "oauth", "token_storage": "file"}
    assert cs.get_integration_metadata("example") == expected
    assert cs.list_integration_metadata()["example"] == expected
    cs.clear_integration_metadata("example")
    assert cs.get_integration_metadata("example") == {}
    assert cs.get_stored_api_key() == (API_KEY if with_api_key else "")
    assert cs.credentials_path().exists() is with_api_key


@pytest.mark.parametrize(
    "raw", ["{not json", "[]", "null"], ids=["invalid-json", "array", "null"]
)
def test_invalid_metadata_is_read_as_empty(raw):
    path = cs.credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw)
    assert cs.read_credentials() == {}


@pytest.mark.parametrize("with_api_key", [False, True])
def test_pot_selection_cleanup_preserves_unrelated_authentication(with_api_key):
    if with_api_key:
        cs.store_potpie_api_key(API_KEY, created_at=CREATED)
    pot_id = "33333333-3333-3333-3333-333333333333"
    cs.register_pot_alias("My-Workspace", pot_id)
    cs.set_active_pot_id(pot_id)
    assert cs.get_pot_aliases() == {"my-workspace": pot_id}
    assert cs.resolve_cli_pot_ref("MY-WORKSPACE") == (pot_id, "")
    cs.clear_active_pot_id()
    assert cs.get_active_pot_id() == ""
    assert "active_pot_id" not in cs.read_credentials()
    assert cs.get_pot_aliases() == {"my-workspace": pot_id}
    cs.clear_pot_scope_state()
    assert cs.get_pot_aliases() == {}
    assert cs.get_stored_api_key() == (API_KEY if with_api_key else "")
    assert cs.credentials_path().exists() is with_api_key


@pytest.mark.parametrize(
    ("ref", "stored", "expected", "error"),
    [
        (
            "550E8400-E29B-41D4-A716-446655440000",
            {},
            "550e8400-e29b-41d4-a716-446655440000",
            "",
        ),
        ("nope", {}, None, "Unknown pot"),
        ("broken", {"broken": "not-a-uuid"}, None, "not a valid UUID"),
    ],
    ids=["uuid-normalization", "unknown-alias", "corrupt-alias"],
)
def test_pot_reference_resolution_reports_invalid_aliases(ref, stored, expected, error):
    cs._write_payload({"pot_aliases": stored})
    resolved, message = cs.resolve_cli_pot_ref(ref)
    assert resolved == expected
    assert error in message
    if ref == "nope":
        assert "pot create" in message


def test_clearing_all_credentials_removes_both_files():
    cs.store_potpie_api_key(API_KEY, created_at=CREATED)
    cs.clear_credentials()
    assert not cs.credentials_path().exists()
    assert cs.get_stored_api_key() == ""


def test_provider_credentials_use_the_file_store_and_preserve_other_fields():
    cs.store_potpie_api_key(API_KEY, created_at=CREATED)
    cs.write_api_base_url("https://api.example.com")
    cs.set_active_pot_id("11111111-1111-1111-1111-111111111111")
    cs.register_pot_alias("demo", "22222222-2222-2222-2222-222222222222")
    cs.write_provider_credentials("github", _github())
    metadata = cs.read_credentials()
    assert metadata["active_pot_id"] == "11111111-1111-1111-1111-111111111111"
    assert metadata["pot_aliases"] == {"demo": "22222222-2222-2222-2222-222222222222"}
    assert "github-token" not in json.dumps(metadata)
    assert metadata["integrations"]["github"]["account"]["login"] == "octocat"
    assert metadata["integrations"]["github"]["token_storage"] == "file"
    assert cs.get_provider_credentials("github")["access_token"] == "github-token"
    assert cs.get_stored_api_key() == API_KEY
    assert cs.get_stored_api_base_url() == "https://api.example.com"
    assert cs.get_integration_status("github") == {
        "provider": "github",
        "authenticated": True,
        "auth_type": "oauth",
        "login": "octocat",
        "email": "octo@example.com",
        "scope": ["repo"],
        "stored_at": CREATED,
        "token_storage": "file",
        "provider_host": "github.com",
    }
    cs._write_integration_secrets_file(
        {"github_token": "replacement", "potpie_api_key": API_KEY}
    )
    assert cs.get_provider_credentials("github")["access_token"] == "replacement"
    cs.clear_provider_credentials("github")
    assert cs.get_provider_credentials("github") == {}
    assert cs.get_integration_status("github")["authenticated"] is False
    assert cs.get_stored_api_key() == API_KEY


def test_missing_provider_secret_is_not_authenticated():
    cs.write_provider_credentials("github", _github())
    cs._write_integration_secrets_file({})
    with pytest.raises(
        cs.ProviderCredentialError,
        match="GitHub token not found in local credentials file",
    ):
        cs.get_provider_credentials("github")
    assert cs.get_integration_status("github")["authenticated"] is False


@pytest.mark.parametrize("operation", ["save", "delete"])
def test_provider_secret_write_errors_do_not_mutate_metadata(monkeypatch, operation):
    cs.write_provider_credentials("github", _github())
    before = cs.read_credentials()

    def fail(_secrets):
        raise OSError("permission denied")

    monkeypatch.setattr(cs, "_write_integration_secrets_file", fail)
    with pytest.raises(cs.ProviderCredentialError, match="local credentials file"):
        if operation == "save":
            cs.write_provider_credentials("github", _github("new-token"))
        else:
            cs.clear_provider_credentials("github")
    assert cs.read_credentials() == before


@pytest.mark.parametrize(
    ("existing", "cleanup_fails"),
    [(None, False), ("old-token", False), (None, True)],
    ids=["new-token-removed", "existing-token-restored", "metadata-error-preserved"],
)
def test_provider_metadata_failure_rolls_back_the_secret(
    monkeypatch, existing, cleanup_fails
):
    if existing:
        cs.write_provider_credentials("github", _github(existing))
    before = cs.read_credentials()
    cleanup_calls = []

    def fail_metadata(_provider, _metadata):
        raise OSError("metadata write failed")

    def fail_cleanup(*args):
        cleanup_calls.append(args)
        raise cs.ProviderCredentialError("cleanup failed")

    monkeypatch.setattr(cs, "write_integration_metadata", fail_metadata)
    if cleanup_fails:
        monkeypatch.setattr(cs, "_delete_file_secret", fail_cleanup)
    with pytest.raises(OSError, match="metadata write failed"):
        cs.write_provider_credentials("github", _github("new-token"))
    assert cs.read_credentials() == before
    if cleanup_fails:
        assert len(cleanup_calls) == 1
    elif existing:
        assert cs.get_provider_credentials("github")["access_token"] == existing
    else:
        assert not cs.integration_secrets_path().exists()


@pytest.mark.parametrize("product", ["jira", "confluence"])
def test_atlassian_products_have_independent_nested_metadata_and_secrets(product):
    for name in ("jira", "confluence"):
        getattr(cs, f"save_{name}_credentials")(
            {
                "email": f"{name}@example.com",
                "api_token": f"{name}-token",
                "site_url": "https://team.atlassian.net",
                "cloud_id": "cloud-1",
            }
        )
    metadata = cs.read_credentials()["integrations"][product]
    assert metadata["provider"] == product
    assert metadata["account"]["email"] == f"{product}@example.com"
    assert metadata["site"]["site_url"] == "https://team.atlassian.net"
    assert "site_url" not in metadata and "email" not in metadata
    assert f"{product}-token" not in json.dumps(metadata)
    assert cs.get_integration_tokens(product)["api_token"] == f"{product}-token"
    assert cs.get_integration_status(product)["authenticated"] is True
    assert product in cs.list_integration_providers()
    cs.clear_integration_tokens(product)
    assert cs.get_integration_tokens(product) == {}
    assert cs.get_integration_status(product)["authenticated"] is False
    other = "confluence" if product == "jira" else "jira"
    assert cs.get_integration_tokens(other)["api_token"] == f"{other}-token"


def test_linear_workspaces_keep_secrets_scoped_to_the_organization():
    for org_id in ("first", "second"):
        cs.save_linear_organization_tokens(
            org_id,
            {
                "access_token": f"access-{org_id}",
                "refresh_token": f"refresh-{org_id}",
                "scope": "read,write",
                "expires_at": 9999999999,
            },
            organization={"id": org_id, "name": org_id.title(), "url_key": org_id},
            account={"name": "Ada", "email": "ada@example.com"},
        )
    metadata = cs.get_integration_metadata("linear")
    assert "organization" not in metadata and "account" not in metadata
    assert set(metadata["organizations"]) == {"first", "second"}
    assert metadata["organizations"]["first"]["scopes"] == ["read", "write"]
    assert "access-first" not in json.dumps(metadata)
    assert cs.get_linear_tokens()["access_token"] == "access-second"
    cs.set_active_linear_organization("first")
    assert cs.get_linear_tokens()["refresh_token"] == "refresh-first"
    assert cs.get_linear_tokens("second")["access_token"] == "access-second"
    assert cs.get_integration_status("linear")["site_name"] == "First"
    assert cs.get_integration_status("linear")["scope"] == ["read", "write"]
    assert cs.get_integration_status("linear")["workspace_count"] == 2
    assert [row["active"] for row in cs.list_linear_organizations()] == [True, False]
    cs.clear_integration_tokens("linear")
    assert cs.get_linear_tokens() == {} and cs.get_linear_tokens("second") == {}
    assert cs.get_integration_status("linear")["authenticated"] is False
    assert not cs.integration_secrets_path().exists()


@pytest.fixture(params=["file", "memory"])
def credential_store(request):
    return (
        FileCredentialStore() if request.param == "file" else InMemoryCredentialStore()
    )


def _save_instance(store, host):
    store.save_gitlab_credentials(
        {
            "instance_host": host,
            "instance_url": f"https://{host}",
            "personal_access_token": f"token-{host}",
        },
        account={"username": f"user@{host}"},
    )


@pytest.mark.parametrize("deletion", ["active", "inactive", "all", "last"])
def test_file_and_memory_stores_share_the_gitlab_selection_contract(
    credential_store, deletion
):
    store = credential_store
    assert store.list_gitlab_instances() == []
    assert (
        store.get_gitlab_credentials() == store.get_gitlab_credentials("missing") == {}
    )
    _save_instance(store, "first.example")
    if isinstance(store, FileCredentialStore):
        assert (
            FileCredentialStore().get_gitlab_credentials()
            == store.get_gitlab_credentials()
        )
        text = cs.credentials_path().read_text()
        assert "token-first.example" not in text
        assert (
            "token-first.example"
            in json.loads(cs.integration_secrets_path().read_text()).values()
        )
    if deletion != "last":
        _save_instance(store, "second.example")
        assert store.get_gitlab_credentials()["instance_host"] == "second.example"
        assert (
            store.get_gitlab_credentials("first.example")["account"]["username"]
            == "user@first.example"
        )
        _save_instance(store, "first.example")
        assert store.get_gitlab_credentials()["instance_host"] == "first.example"
        assert {
            row["instance_host"]: row["active"] for row in store.list_gitlab_instances()
        } == {"first.example": True, "second.example": False}
        assert all(
            "personal_access_token" not in row for row in store.list_gitlab_instances()
        )
    if deletion == "all":
        store.clear_gitlab_credentials()
    else:
        removed = "second.example" if deletion == "inactive" else "first.example"
        store.clear_gitlab_credentials(removed)
        assert store.get_gitlab_credentials(removed) == {}
    remaining = "first.example" if deletion == "inactive" else "second.example"
    if deletion in {"all", "last"}:
        assert store.list_gitlab_instances() == []
        assert store.get_gitlab_credentials() == {}
        if isinstance(store, FileCredentialStore):
            assert FileCredentialStore().get_gitlab_credentials() == {}
            assert not cs.integration_secrets_path().exists()
        assert (
            store.get_gitlab_credentials("first.example")
            == store.get_gitlab_credentials("second.example")
            == {}
        )
    else:
        assert store.get_gitlab_credentials() == store.get_gitlab_credentials(remaining)
        assert (
            store.get_gitlab_credentials()["personal_access_token"]
            == f"token-{remaining}"
        )
        assert {
            row["instance_host"]: row["active"] for row in store.list_gitlab_instances()
        } == {remaining: True}


@pytest.mark.parametrize(
    ("environment", "stored", "expected"),
    [
        (None, "", "http://localhost:8001"),
        (None, "https://stored.example/", "https://stored.example"),
        ("https://env.example/", "https://stored.example", "https://env.example"),
    ],
    ids=["distribution-default", "stored", "environment"],
)
def test_api_url_precedence_and_normalization(
    monkeypatch, environment, stored, expected
):
    monkeypatch.setenv("POTPIE_ENVIRONMENT", "test")
    monkeypatch.delenv("POTPIE_API_URL", raising=False)
    if environment is not None:
        monkeypatch.setenv("POTPIE_API_URL", environment)
    monkeypatch.setattr(config, "get_stored_api_base_url", lambda: stored)
    assert config.resolve_potpie_api_base_url() == expected


@pytest.mark.parametrize(
    "mode", ["environment-key", "stored-key", "cached-session", "refresh-session"]
)
def test_authentication_selection_and_session_rotation(monkeypatch, mode):
    monkeypatch.delenv("POTPIE_API_KEY", raising=False)
    if mode == "environment-key":
        monkeypatch.setenv("POTPIE_API_KEY", "sk-env")
    monkeypatch.setattr(config, "get_stored_api_key", lambda: "sk-stored")
    monkeypatch.setattr(
        config, "get_potpie_auth_type", lambda: "" if mode == "stored-key" else "potpie"
    )
    monkeypatch.setattr(config, "get_potpie_firebase_refresh_token", lambda: "refresh")
    monkeypatch.setattr(
        config,
        "get_potpie_firebase_id_token",
        lambda: "cached" if mode == "cached-session" else "",
    )
    monkeypatch.setattr(config, "get_potpie_firebase_api_key", lambda: "firebase-key")
    monkeypatch.setattr(
        config, "id_token_expires_at", lambda _token: config.time.time() + 60
    )
    refreshes, rotations, cached = [], [], []

    def refresh(token, *, firebase_api_key=None):
        refreshes.append((token, firebase_api_key))
        return FirebaseSession(
            id_token="new-id", refresh_token="rotated", expires_at=123
        )

    monkeypatch.setattr(config, "refresh_id_token", refresh)
    monkeypatch.setattr(
        config, "update_potpie_firebase_refresh_token", rotations.append
    )
    monkeypatch.setattr(config, "store_potpie_firebase_id_token", cached.append)
    auth = config.resolve_potpie_auth_config()
    if mode in {"environment-key", "stored-key"}:
        assert auth.mode == "api_key"
        assert auth.headers == {
            "X-API-Key": "sk-env" if mode == "environment-key" else "sk-stored"
        }
    else:
        assert auth.mode == "potpie"
        assert auth.headers == {
            "Authorization": "Bearer cached"
            if mode == "cached-session"
            else "Bearer new-id"
        }
    assert refreshes == (
        [("refresh", "firebase-key")] if mode == "refresh-session" else []
    )
    assert rotations == (["rotated"] if mode == "refresh-session" else [])
    assert cached == (["new-id"] if mode == "refresh-session" else [])


@pytest.mark.parametrize(
    "resolver",
    [config.resolve_potpie_api_key, config.resolve_potpie_auth_config],
    ids=["api-key", "authentication"],
)
def test_missing_authentication_is_explicit(monkeypatch, resolver):
    monkeypatch.delenv("POTPIE_API_KEY", raising=False)
    monkeypatch.setattr(config, "get_stored_api_key", lambda: "")
    monkeypatch.setattr(config, "get_potpie_auth_type", lambda: "")
    monkeypatch.setattr(config, "get_potpie_firebase_refresh_token", lambda: "")
    with pytest.raises(ValueError, match="Potpie.*missing"):
        resolver()


@pytest.mark.parametrize("product", ["jira", "confluence"])
def test_product_credentials_require_a_nonempty_secret(product):
    with pytest.raises(cs.ProviderCredentialError, match="API token is required"):
        getattr(cs, f"save_{product}_credentials")(
            {"email": "ada@example.com", "api_token": " "}
        )
    assert not cs.credentials_path().exists()


def test_unknown_stored_integration_is_explicit():
    with pytest.raises(ValueError, match="Unknown integration provider"):
        cs.get_integration_status("unknown")
