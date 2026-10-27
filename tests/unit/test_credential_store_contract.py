"""GitLab credential behavior shared by the production store and auth-test fake."""

import json
from pathlib import Path

import pytest

from potpie.auth.adapters import credentials_store as cs
from potpie.auth.adapters.credentials import FileCredentialStore
from potpie.auth.ports.credentials import CredentialStore
from tests._auth_fakes import InMemoryCredentialStore


@pytest.fixture
def file_credential_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> FileCredentialStore:
    # XDG isolates metadata, the real JSON secret store, and legacy fallback.
    # No credential reads or writes need to be replaced with test doubles.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return FileCredentialStore()


@pytest.fixture(params=["file", "memory"])
def credential_store(request: pytest.FixtureRequest) -> CredentialStore:
    if request.param == "file":
        return request.getfixturevalue("file_credential_store")
    return InMemoryCredentialStore()


def _save_instance(store: CredentialStore, host: str) -> None:
    store.save_gitlab_credentials(
        {
            "instance_host": host,
            "instance_url": f"https://{host}",
            "personal_access_token": f"test-token-for-{host}",
        },
        account={"username": f"user@{host}"},
    )


def test_empty_store_has_no_gitlab_credentials(
    credential_store: CredentialStore,
) -> None:
    assert credential_store.list_gitlab_instances() == []
    assert credential_store.get_gitlab_credentials() == {}
    assert credential_store.get_gitlab_credentials("missing.example") == {}


def test_saving_instances_selects_the_latest_and_preserves_each_account(
    credential_store: CredentialStore,
) -> None:
    _save_instance(credential_store, "first.example")
    assert credential_store.get_gitlab_credentials()["instance_host"] == "first.example"
    assert credential_store.list_gitlab_instances()[0]["active"] is True

    _save_instance(credential_store, "second.example")
    for host in ("first.example", "second.example"):
        credentials = credential_store.get_gitlab_credentials(host)
        assert credentials["instance_host"] == host
        assert credentials["personal_access_token"] == f"test-token-for-{host}"
        assert credentials["account"] == {"username": f"user@{host}"}
        # Explicit host lookups must not change the default selection.
        assert credential_store.get_gitlab_credentials()["instance_host"] == (
            "second.example"
        )

    assert credential_store.get_gitlab_credentials() == (
        credential_store.get_gitlab_credentials("second.example")
    )
    instances = credential_store.list_gitlab_instances()
    assert {row["instance_host"]: row["active"] for row in instances} == {
        "first.example": False,
        "second.example": True,
    }
    assert all("personal_access_token" not in row for row in instances)


def test_saving_an_existing_instance_makes_it_active(
    credential_store: CredentialStore,
) -> None:
    _save_instance(credential_store, "first.example")
    _save_instance(credential_store, "second.example")
    _save_instance(credential_store, "first.example")

    assert credential_store.get_gitlab_credentials() == (
        credential_store.get_gitlab_credentials("first.example")
    )
    assert {
        row["instance_host"]: row["active"]
        for row in credential_store.list_gitlab_instances()
    } == {"first.example": True, "second.example": False}


@pytest.mark.parametrize(
    ("deleted_host", "remaining_host"),
    [
        pytest.param("first.example", "second.example", id="delete-inactive"),
        pytest.param("second.example", "first.example", id="delete-active"),
    ],
)
def test_deleting_an_instance_keeps_the_remaining_instance_active(
    credential_store: CredentialStore, deleted_host: str, remaining_host: str
) -> None:
    _save_instance(credential_store, "first.example")
    _save_instance(credential_store, "second.example")

    credential_store.clear_gitlab_credentials(deleted_host)

    assert credential_store.get_gitlab_credentials(deleted_host) == {}
    remaining = credential_store.get_gitlab_credentials(remaining_host)
    assert remaining["personal_access_token"] == f"test-token-for-{remaining_host}"
    assert credential_store.get_gitlab_credentials() == remaining
    assert {
        row["instance_host"]: row["active"]
        for row in credential_store.list_gitlab_instances()
    } == {remaining_host: True}


def test_deleting_the_last_instance_clears_the_selection(
    credential_store: CredentialStore,
) -> None:
    _save_instance(credential_store, "first.example")

    credential_store.clear_gitlab_credentials("first.example")

    assert credential_store.list_gitlab_instances() == []
    assert credential_store.get_gitlab_credentials() == {}
    assert credential_store.get_gitlab_credentials("first.example") == {}


def test_clearing_all_instances_removes_default_and_explicit_credentials(
    credential_store: CredentialStore,
) -> None:
    _save_instance(credential_store, "first.example")
    _save_instance(credential_store, "second.example")

    credential_store.clear_gitlab_credentials()

    assert credential_store.list_gitlab_instances() == []
    assert credential_store.get_gitlab_credentials() == {}
    assert credential_store.get_gitlab_credentials("first.example") == {}
    assert credential_store.get_gitlab_credentials("second.example") == {}


def test_file_store_persists_metadata_and_secrets_in_isolated_storage(
    file_credential_store: FileCredentialStore, tmp_path: Path
) -> None:
    metadata_path = tmp_path / "xdg" / "potpie" / "credentials.json"
    secrets_path = tmp_path / "xdg" / "potpie" / "integration_secrets.json"
    assert file_credential_store.credentials_path() == metadata_path
    assert cs.integration_secrets_path() == secrets_path
    assert cs.legacy_config_dir().parent == tmp_path / "xdg"

    _save_instance(file_credential_store, "first.example")
    _save_instance(file_credential_store, "second.example")

    metadata = metadata_path.read_text(encoding="utf-8")
    secrets = json.loads(secrets_path.read_text(encoding="utf-8"))
    for host in ("first.example", "second.example"):
        assert f"test-token-for-{host}" not in metadata
        assert f"test-token-for-{host}" in secrets.values()

    reopened = FileCredentialStore()
    assert reopened.get_gitlab_credentials() == (
        file_credential_store.get_gitlab_credentials("second.example")
    )
    reopened.clear_gitlab_credentials("second.example")
    assert FileCredentialStore().get_gitlab_credentials()["instance_host"] == (
        "first.example"
    )
    secrets = json.loads(secrets_path.read_text(encoding="utf-8"))
    assert set(secrets.values()) == {"test-token-for-first.example"}

    reopened.clear_gitlab_credentials()
    assert FileCredentialStore().get_gitlab_credentials() == {}
    assert not secrets_path.exists()
