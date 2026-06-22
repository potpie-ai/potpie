"""Unit tests for Bitbucket interactive read flows."""

from __future__ import annotations

import pytest

from adapters.inbound.cli.auth.bitbucket_read import run_bitbucket_use_flow
from adapters.inbound.cli.commands._common import set_store
from adapters.outbound.cli_auth.bitbucket_read_client import BitbucketReadError
from tests._auth_fakes import InMemoryCredentialStore


def _patch_load_bitbucket_credentials(
    monkeypatch: pytest.MonkeyPatch,
    store: InMemoryCredentialStore,
) -> None:
    """`load_bitbucket_read_credentials` reads the real store; use the test fake."""
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.bitbucket_read.load_bitbucket_read_credentials",
        lambda: dict(store.get_bitbucket_credentials()),
    )


@pytest.fixture(autouse=True)
def _reset_store() -> None:
    set_store(InMemoryCredentialStore())


def test_run_bitbucket_use_flow_non_tty_requires_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(BitbucketReadError, match="Interactive workspace selection"):
        run_bitbucket_use_flow()


def test_run_bitbucket_use_flow_non_tty_with_workspace_and_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryCredentialStore()
    store.save_bitbucket_credentials(
        {"email": "user@example.com", "api_token": "bb-token"}
    )
    set_store(store)
    _patch_load_bitbucket_credentials(monkeypatch, store)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.bitbucket_read.fetch_bitbucket_workspaces",
        lambda: [{"key": "potpie", "name": "Potpie"}],
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.bitbucket_read.fetch_bitbucket_repositories",
        lambda workspace_key, limit=50: [{"key": "backend", "name": "Backend"}],
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.bitbucket_read.fetch_bitbucket_pull_requests",
        lambda workspace_key, repo_slug, limit=10: [
            {"id": 1, "title": "Fix auth", "state": "OPEN"}
        ],
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.bitbucket_read.save_bitbucket_workspace_prefs",
        lambda **kwargs: None,
    )

    result = run_bitbucket_use_flow(
        workspace_key="potpie",
        repo_slug="backend",
        limit=5,
    )

    assert result["workspace_key"] == "potpie"
    assert result["repo_key"] == "backend"
    assert result["items"][0]["title"] == "Fix auth"


def test_run_bitbucket_use_flow_single_workspace_and_repo_auto_picks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryCredentialStore()
    store.save_bitbucket_credentials(
        {"email": "user@example.com", "api_token": "bb-token"}
    )
    set_store(store)
    _patch_load_bitbucket_credentials(monkeypatch, store)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.bitbucket_read.fetch_bitbucket_workspaces",
        lambda: [{"key": "solo-ws", "name": "Solo"}],
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.bitbucket_read.fetch_bitbucket_repositories",
        lambda workspace_key, limit=50: [{"key": "solo-repo", "name": "Solo Repo"}],
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.bitbucket_read.fetch_bitbucket_pull_requests",
        lambda workspace_key, repo_slug, limit=10: [],
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.bitbucket_read.save_bitbucket_workspace_prefs",
        lambda **kwargs: None,
    )

    result = run_bitbucket_use_flow()

    assert result["workspace_key"] == "solo-ws"
    assert result["repo_key"] == "solo-repo"


def test_run_bitbucket_use_flow_prompts_when_multiple_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryCredentialStore()
    store.save_bitbucket_credentials(
        {"email": "user@example.com", "api_token": "bb-token"}
    )
    set_store(store)
    _patch_load_bitbucket_credentials(monkeypatch, store)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.bitbucket_read.fetch_bitbucket_workspaces",
        lambda: [
            {"key": "a", "name": "A"},
            {"key": "b", "name": "B"},
        ],
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.bitbucket_read.fetch_bitbucket_repositories",
        lambda workspace_key, limit=50: [
            {"key": "repo-a", "name": "Repo A"},
            {"key": "repo-b", "name": "Repo B"},
        ],
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.bitbucket_read.fetch_bitbucket_pull_requests",
        lambda workspace_key, repo_slug, limit=10: [],
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.bitbucket_read._prompt_workspace",
        lambda workspaces, label="": workspaces[1],
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.bitbucket_read.save_bitbucket_workspace_prefs",
        lambda **kwargs: None,
    )

    result = run_bitbucket_use_flow()

    assert result["workspace_key"] == "b"
    assert result["repo_key"] == "repo-b"
