"""Tests for the local-fs adapter's auth-token resolver."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from sandbox.adapters.outbound.local.auth import (
    env_token_resolver,
    resolve_token,
    set_token_resolver,
)


@pytest.fixture(autouse=True)
def _isolate_env_and_resolver(monkeypatch: pytest.MonkeyPatch):
    """Wipe known auth env vars and restore the default resolver per test.

    Without this, a test that sets a custom resolver leaks into the next one
    via the module-level `_resolver` global.
    """
    for var in (
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN_LIST",
        "CODE_PROVIDER_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    set_token_resolver(env_token_resolver)
    yield
    set_token_resolver(env_token_resolver)


def test_default_returns_none_with_no_env() -> None:
    assert resolve_token() is None


def test_gh_token_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_TOKEN", "ghp_a")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_b")
    assert resolve_token() == "ghp_a"


def test_github_token_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_b")
    assert resolve_token() == "ghp_b"


def test_token_list_first_non_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_TOKEN_LIST", " , a-token,\nb-token")
    assert resolve_token() == "a-token"


def test_legacy_var_last(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODE_PROVIDER_TOKEN", "legacy-token")
    assert resolve_token() == "legacy-token"


def test_set_resolver_overrides() -> None:
    captured = {}

    def custom(*, repo_name: str | None, user_id: str | None) -> str | None:
        captured["args"] = (repo_name, user_id)
        return "from-resolver"

    set_token_resolver(custom)
    assert resolve_token(repo_name="owner/repo", user_id="u1") == "from-resolver"
    assert captured["args"] == ("owner/repo", "u1")


def test_resolver_can_return_none() -> None:
    set_token_resolver(lambda *, repo_name, user_id: None)
    assert resolve_token() is None
