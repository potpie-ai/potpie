"""Unit tests for the typed-auth resolver introduced in Fix 4.

``_resolve_auth`` returns the token plus a ``kind`` label (``"context"``,
``"app"``, ``"user_oauth"``, ``"env"``, ``"none"``) so logs can reflect
which branch of the chain produced the credential without dumping the
secret. These tests pin:

* The kind label matches the resolution branch.
* ``"none"`` is returned (not raised) when no credential is found.
* The back-compat ``_resolve_auth_token`` returns just the token, for
  callers that haven't migrated to the typed form yet.
* :class:`PotpieRemoteAuthProvider` reflects ``kind`` onto the
  :class:`RemoteAuth.kind` field that downstream observability reads.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from app.modules.intelligence.tools.sandbox import client as sandbox_client_mod
from app.modules.intelligence.tools.sandbox.client import (
    ResolvedAuth,
    _resolve_auth,
    _resolve_auth_token,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a clean env so chain branches don't interfere."""
    for var in ("GH_TOKEN", "GITHUB_TOKEN", "GITHUB_APP_ID"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def no_context_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``get_auth_token`` to return None so we exercise later
    branches even when the test runner's contextvars happen to be set."""
    monkeypatch.setattr(sandbox_client_mod, "get_auth_token", lambda: None)


# ======================================================================
# Branch labelling
# ======================================================================
class TestResolveAuth:
    def test_context_var_branch_returns_context_kind(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sandbox_client_mod, "get_auth_token", lambda: "ctx-token"
        )
        result = _resolve_auth(user_id="u1", repo_name=None)
        assert result == ResolvedAuth(token="ctx-token", kind="context")

    def test_env_branch_returns_env_kind(
        self,
        monkeypatch: pytest.MonkeyPatch,
        no_context_token: None,
    ) -> None:
        monkeypatch.setenv("GH_TOKEN", "env-token-1")
        result = _resolve_auth(user_id=None, repo_name=None)
        assert result == ResolvedAuth(token="env-token-1", kind="env")

    def test_github_token_env_var_also_routed_to_env_kind(
        self,
        monkeypatch: pytest.MonkeyPatch,
        no_context_token: None,
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "ci-token")
        result = _resolve_auth(user_id=None, repo_name=None)
        assert result == ResolvedAuth(token="ci-token", kind="env")

    def test_no_credential_returns_none_kind_not_exception(
        self, no_context_token: None
    ) -> None:
        """The resolver never raises — it returns ``ResolvedAuth(None, "none")``
        so callers can choose to clone anonymously (which works for public
        repos) instead of having to wrap every call in try/except.
        """
        result = _resolve_auth(user_id=None, repo_name=None)
        assert result == ResolvedAuth(token=None, kind="none")
        # And the back-compat shim returns just the bare None.
        assert _resolve_auth_token(user_id=None, repo_name=None) is None

    def test_context_takes_priority_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If both contextvar and env are set, contextvar wins — the
        harness uses contextvar to pin a specific user's token for a
        given run, and env should never override that.
        """
        monkeypatch.setattr(
            sandbox_client_mod, "get_auth_token", lambda: "from-context"
        )
        monkeypatch.setenv("GH_TOKEN", "from-env")
        result = _resolve_auth(user_id="u1", repo_name="owner/repo")
        assert result.token == "from-context"
        assert result.kind == "context"

    def test_app_branch_failure_falls_through_silently(
        self,
        monkeypatch: pytest.MonkeyPatch,
        no_context_token: None,
    ) -> None:
        """When the GitHub App lookup raises, the resolver swallows the
        error and continues to the next branch instead of failing the
        clone outright. Without this, a transient App lookup blip would
        block every agent run.
        """

        def _boom(repo_name: str) -> Any:  # noqa: ARG001
            raise RuntimeError("App auth blew up")

        with patch(
            "app.modules.code_provider.provider_factory.CodeProviderFactory.create_github_app_provider",
            side_effect=_boom,
        ):
            monkeypatch.setenv("GH_TOKEN", "fallback")
            result = _resolve_auth(user_id=None, repo_name="owner/repo")
        # We landed on env, not raised.
        assert result.kind == "env"
        assert result.token == "fallback"


# ======================================================================
# RemoteAuth.kind passthrough
# ======================================================================
class TestRemoteAuthProviderPassthrough:
    """``PotpieRemoteAuthProvider`` must surface the resolver's ``kind``
    on ``RemoteAuth.kind`` so ``SandboxClient.push`` log lines tell ops
    whether a given push attributes to the bot or to the user."""

    async def test_kind_round_trips_through_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.modules.sandbox_repos import PotpieRemoteAuthProvider
        from sandbox import RepoIdentity

        # Stub the resolver so we can assert kind round-trips for each
        # branch independently.
        monkeypatch.setattr(
            sandbox_client_mod,
            "_resolve_auth",
            lambda user_id, repo_name: ResolvedAuth(
                token="stub-token", kind="user_oauth"
            ),
        )
        provider = PotpieRemoteAuthProvider()
        auth = await provider.auth_for_remote(
            repo=RepoIdentity(repo_name="owner/repo"),
            user_id="u1",
        )
        assert auth is not None
        assert auth.token == "stub-token"
        assert auth.kind == "user_oauth"

    async def test_no_token_returns_none_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the chain produces nothing, the provider returns ``None``
        and the SandboxClient pushes anonymously — the doc-stated
        fallback for public repos."""
        from app.modules.sandbox_repos import PotpieRemoteAuthProvider
        from sandbox import RepoIdentity

        monkeypatch.setattr(
            sandbox_client_mod,
            "_resolve_auth",
            lambda user_id, repo_name: ResolvedAuth(token=None, kind="none"),
        )
        provider = PotpieRemoteAuthProvider()
        auth = await provider.auth_for_remote(
            repo=RepoIdentity(repo_name="owner/repo"),
            user_id="u1",
        )
        assert auth is None
