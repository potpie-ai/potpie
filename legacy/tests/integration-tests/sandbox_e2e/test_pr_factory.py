"""End-to-end coverage of the GitHubGitPlatformProvider factory pattern.

These tests pin the contracts that came out of Fix 3:

* The platform provider takes a **factory** (``Callable[[str], ICodeProvider]``)
  rather than a singleton — installation tokens expire in 1h and the
  installation id is per-repo, so a singleton would silently break PR
  creation after the first hour.
* The factory is invoked **once per** :meth:`create_pull_request` call,
  with the request's ``repo_name`` as the only argument.
* ``SandboxClient.create_pull_request`` reaches the provider only when
  the workspace is writable; the existing read-only refusal stays in
  place (covered in ``test_local_sandbox.py`` already, not duplicated
  here).
* ``__init__`` invariant: exactly one of ``provider_factory`` or
  ``provider`` must be passed.

The tests use a fake ``ICodeProvider`` so the assertions don't depend
on GitHub being reachable — the real ``CodeProviderFactory.create_provider_with_fallback``
is wired in production via :func:`get_sandbox_client`, but here we want
to assert the bridge's own behaviour in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sandbox import (
    PullRequest,
    PullRequestRequest,
    RepoIdentity,
    SandboxClient,
    SandboxOpError,
    SandboxSettings,
    WorkspaceMode,
    build_sandbox_container,
)
from sandbox.domain.errors import PullRequestFailed

from app.modules.sandbox_repos import GitHubGitPlatformProvider


REPO_NAME = "owner/test-repo"


# ----------------------------------------------------------------------
# Fake ICodeProvider
# ----------------------------------------------------------------------
class _FakeProvider:
    """Minimal stand-in for ICodeProvider; only `create_pull_request` is
    used by the bridge."""

    def __init__(
        self,
        *,
        succeed: bool = True,
        pr_number: int = 42,
        url: str = "https://github.com/owner/test-repo/pull/42",
        error: str | None = None,
    ) -> None:
        self.succeed = succeed
        self.pr_number = pr_number
        self.url = url
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create_pull_request(
        self,
        repo_name: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str,
        reviewers: list[str] | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "repo_name": repo_name,
                "title": title,
                "body": body,
                "head_branch": head_branch,
                "base_branch": base_branch,
                "reviewers": reviewers,
                "labels": labels,
            }
        )
        if not self.succeed:
            return {"success": False, "error": self.error or "boom"}
        return {
            "success": True,
            "pr_number": self.pr_number,
            "url": self.url,
        }


# ======================================================================
# Construction invariants
# ======================================================================
class TestConstruction:
    def test_requires_exactly_one_of_factory_or_provider(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            GitHubGitPlatformProvider()
        with pytest.raises(ValueError, match="exactly one"):
            GitHubGitPlatformProvider(
                provider_factory=lambda r: _FakeProvider(),
                provider=_FakeProvider(),
            )

    def test_legacy_singleton_construction_still_works(self) -> None:
        """The fixed-provider shape is the back-compat path used by tests
        and embedded callers that build their own GitHubProvider."""
        fake = _FakeProvider()
        bridge = GitHubGitPlatformProvider(provider=fake)
        # The bridge stashes it; we don't assert internal state here
        # because that's a private contract — just that construction succeeds.
        assert bridge.kind == "github"


# ======================================================================
# Factory invocation contract
# ======================================================================
class TestFactoryDispatch:
    """The factory is the load-bearing piece — it gets called per PR
    request with the repo_name, and the bridge uses its return value
    fresh each time."""

    async def test_factory_called_once_per_pr_with_repo_name(self) -> None:
        calls: list[str] = []

        def factory(repo_name: str) -> _FakeProvider:
            calls.append(repo_name)
            return _FakeProvider()

        bridge = GitHubGitPlatformProvider(provider_factory=factory)
        request = PullRequestRequest(
            repo=RepoIdentity(repo_name=REPO_NAME),
            title="t", body="b",
            head_branch="agent/edits-1", base_branch="main",
        )
        pr = await bridge.create_pull_request(request)
        assert isinstance(pr, PullRequest)
        assert pr.id == 42
        assert calls == [REPO_NAME], (
            "factory must be called exactly once per PR with the request's "
            "repo_name — that's how the fresh-token-per-repo property holds"
        )

    async def test_factory_reinvoked_for_every_request(self) -> None:
        """Two PR requests = two factory invocations.

        This pins the freshness property: the bridge must NOT cache the
        provider, otherwise installation tokens (1h TTL) would expire
        mid-conversation and PR creation would silently break.
        """
        invocations = 0
        providers: list[_FakeProvider] = []

        def factory(repo_name: str) -> _FakeProvider:
            nonlocal invocations
            invocations += 1
            new = _FakeProvider(pr_number=invocations)
            providers.append(new)
            return new

        bridge = GitHubGitPlatformProvider(provider_factory=factory)

        async def open_pr(branch: str) -> PullRequest:
            return await bridge.create_pull_request(
                PullRequestRequest(
                    repo=RepoIdentity(repo_name=REPO_NAME),
                    title=f"t-{branch}",
                    body="b",
                    head_branch=branch,
                    base_branch="main",
                )
            )

        await open_pr("agent/edits-a")
        await open_pr("agent/edits-b")
        assert invocations == 2, (
            "factory must be re-invoked on every PR — caching the "
            "provider across calls is a regression"
        )
        # And each invocation got its own provider instance.
        assert providers[0] is not providers[1]

    async def test_factory_passes_request_kwargs_through(self) -> None:
        fake = _FakeProvider()
        bridge = GitHubGitPlatformProvider(
            provider_factory=lambda repo_name: fake
        )
        await bridge.create_pull_request(
            PullRequestRequest(
                repo=RepoIdentity(repo_name=REPO_NAME),
                title="The PR",
                body="Body **md**",
                head_branch="agent/edits-x",
                base_branch="develop",
                reviewers=("alice", "bob"),
                labels=("bug", "auto"),
            )
        )
        assert fake.calls == [
            {
                "repo_name": REPO_NAME,
                "title": "The PR",
                "body": "Body **md**",
                "head_branch": "agent/edits-x",
                "base_branch": "develop",
                "reviewers": ["alice", "bob"],
                "labels": ["bug", "auto"],
            }
        ]

    async def test_provider_failure_surfaces_as_typed_error(self) -> None:
        """When the underlying GitHub call returns ``success=False``,
        the bridge raises :class:`PullRequestFailed` — so the agent
        sees a typed error, not a generic exception."""
        fake = _FakeProvider(succeed=False, error="branch protection")
        bridge = GitHubGitPlatformProvider(
            provider_factory=lambda repo_name: fake
        )
        with pytest.raises(PullRequestFailed, match="branch protection"):
            await bridge.create_pull_request(
                PullRequestRequest(
                    repo=RepoIdentity(repo_name=REPO_NAME),
                    title="t", body="b",
                    head_branch="agent/edits", base_branch="main",
                )
            )


# ======================================================================
# End-to-end: SandboxClient → bridge → fake provider
# ======================================================================
class TestEndToEnd:
    """Bring up a real ``SandboxClient`` against the local fixture
    upstream, wire the bridge with a fake factory, and verify the full
    ``client.create_pull_request`` flow.

    This is the contract the agent's ``sandbox_pr`` tool depends on.
    """

    async def test_client_create_pull_request_round_trip(
        self, repos_base: Path, metadata_path: Path, upstream_repo: Path
    ) -> None:
        fake = _FakeProvider(pr_number=101, url="https://example/pr/101")
        bridge = GitHubGitPlatformProvider(
            provider_factory=lambda repo_name: fake
        )
        settings = SandboxSettings(
            provider="local", runtime="local_subprocess",
            repos_base_path=str(repos_base),
            metadata_path=str(metadata_path),
            local_allow_write=True,
        )
        container = build_sandbox_container(
            settings,
            git_platform_provider=bridge,
        )
        client = SandboxClient.from_container(container)
        handle = await client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-pr", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="pr-flow",
        )
        # Make a tiny commit so the branch has something on it (the
        # fake factory doesn't actually verify the remote, but it's
        # how the real flow looks end-to-end).
        await client.write_file(handle, "x.txt", b"x\n")
        await client.commit(handle, "x", author=("Bot", "b@b.com"))

        pr = await client.create_pull_request(
            handle,
            repo=REPO_NAME,
            title="agent PR",
            body="generated by potpie",
            base_branch="main",
        )
        assert pr.id == 101
        assert pr.url == "https://example/pr/101"
        assert pr.head_branch == "agent/edits-pr"
        assert pr.base_branch == "main"
        # The bridge passed the request straight through.
        assert len(fake.calls) == 1
        assert fake.calls[0]["repo_name"] == REPO_NAME
        assert fake.calls[0]["head_branch"] == "agent/edits-pr"

    async def test_client_create_pull_request_refuses_readonly(
        self, repos_base: Path, metadata_path: Path, upstream_repo: Path
    ) -> None:
        """Already covered in test_local_sandbox.py (without a wired
        platform provider — there the test asserts the refusal happens
        BEFORE the platform call). Here we verify the same invariant
        when the bridge IS wired: capability gating fires first, the
        factory is never invoked.
        """
        fake = _FakeProvider()
        invocations = 0

        def factory(repo_name: str) -> _FakeProvider:
            nonlocal invocations
            invocations += 1
            return fake

        bridge = GitHubGitPlatformProvider(provider_factory=factory)
        settings = SandboxSettings(
            provider="local", runtime="local_subprocess",
            repos_base_path=str(repos_base),
            metadata_path=str(metadata_path),
            local_allow_write=True,
        )
        container = build_sandbox_container(
            settings,
            git_platform_provider=bridge,
        )
        client = SandboxClient.from_container(container)
        analysis = await client.get_workspace(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="main", base_ref="main",
            mode=WorkspaceMode.ANALYSIS,
        )
        with pytest.raises(SandboxOpError, match="writable"):
            await client.create_pull_request(
                analysis,
                repo=REPO_NAME,
                title="should not happen",
                body="...",
                base_branch="main",
            )
        assert invocations == 0, (
            "capability gate must fire before the factory is invoked — "
            "otherwise auth is resolved for nothing and the failure mode "
            "looks like a platform error to the user"
        )
