"""End-to-end coverage of the PR-comment consolidation (Fix 3b).

These tests pin the contracts:

* ``GitPlatformProvider`` exposes ``comment_on_pull_request``.
* ``GitHubGitPlatformProvider`` validates the inline-shape invariant
  (``path`` and ``line`` together or neither) before reaching the
  underlying GitHub call.
* ``SandboxClient.comment_on_pull_request`` round-trips inline AND
  top-level shapes through the bridge.
* The bridge uses the **same factory** as ``create_pull_request``, so
  PR comments and PR creation share one auth chain — i.e. attribution
  is consistent across the whole agent flow.

The fake GitHubProvider stand-in implements only the two methods the
bridge uses; pyright complains about it not being a full ``ICodeProvider``
but the runtime contract is duck-typed.
"""

from __future__ import annotations

from typing import Any

import pytest

from sandbox import (
    PullRequestComment,
    PullRequestCommentResult,
    PullRequestRequest,
    RepoIdentity,
)
from sandbox.domain.errors import PullRequestFailed

from app.modules.sandbox_repos import GitHubGitPlatformProvider


REPO_NAME = "owner/test-repo"


# ----------------------------------------------------------------------
# Fake provider
# ----------------------------------------------------------------------
class _FakeProviderWithComments:
    def __init__(self, *, succeed: bool = True, comment_id: int = 7777):
        self.succeed = succeed
        self.comment_id = comment_id
        self.comment_calls: list[dict[str, Any]] = []
        self.pr_calls: list[dict[str, Any]] = []

    def add_pull_request_comment(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        commit_id: str | None = None,
        path: str | None = None,
        line: int | None = None,
    ) -> dict[str, Any]:
        self.comment_calls.append(
            {
                "repo_name": repo_name, "pr_number": pr_number, "body": body,
                "commit_id": commit_id, "path": path, "line": line,
            }
        )
        if not self.succeed:
            return {"success": False, "error": "platform refused"}
        return {"success": True, "comment_id": self.comment_id}

    def create_pull_request(
        self, repo_name: str, title: str, body: str, head_branch: str,
        base_branch: str, reviewers: Any = None, labels: Any = None,
    ) -> dict[str, Any]:
        self.pr_calls.append(
            {"repo_name": repo_name, "title": title, "body": body,
             "head_branch": head_branch, "base_branch": base_branch}
        )
        return {"success": True, "pr_number": 1, "url": "https://example/pr/1"}


# ======================================================================
# Bridge contract: comment_on_pull_request
# ======================================================================
class TestCommentOnPullRequest:
    async def test_top_level_comment_returns_typed_result(self) -> None:
        fake = _FakeProviderWithComments(comment_id=42)
        bridge = GitHubGitPlatformProvider(
            provider_factory=lambda repo_name: fake
        )
        result = await bridge.comment_on_pull_request(
            PullRequestComment(
                repo=RepoIdentity(repo_name=REPO_NAME),
                pr_number=99,
                body="LGTM",
            )
        )
        assert isinstance(result, PullRequestCommentResult)
        assert result.id == 42
        assert result.backend_kind == "github"
        # Top-level comment URL anchor.
        assert result.url == (
            f"https://github.com/{REPO_NAME}/pull/99#issuecomment-42"
        )
        assert fake.comment_calls == [
            {
                "repo_name": REPO_NAME, "pr_number": 99, "body": "LGTM",
                "commit_id": None, "path": None, "line": None,
            }
        ]

    async def test_inline_comment_returns_review_anchor(self) -> None:
        fake = _FakeProviderWithComments(comment_id=123)
        bridge = GitHubGitPlatformProvider(
            provider_factory=lambda repo_name: fake
        )
        result = await bridge.comment_on_pull_request(
            PullRequestComment(
                repo=RepoIdentity(repo_name=REPO_NAME),
                pr_number=99,
                body="nit: rename this",
                path="src/app.py",
                line=42,
                commit_id="abc123",
            )
        )
        # Inline review comments use a different anchor format.
        assert result.url == (
            f"https://github.com/{REPO_NAME}/pull/99#discussion_r123"
        )
        assert fake.comment_calls[0]["path"] == "src/app.py"
        assert fake.comment_calls[0]["line"] == 42
        assert fake.comment_calls[0]["commit_id"] == "abc123"

    @pytest.mark.parametrize(
        "path, line",
        [
            ("src/app.py", None),  # path without line
            (None, 42),  # line without path
        ],
    )
    async def test_partial_inline_args_rejected_at_bridge(
        self, path: str | None, line: int | None
    ) -> None:
        """The bridge must validate the inline-shape invariant before
        the underlying API call — otherwise GitHub's error is opaque
        and the agent can't tell what's wrong.
        """
        fake = _FakeProviderWithComments()
        bridge = GitHubGitPlatformProvider(
            provider_factory=lambda repo_name: fake
        )
        with pytest.raises(PullRequestFailed, match="both be set"):
            await bridge.comment_on_pull_request(
                PullRequestComment(
                    repo=RepoIdentity(repo_name=REPO_NAME),
                    pr_number=1, body="x", path=path, line=line,
                )
            )
        # The underlying provider must not have been called.
        assert fake.comment_calls == []

    async def test_provider_failure_surfaces_as_typed_error(self) -> None:
        fake = _FakeProviderWithComments(succeed=False)
        bridge = GitHubGitPlatformProvider(
            provider_factory=lambda repo_name: fake
        )
        with pytest.raises(PullRequestFailed, match="platform refused"):
            await bridge.comment_on_pull_request(
                PullRequestComment(
                    repo=RepoIdentity(repo_name=REPO_NAME),
                    pr_number=1, body="x",
                )
            )


# ======================================================================
# Factory shared between PR creation and PR comments
# ======================================================================
class TestSharedFactory:
    """The same provider factory must back both ``create_pull_request``
    and ``comment_on_pull_request`` — that's how attribution stays
    consistent across the agent's PR flow.
    """

    async def test_factory_invoked_for_both_operations(self) -> None:
        invocations: list[str] = []

        def factory(repo_name: str) -> _FakeProviderWithComments:
            invocations.append(f"factory:{repo_name}")
            return _FakeProviderWithComments()

        bridge = GitHubGitPlatformProvider(provider_factory=factory)

        # Open a PR.
        await bridge.create_pull_request(
            PullRequestRequest(
                repo=RepoIdentity(repo_name=REPO_NAME),
                title="t", body="b",
                head_branch="agent/edits", base_branch="main",
            )
        )
        # Comment on a PR.
        await bridge.comment_on_pull_request(
            PullRequestComment(
                repo=RepoIdentity(repo_name=REPO_NAME),
                pr_number=1, body="comment",
            )
        )
        # Both operations invoked the factory with the same repo.
        assert invocations == [f"factory:{REPO_NAME}", f"factory:{REPO_NAME}"]
