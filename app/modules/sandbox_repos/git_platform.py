"""GitPlatformProvider implementation backed by the existing GitHub provider.

The sandbox-core module defines ``GitPlatformProvider`` as a Protocol and
ships PR creation as a value-typed contract (`PullRequestRequest` →
`PullRequest`). This bridge wraps the legacy
``app.modules.code_provider.github.GitHubProvider`` so the sandbox
client reuses the existing GitHub authentication chain (GitHub App
token, OAuth, env-token fallback) without re-implementing it.

Construction shape: a **factory callable** rather than a singleton
provider. The factory is invoked per-PR with ``request.repo.repo_name``
so each call gets a freshly authenticated provider — GitHub App
installation tokens expire in 1h, and the installation id is per-repo,
so caching one provider at startup would silently break PR creation.
The legacy provider is sync; we hop into a thread to keep the sandbox
service async-clean.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

from sandbox.domain.errors import PullRequestFailed
from sandbox.domain.models import (
    PullRequest,
    PullRequestComment,
    PullRequestCommentResult,
    PullRequestRequest,
)

if TYPE_CHECKING:
    from app.modules.code_provider.base.code_provider_interface import ICodeProvider


# Typed against the interface, not the concrete GitHub class, so the
# factory can hand back any `ICodeProvider` impl (GitHub.com, GitHub
# Enterprise, GitBucket, future GitLab) without an isinstance dance.
# `create_pull_request` is on the interface — the only method we call.
ProviderFactory = Callable[[str], "ICodeProvider"]


class GitHubGitPlatformProvider:
    """Adapter that delegates PR operations to the legacy code-provider.

    Despite the historical name, this works for any provider that
    implements :class:`ICodeProvider.create_pull_request` — GitHub,
    GitHub Enterprise, GitBucket. We keep the ``GitHub`` prefix only
    because GitHub.com is the dominant call site; renaming is a
    downstream cleanup.

    The constructor takes ``provider_factory`` — a callable that, given
    a ``repo_name`` (``"owner/repo"``), returns a freshly authenticated
    code provider. The factory is called on every
    :meth:`create_pull_request` invocation; do NOT cache the result
    upstream of this class.

    ``provider`` is kept as a kwarg for back-compat with the older
    one-shot construction shape used by tests; passing both raises.
    """

    kind = "github"

    def __init__(
        self,
        *,
        provider_factory: ProviderFactory | None = None,
        provider: "ICodeProvider | None" = None,
    ) -> None:
        if (provider_factory is None) == (provider is None):
            raise ValueError(
                "GitHubGitPlatformProvider requires exactly one of "
                "`provider_factory` (production) or `provider` (legacy/tests)."
            )
        self._provider_factory: ProviderFactory | None = provider_factory
        self._fixed_provider: "ICodeProvider | None" = provider

    async def create_pull_request(
        self, request: PullRequestRequest
    ) -> PullRequest:
        provider = await asyncio.to_thread(self._get_provider, request.repo.repo_name)
        result: dict[str, Any] = await asyncio.to_thread(
            provider.create_pull_request,
            request.repo.repo_name,
            request.title,
            request.body,
            request.head_branch,
            request.base_branch,
            list(request.reviewers) or None,
            list(request.labels) or None,
        )
        if not result.get("success"):
            raise PullRequestFailed(
                str(result.get("error") or "GitHub PR creation failed")
            )
        return PullRequest(
            id=int(result["pr_number"]),
            url=str(result["url"]),
            title=request.title,
            head_branch=request.head_branch,
            base_branch=request.base_branch,
            backend_kind=self.kind,
        )

    async def comment_on_pull_request(
        self, request: PullRequestComment
    ) -> PullRequestCommentResult:
        """Post a comment on a PR through the legacy GitHubProvider.

        Inline (``path`` + ``line``) and top-level shapes are dispatched
        identically — the provider's ``add_pull_request_comment``
        switches on ``path``/``line`` itself. We construct the result
        URL from ``repo_name`` + ``pr_number`` + ``comment_id`` because
        the underlying API only returns the id (the URL pattern is
        stable on GitHub).
        """
        if (request.path is None) != (request.line is None):
            raise PullRequestFailed(
                "PullRequestComment.path and .line must both be set "
                "(inline comment) or both omitted (top-level comment)"
            )
        provider = await asyncio.to_thread(
            self._get_provider, request.repo.repo_name
        )
        result: dict[str, Any] = await asyncio.to_thread(
            provider.add_pull_request_comment,
            request.repo.repo_name,
            request.pr_number,
            request.body,
            request.commit_id,
            request.path,
            request.line,
        )
        if not result.get("success"):
            raise PullRequestFailed(
                str(result.get("error") or "GitHub PR comment failed")
            )
        comment_id = result["comment_id"]
        # Inline review comments and top-level issue comments use
        # different URL anchors on GitHub. Construct the right one so
        # the agent can surface a stable link to the user.
        anchor = (
            f"discussion_r{comment_id}"
            if (request.path and request.line)
            else f"issuecomment-{comment_id}"
        )
        url = (
            f"https://github.com/{request.repo.repo_name}"
            f"/pull/{request.pr_number}#{anchor}"
        )
        return PullRequestCommentResult(
            id=int(comment_id),
            url=url,
            backend_kind=self.kind,
        )

    def _get_provider(self, repo_name: str) -> "ICodeProvider":
        """Return the provider used for this PR.

        The factory path runs the auth chain inline so credentials are
        always fresh. The fixed-provider path (legacy / tests) hands
        back the same instance every time and is the caller's
        responsibility to keep authenticated.
        """
        if self._provider_factory is not None:
            return self._provider_factory(repo_name)
        assert self._fixed_provider is not None  # invariant from __init__
        return self._fixed_provider
