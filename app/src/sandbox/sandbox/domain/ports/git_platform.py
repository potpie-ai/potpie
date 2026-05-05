"""Git-platform provider port (PRs, comments, reviews).

Sandbox-core stops at ``commit`` and ``push``; everything beyond that
(opening a pull request, adding a comment, requesting a review) is the
remote forge's responsibility — and forges have very different APIs
across GitHub, GitLab, Bitbucket, Gitea, etc. Splitting this into its
own port lets adapters wrap whichever forge SDK is in scope without
contaminating the workspace/runtime layers.

PR creation and PR comments are surfaced today; review hooks are easy
to add the same way as we need them.
"""

from __future__ import annotations

from typing import Protocol

from sandbox.domain.models import (
    PullRequest,
    PullRequestComment,
    PullRequestCommentResult,
    PullRequestRequest,
)


class GitPlatformProvider(Protocol):
    kind: str

    async def create_pull_request(
        self, request: PullRequestRequest
    ) -> PullRequest:
        """Open a PR from ``request.head_branch`` into ``request.base_branch``.

        Implementations should raise :class:`PullRequestFailed` (or a
        subclass) on failure rather than returning a sentinel — the
        caller is the application service which already maps domain
        errors at the boundary.
        """
        ...

    async def comment_on_pull_request(
        self, request: PullRequestComment
    ) -> PullRequestCommentResult:
        """Post a comment on an existing PR — top-level or inline.

        Implementations should raise :class:`PullRequestFailed` on
        failure, mirroring :meth:`create_pull_request`. ``path`` and
        ``line`` are inline-only — when one is set, both must be.
        """
        ...
