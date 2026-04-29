"""GitPlatformProvider implementation backed by the existing GitHub provider.

The sandbox-core module defines ``GitPlatformProvider`` as a Protocol and
ships PR creation as a value-typed contract (`PullRequestRequest` →
`PullRequest`). This bridge wraps the legacy
``app.modules.code_provider.github.GitHubProvider`` so the sandbox
client can reuse the existing GitHub authentication chain (GitHub App
token, OAuth, env-token fallback) without re-implementing it.

The legacy provider is sync; we hop into a thread to keep the sandbox
service async-clean.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from sandbox.domain.errors import PullRequestFailed
from sandbox.domain.models import PullRequest, PullRequestRequest

if TYPE_CHECKING:
    from app.modules.code_provider.github.github_provider import GitHubProvider


class GitHubGitPlatformProvider:
    """Adapter that delegates PR operations to the legacy GitHubProvider.

    Construct with an already-authenticated provider instance:
    auth-resolution stays in the legacy code path, the bridge is purely
    a port adaptation.
    """

    kind = "github"

    def __init__(self, provider: "GitHubProvider") -> None:
        self._provider = provider

    async def create_pull_request(
        self, request: PullRequestRequest
    ) -> PullRequest:
        result: dict[str, Any] = await asyncio.to_thread(
            self._provider.create_pull_request,
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
