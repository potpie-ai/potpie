"""Adapt Potpie's code provider (GitHub) to context-engine SourceControlPort."""

from __future__ import annotations

from typing import Any, Iterator

from adapters.outbound.connectors.github.api_client import PyGithubSourceControl


class CodeProviderSourceControl:
    """Delegates to an object exposing get_pull_request, get_client, etc. (GitHubProvider)."""

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    def get_pull_request(
        self,
        repo_name: str,
        pr_number: int,
        include_diff: bool = False,
    ) -> dict[str, Any]:
        return self._provider.get_pull_request(repo_name, pr_number, include_diff)

    def get_pull_request_commits(
        self, repo_name: str, pr_number: int
    ) -> list[dict[str, Any]]:
        return self._provider.get_pull_request_commits(repo_name, pr_number)

    def get_pull_request_review_comments(
        self,
        repo_name: str,
        pr_number: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._provider.get_pull_request_review_comments(
            repo_name, pr_number, limit
        )

    def get_pull_request_issue_comments(
        self,
        repo_name: str,
        pr_number: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self._provider.get_pull_request_issue_comments(
            repo_name, pr_number, limit
        )

    def get_issue(self, repo_name: str, issue_number: int) -> dict[str, Any]:
        return self._provider.get_issue(repo_name, issue_number)

    def iter_closed_pulls(self, repo_name: str) -> Iterator[Any]:
        client = self._provider.get_client()
        if client is None:
            return iter(())
        repo = client.get_repo(repo_name)
        return iter(repo.get_pulls(state="closed", sort="updated", direction="desc"))

    def list_pull_requests(
        self,
        repo_name: str,
        *,
        state: str = "all",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compact PR refs, newest-first, bounded by the backfill window/cap.

        Delegates to :class:`PyGithubSourceControl` so the window + item-cap
        contract (shared by every connector's list tools) lives in one place
        rather than being re-derived here. Providers without a PyGithub client
        (local/tunnel) have nothing to enumerate, so we return ``[]`` — the
        same fail-soft as :meth:`iter_closed_pulls`.
        """
        client = self._provider.get_client()
        if client is None:
            return []
        return PyGithubSourceControl(client).list_pull_requests(
            repo_name, state=state, limit=limit
        )

    def list_issues(
        self,
        repo_name: str,
        *,
        state: str = "all",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compact issue refs (PRs excluded), newest-first, window/cap bounded.

        See :meth:`list_pull_requests` for why this delegates rather than
        delegating to the provider's own ``list_issues`` (which honours neither
        the trailing window nor the hard item cap).
        """
        client = self._provider.get_client()
        if client is None:
            return []
        return PyGithubSourceControl(client).list_issues(
            repo_name, state=state, limit=limit
        )
