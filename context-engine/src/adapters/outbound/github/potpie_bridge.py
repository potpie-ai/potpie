"""Wrap Potpie's code provider object as SourceControlPort."""

from __future__ import annotations

from typing import Any, Iterator


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

    def get_pull_request_commits(self, repo_name: str, pr_number: int) -> list[dict[str, Any]]:
        return self._provider.get_pull_request_commits(repo_name, pr_number)

    def get_pull_request_review_comments(
        self,
        repo_name: str,
        pr_number: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._provider.get_pull_request_review_comments(repo_name, pr_number, limit)

    def get_pull_request_issue_comments(
        self,
        repo_name: str,
        pr_number: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self._provider.get_pull_request_issue_comments(repo_name, pr_number, limit)

    def get_issue(self, repo_name: str, issue_number: int) -> dict[str, Any]:
        return self._provider.get_issue(repo_name, issue_number)

    def iter_closed_pulls(self, repo_name: str) -> Iterator[Any]:
        client = self._provider.get_client()
        if client is None:
            return iter(())
        repo = client.get_repo(repo_name)
        return iter(repo.get_pulls(state="closed", sort="updated", direction="asc"))
