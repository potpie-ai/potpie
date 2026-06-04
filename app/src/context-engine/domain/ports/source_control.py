"""Git / GitHub read API (port)."""

from typing import Any, Iterator, Protocol


class SourceControlPort(Protocol):
    def get_pull_request(
        self,
        repo_name: str,
        pr_number: int,
        include_diff: bool = False,
    ) -> dict[str, Any]:
        ...

    def get_pull_request_commits(self, repo_name: str, pr_number: int) -> list[dict[str, Any]]:
        ...

    def get_pull_request_review_comments(
        self,
        repo_name: str,
        pr_number: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        ...

    def get_pull_request_issue_comments(
        self,
        repo_name: str,
        pr_number: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        ...

    def get_issue(self, repo_name: str, issue_number: int) -> dict[str, Any]:
        ...

    def iter_closed_pulls(self, repo_name: str) -> Iterator[Any]:
        """Yield closed PRs (sort=updated, direction=asc). Each has .number, .merged_at."""
