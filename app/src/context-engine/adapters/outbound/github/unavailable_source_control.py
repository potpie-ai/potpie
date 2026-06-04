"""Stub GitHub source control when no token is configured (raw ingest does not call GitHub)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from domain.ports.source_control import SourceControlPort

_MSG = (
    "GitHub API is not configured (set CONTEXT_ENGINE_GITHUB_TOKEN or GITHUB_TOKEN). "
    "This operation requires GitHub access."
)


class UnavailableSourceControl(SourceControlPort):
    def get_pull_request(
        self,
        repo_name: str,
        pr_number: int,
        include_diff: bool = False,
    ) -> dict[str, Any]:
        raise RuntimeError(_MSG)

    def get_pull_request_commits(self, repo_name: str, pr_number: int) -> list[dict[str, Any]]:
        raise RuntimeError(_MSG)

    def get_pull_request_review_comments(
        self,
        repo_name: str,
        pr_number: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        raise RuntimeError(_MSG)

    def get_pull_request_issue_comments(
        self,
        repo_name: str,
        pr_number: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        raise RuntimeError(_MSG)

    def get_issue(self, repo_name: str, issue_number: int) -> dict[str, Any]:
        raise RuntimeError(_MSG)

    def iter_closed_pulls(self, repo_name: str) -> Iterator[Any]:
        raise RuntimeError(_MSG)
