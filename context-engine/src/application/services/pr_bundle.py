"""Fetch full PR payload for ingestion (orchestration over SourceControlPort)."""

from __future__ import annotations

import logging
from typing import Any

from domain.deterministic_extractors import extract_issue_refs
from domain.review_thread_grouper import group_review_threads
from domain.ports.source_control import SourceControlPort

logger = logging.getLogger(__name__)


def fetch_full_pr(
    source: SourceControlPort,
    repo_name: str,
    pr_number: int,
) -> dict[str, Any]:
    pr_data = source.get_pull_request(repo_name, pr_number, include_diff=True)
    commits = source.get_pull_request_commits(repo_name, pr_number)
    review_comments = source.get_pull_request_review_comments(repo_name, pr_number)
    review_threads = group_review_threads(review_comments)
    issue_comments = source.get_pull_request_issue_comments(repo_name, pr_number)

    issue_refs: set[int] = set(extract_issue_refs(pr_data.get("body")))
    for commit in commits:
        issue_refs.update(extract_issue_refs(commit.get("message")))

    linked_issues = []
    for issue_number in sorted(issue_refs):
        try:
            linked_issues.append(source.get_issue(repo_name, issue_number))
        except Exception as exc:
            logger.warning(
                "Failed fetching linked issue #%s for %s: %s",
                issue_number,
                repo_name,
                exc,
            )

    return {
        "pr_data": pr_data,
        "commits": commits,
        "review_threads": review_threads,
        "issue_comments": issue_comments,
        "linked_issues": linked_issues,
    }
