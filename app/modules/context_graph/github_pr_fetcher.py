"""Helpers to fetch complete PR payloads for ingestion."""

from typing import Any

from app.modules.context_graph.deterministic_extractors import extract_issue_refs
from app.modules.context_graph.review_thread_grouper import group_review_threads
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


def fetch_full_pr(github_provider: Any, repo_name: str, pr_number: int) -> dict[str, Any]:
    pr_data = github_provider.get_pull_request(repo_name, pr_number, include_diff=True)
    commits = github_provider.get_pull_request_commits(repo_name, pr_number)
    review_comments = github_provider.get_pull_request_review_comments(repo_name, pr_number)
    review_threads = group_review_threads(review_comments)
    issue_comments = github_provider.get_pull_request_issue_comments(repo_name, pr_number)

    issue_refs = set(extract_issue_refs(pr_data.get("body")))
    for commit in commits:
        issue_refs.update(extract_issue_refs(commit.get("message")))

    linked_issues = []
    for issue_number in sorted(issue_refs):
        try:
            linked_issues.append(github_provider.get_issue(repo_name, issue_number))
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
