"""GitHub API client used internally by :class:`GitHubConnector`.

This module owns the connector's read access to GitHub. The
``GitHubReadPort`` Protocol below is the narrow internal contract tests
substitute via fakes; it is not exported as a domain port (it lives
behind the connector boundary).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Iterator, List, Protocol

from github import Github

from domain.backfill_window import backfill_window_since, clamp_backfill_limit

logger = logging.getLogger(__name__)


class GitHubReadPort(Protocol):
    """Connector-internal read surface over GitHub."""

    def get_pull_request(
        self,
        repo_name: str,
        pr_number: int,
        include_diff: bool = False,
    ) -> dict[str, Any]: ...

    def get_pull_request_commits(
        self, repo_name: str, pr_number: int
    ) -> list[dict[str, Any]]: ...

    def get_pull_request_review_comments(
        self,
        repo_name: str,
        pr_number: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]: ...

    def get_pull_request_issue_comments(
        self,
        repo_name: str,
        pr_number: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]: ...

    def get_issue(self, repo_name: str, issue_number: int) -> dict[str, Any]: ...

    def iter_closed_pulls(self, repo_name: str) -> Iterator[Any]: ...

    def list_pull_requests(
        self,
        repo_name: str,
        *,
        state: str = "all",
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

    def list_issues(
        self,
        repo_name: str,
        *,
        state: str = "all",
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...


class PyGithubSourceControl(GitHubReadPort):
    def __init__(self, client: Github) -> None:
        self._client = client

    def get_pull_request(
        self,
        repo_name: str,
        pr_number: int,
        include_diff: bool = False,
    ) -> dict[str, Any]:
        repo = self._client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)

        labels: List[dict[str, Any]] = []
        for lb in pr.labels or []:
            labels.append({"name": lb.name})

        milestone = None
        if pr.milestone:
            milestone = {"title": pr.milestone.title}

        result: dict[str, Any] = {
            "number": pr.number,
            "title": pr.title,
            "body": pr.body,
            "state": pr.state,
            "created_at": pr.created_at.isoformat() if pr.created_at else None,
            "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
            "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
            "head_branch": pr.head.ref,
            "base_branch": pr.base.ref,
            "url": pr.html_url,
            "author": pr.user.login if pr.user else "unknown",
            "labels": labels,
            "milestone": milestone,
        }

        if include_diff:
            files = pr.get_files()
            result["files"] = [
                {
                    "filename": f.filename,
                    "status": f.status,
                    "additions": f.additions,
                    "deletions": f.deletions,
                    "patch": f.patch,
                }
                for f in files
            ]

        return result

    def get_pull_request_commits(
        self, repo_name: str, pr_number: int
    ) -> list[dict[str, Any]]:
        repo = self._client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        commits = []
        for commit in pr.get_commits():
            commit_author = None
            if getattr(commit, "author", None):
                commit_author = commit.author.login
            elif getattr(commit, "commit", None) and getattr(
                commit.commit, "author", None
            ):
                commit_author = commit.commit.author.name

            committed_at = None
            if getattr(commit, "commit", None) and getattr(
                commit.commit, "author", None
            ):
                authored_dt = getattr(commit.commit.author, "date", None)
                if authored_dt:
                    committed_at = authored_dt.isoformat()

            commits.append(
                {
                    "sha": commit.sha,
                    "message": commit.commit.message if commit.commit else None,
                    "author": commit_author,
                    "committed_at": committed_at,
                }
            )
        return commits

    def get_pull_request_review_comments(
        self,
        repo_name: str,
        pr_number: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        repo = self._client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        comments = []
        for idx, comment in enumerate(pr.get_review_comments()):
            if idx >= limit:
                break
            comments.append(
                {
                    "id": comment.id,
                    "body": comment.body,
                    "user": {"login": comment.user.login} if comment.user else None,
                    "path": comment.path,
                    "line": comment.line,
                    "in_reply_to_id": comment.in_reply_to_id,
                    "diff_hunk": comment.diff_hunk,
                    "created_at": comment.created_at.isoformat()
                    if comment.created_at
                    else None,
                }
            )
        return comments

    def get_pull_request_issue_comments(
        self,
        repo_name: str,
        pr_number: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        repo = self._client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        comments = []
        for idx, comment in enumerate(pr.get_issue_comments()):
            if idx >= limit:
                break
            comments.append(
                {
                    "id": comment.id,
                    "body": comment.body,
                    "user": {"login": comment.user.login} if comment.user else None,
                    "created_at": comment.created_at.isoformat()
                    if comment.created_at
                    else None,
                }
            )
        return comments

    def get_issue(self, repo_name: str, issue_number: int) -> dict[str, Any]:
        repo = self._client.get_repo(repo_name)
        issue = repo.get_issue(issue_number)
        labels: List[dict[str, Any]] = []
        for lb in issue.labels or []:
            labels.append({"name": lb.name})
        return {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "state": issue.state,
            "created_at": issue.created_at.isoformat() if issue.created_at else None,
            "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
            "url": issue.html_url,
            "author": issue.user.login if issue.user else "unknown",
            "labels": labels,
        }

    def iter_closed_pulls(self, repo_name: str) -> Iterator[Any]:
        repo = self._client.get_repo(repo_name)
        # Newest-first so each backfill run fills recent context first; older PRs in later runs.
        return iter(repo.get_pulls(state="closed", sort="updated", direction="desc"))

    def list_pull_requests(
        self,
        repo_name: str,
        *,
        state: str = "all",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compact PR refs, newest-first, bounded by the backfill window/cap.

        Cheap enumeration for the agent's backfill todo list: identity +
        title + state only. The agent hydrates each via
        ``github_get_pull_request`` when it needs the body/diff/comments.
        Iteration is updated-desc so the window check can short-circuit the
        moment it walks past the cutoff.
        """
        repo = self._client.get_repo(repo_name)
        since = backfill_window_since()
        cap = clamp_backfill_limit(limit)
        out: list[dict[str, Any]] = []
        pulls = repo.get_pulls(
            state=state if state in ("open", "closed", "all") else "all",
            sort="updated",
            direction="desc",
        )
        for pr in pulls:
            updated = getattr(pr, "updated_at", None)
            if since is not None and updated is not None and _aware(updated) < since:
                break
            out.append(
                {
                    "number": pr.number,
                    "title": pr.title,
                    "state": pr.state,
                    "merged": bool(getattr(pr, "merged_at", None)),
                    "created_at": pr.created_at.isoformat() if pr.created_at else None,
                    "updated_at": updated.isoformat() if updated else None,
                    "url": pr.html_url,
                    "author": pr.user.login if pr.user else None,
                }
            )
            if len(out) >= cap:
                break
        return out

    def list_issues(
        self,
        repo_name: str,
        *,
        state: str = "all",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compact issue refs, newest-first, bounded by the backfill window/cap.

        Excludes pull requests (GitHub's REST API folds PRs into issues; PRs
        come from :meth:`list_pull_requests`). The agent hydrates each via
        ``github_get_issue``.
        """
        repo = self._client.get_repo(repo_name)
        since = backfill_window_since()
        cap = clamp_backfill_limit(limit)
        out: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "state": state if state in ("open", "closed", "all") else "all",
            "sort": "updated",
            "direction": "desc",
        }
        if since is not None:
            kwargs["since"] = since
        for issue in repo.get_issues(**kwargs):
            if getattr(issue, "pull_request", None) is not None:
                continue  # a PR masquerading as an issue
            updated = getattr(issue, "updated_at", None)
            labels: List[dict[str, Any]] = []
            for lb in issue.labels or []:
                labels.append({"name": lb.name})
            out.append(
                {
                    "number": issue.number,
                    "title": issue.title,
                    "state": issue.state,
                    "created_at": issue.created_at.isoformat()
                    if issue.created_at
                    else None,
                    "updated_at": updated.isoformat() if updated else None,
                    "url": issue.html_url,
                    "author": issue.user.login if issue.user else None,
                    "labels": labels,
                    "comments": getattr(issue, "comments", None),
                }
            )
            if len(out) >= cap:
                break
        return out


def _aware(dt: datetime) -> datetime:
    """PyGithub historically returned naive UTC datetimes; normalize to aware."""
    from datetime import timezone

    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
