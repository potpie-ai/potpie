"""PyGithub implementation of SourceControlPort."""

from __future__ import annotations

import logging
from typing import Any, Iterator, List

from github import Github

from domain.ports.source_control import SourceControlPort

logger = logging.getLogger(__name__)


class PyGithubSourceControl(SourceControlPort):
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

    def get_pull_request_commits(self, repo_name: str, pr_number: int) -> list[dict[str, Any]]:
        repo = self._client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        commits = []
        for commit in pr.get_commits():
            commit_author = None
            if getattr(commit, "author", None):
                commit_author = commit.author.login
            elif getattr(commit, "commit", None) and getattr(commit.commit, "author", None):
                commit_author = commit.commit.author.name

            committed_at = None
            if getattr(commit, "commit", None) and getattr(commit.commit, "author", None):
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
                    "created_at": comment.created_at.isoformat() if comment.created_at else None,
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
                    "created_at": comment.created_at.isoformat() if comment.created_at else None,
                }
            )
        return comments

    def get_issue(self, repo_name: str, issue_number: int) -> dict[str, Any]:
        repo = self._client.get_repo(repo_name)
        issue = repo.get_issue(issue_number)
        return {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "state": issue.state,
            "created_at": issue.created_at.isoformat() if issue.created_at else None,
            "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
            "url": issue.html_url,
            "author": issue.user.login if issue.user else "unknown",
        }

    def iter_closed_pulls(self, repo_name: str) -> Iterator[Any]:
        repo = self._client.get_repo(repo_name)
        return iter(repo.get_pulls(state="closed", sort="updated", direction="asc"))
