"""Build rich episode text for Graphiti ingestion."""

from datetime import datetime
from typing import Any, Optional


def _safe_iso(dt_value: Optional[Any]) -> str:
    if dt_value is None:
        return ""
    if isinstance(dt_value, datetime):
        return dt_value.isoformat()
    return str(dt_value)


def _to_datetime(value: Optional[Any]) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            pass
    return datetime.utcnow()


def build_pr_episode(
    pr_data: dict[str, Any],
    commits: list[dict[str, Any]],
    review_threads: list[dict[str, Any]],
    linked_issues: list[dict[str, Any]],
    issue_comments: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    number = pr_data.get("number")
    title = pr_data.get("title") or ""
    author = pr_data.get("author") or "unknown"
    head_branch = pr_data.get("head_branch") or ""
    base_branch = pr_data.get("base_branch") or ""
    merged_at = _safe_iso(pr_data.get("merged_at"))
    body = (pr_data.get("body") or "").strip()
    files = pr_data.get("files") or []
    labels = pr_data.get("labels") or []
    milestone = pr_data.get("milestone")

    files_changed = ", ".join(f.get("filename", "") for f in files if f.get("filename")) or "None"

    related_issue_lines = []
    for issue in linked_issues:
        issue_title = issue.get("title") or ""
        issue_body = (issue.get("body") or "").strip().replace("\n", " ")
        issue_excerpt = issue_body[:280]
        related_issue_lines.append(f"- #{issue.get('number')}: {issue_title} | {issue_excerpt}")
    related_issues_section = "\n".join(related_issue_lines) if related_issue_lines else "- None"

    commit_lines = []
    for commit in commits:
        message = (commit.get("message") or "").strip().splitlines()[0] if commit.get("message") else ""
        commit_lines.append(f"- {commit.get('sha')}: {message}")
    commits_section = "\n".join(commit_lines) if commit_lines else "- None"

    discussion_lines = []
    for thread in review_threads:
        discussion_lines.append(
            f"- File: {thread.get('path')}, Line: {thread.get('line')}, Diff: {thread.get('diff_hunk') or ''}"
        )
        for comment in thread.get("comments", []):
            discussion_lines.append(
                f"  - {comment.get('author') or 'unknown'}: {(comment.get('body') or '').strip()}"
            )
    discussions_section = "\n".join(discussion_lines) if discussion_lines else "- None"

    issue_comment_lines = []
    for c in issue_comments or []:
        issue_comment_lines.append(f"- {(c.get('user') or {}).get('login')}: {(c.get('body') or '').strip()}")
    issue_comments_section = "\n".join(issue_comment_lines) if issue_comment_lines else "- None"

    labels_text = ", ".join(l.get("name", "") if isinstance(l, dict) else str(l) for l in labels if l) or "None"
    milestone_text = (
        milestone.get("title")
        if isinstance(milestone, dict)
        else (str(milestone) if milestone else "None")
    )

    episode_body = (
        f"PR #{number}: {title}\n\n"
        f"Author: {author}\n"
        f"From branch: {head_branch}\n"
        f"To branch: {base_branch}\n"
        f"Merged at: {merged_at}\n\n"
        f"Files changed: {files_changed}\n\n"
        "WHY THIS CHANGE WAS MADE:\n"
        f"{body or 'No PR description provided.'}\n\n"
        "RELATED ISSUES:\n"
        f"{related_issues_section}\n\n"
        "COMMITS:\n"
        f"{commits_section}\n\n"
        "REVIEW DISCUSSIONS:\n"
        f"{discussions_section}\n\n"
        "PR ISSUE COMMENTS:\n"
        f"{issue_comments_section}\n\n"
        f"Labels: {labels_text}\n"
        f"Milestone/Feature: {milestone_text}\n"
    )

    return {
        "name": f"pr_{number}_merged",
        "episode_body": episode_body,
        "source_description": f"GitHub Pull Request #{number}",
        "source_id": f"pr_{number}_merged",
        "reference_time": _to_datetime(pr_data.get("merged_at") or pr_data.get("updated_at")),
    }


def build_commit_episode(commit_data: dict[str, Any], branch: str) -> dict[str, Any]:
    sha = commit_data.get("sha") or "unknown"
    message = (commit_data.get("message") or "").strip()
    author = commit_data.get("author") or "unknown"
    committed_at = commit_data.get("committed_at")
    body = (
        f"Standalone commit {sha}\n\n"
        f"Author: {author}\n"
        f"Branch: {branch}\n"
        f"Committed at: {_safe_iso(committed_at)}\n\n"
        f"Message:\n{message}\n"
    )
    return {
        "name": f"commit_{sha}",
        "episode_body": body,
        "source_description": f"GitHub Commit {sha}",
        "source_id": f"commit_{sha}",
        "reference_time": _to_datetime(committed_at),
    }
