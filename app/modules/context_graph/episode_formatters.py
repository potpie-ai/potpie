"""Pure functions to format GitHub PR and commit data into Graphiti episode text and source_id."""

from typing import Any


def _pr_author(pr_data: dict) -> str:
    """Get author from PR dict (provider returns 'author', webhook has 'user')."""
    return (
        pr_data.get("author")
        or (pr_data.get("user", {}) or {}).get("login")
        or "unknown"
    )


def _pr_branches(pr_data: dict) -> tuple[str, str]:
    """Get head and base branch from PR dict."""
    head = pr_data.get("head_branch") or (pr_data.get("head") or {}).get("ref") or "?"
    base = pr_data.get("base_branch") or (pr_data.get("base") or {}).get("ref") or "?"
    return head, base


def _pr_number(pr_data: dict) -> int:
    """Get PR number from PR dict."""
    return int(pr_data.get("number", 0) or 0)


def format_github_pr_episode(pr_data: dict, event_type: str) -> dict[str, Any]:
    """Format a GitHub PR into episode fields for Graphiti.

    Args:
        pr_data: Dict with keys from GitHubProvider.get_pull_request() or webhook
                 (number, title, body, head_branch/base_branch or head/base.ref, author or user.login).
        event_type: "opened" or "merged".

    Returns:
        Dict with name, episode_body, source_description, source_id.
    """
    n = _pr_number(pr_data)
    if n <= 0:
        raise ValueError("pr_data must contain 'number'")
    head, base = _pr_branches(pr_data)
    author = _pr_author(pr_data)
    title = (pr_data.get("title") or "").strip() or "(no title)"
    body = (pr_data.get("body") or "").strip() or "(no description)"
    files = pr_data.get("files") or []
    file_list = ", ".join(
        f.get("filename", str(f)) for f in (files if isinstance(files, list) else [])
    ) or "—"

    source_id = f"pr_{n}_{event_type}"
    name = f"PR #{n} ({event_type})"
    source_description = f"GitHub PR #{n} {event_type}"

    episode_body = (
        f"PR #{n}: {title}. "
        f"Branch: {head} → {base}. "
        f"Author: {author}. "
        f"Files changed: {file_list}. "
        f"Description: {body}"
    )
    return {
        "name": name,
        "episode_body": episode_body,
        "source_description": source_description,
        "source_id": source_id,
    }


def format_github_commit_episode(commit_data: dict, branch_name: str) -> dict[str, Any]:
    """Format a GitHub commit into episode fields for Graphiti.

    Args:
        commit_data: Dict with sha, message (or commit.message), author (or commit.author.login).
        branch_name: Branch this commit belongs to.

    Returns:
        Dict with name, episode_body, source_description, source_id.
    """
    sha = (commit_data.get("sha") or commit_data.get("id") or "")[:12] or "unknown"
    message = (
        (commit_data.get("message") or (commit_data.get("commit") or {}).get("message") or "")
        .strip()
        .split("\n")[0]
        or "(no message)"
    )
    author = "unknown"
    if "author" in commit_data and isinstance(commit_data["author"], dict):
        author = commit_data["author"].get("login", author)
    elif "commit" in commit_data and isinstance(commit_data["commit"], dict):
        author = (commit_data["commit"].get("author") or {}).get("name", author)

    source_id = f"commit_{sha}"
    name = f"Commit {sha}"
    source_description = f"GitHub commit {sha} on {branch_name}"

    episode_body = (
        f"Commit {sha}: {message}. "
        f"Branch: {branch_name}. "
        f"Author: {author}."
    )
    return {
        "name": name,
        "episode_body": episode_body,
        "source_description": source_description,
        "source_id": source_id,
    }
