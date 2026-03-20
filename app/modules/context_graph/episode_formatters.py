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


def _format_comment_list(
    items: list,
    *,
    body_key: str = "body",
    author_key: str = "user",
    date_key: str = "created_at",
    extra_keys: list[str] | None = None,
) -> str:
    """Format a list of comment-like dicts into a single string for episode body."""
    if not items:
        return ""
    lines = []
    for i, c in enumerate(items if isinstance(items, list) else [], 1):
        if isinstance(c, str):
            lines.append(c)
            continue
        body = (c.get(body_key) or "").strip() or "(no body)"
        author = "unknown"
        if author_key in c and c[author_key]:
            u = c[author_key]
            author = u.get("login", u.get("name", author)) if isinstance(u, dict) else str(u)
        date_str = ""
        if date_key in c and c[date_key]:
            date_str = f" [{c.get(date_key)}]" if c.get(date_key) else ""
        extra = ""
        if extra_keys:
            parts = [f"{k}: {c.get(k)}" for k in extra_keys if c.get(k)]
            if parts:
                extra = " " + " ".join(parts)
        lines.append(f"Comment {i} ({author}{date_str}): {body}{extra}")
    return "\n".join(lines)


def format_github_pr_episode(pr_data: dict, event_type: str) -> dict[str, Any]:
    """Format a GitHub PR into episode fields for Graphiti.

    Includes PR body, optional comments, review comments, and commit messages when provided.

    Args:
        pr_data: Dict with number, title, body, head_branch/base_branch, author, files.
                 Optional: comments (list of {body, user: {login}}, created_at),
                 review_comments (list of {body, user, path, created_at}),
                 commit_messages (list of {sha, message, author} or list of strings).
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

    # Optional: issue comments on the PR
    comments = pr_data.get("comments") or []
    if comments:
        episode_body += "\n\nComments:\n" + _format_comment_list(comments)

    # Optional: review comments (inline)
    review_comments = pr_data.get("review_comments") or []
    if review_comments:
        episode_body += "\n\nReview comments:\n" + _format_comment_list(
            review_comments, extra_keys=["path"]
        )

    # Optional: commit messages in this PR
    commit_messages = pr_data.get("commit_messages") or []
    if commit_messages:
        lines = []
        for i, cm in enumerate(commit_messages if isinstance(commit_messages, list) else [], 1):
            if isinstance(cm, str):
                lines.append(f"Commit {i}: {cm}")
            elif isinstance(cm, dict):
                sha = (cm.get("sha") or "")[:12] or "?"
                msg = (cm.get("message") or "").strip() or "(no message)"
                by = cm.get("author") or "?"
                lines.append(f"Commit {i} ({sha}, {by}): {msg}")
            else:
                lines.append(f"Commit {i}: {cm}")
        episode_body += "\n\nCommits in this PR:\n" + "\n".join(lines)

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
