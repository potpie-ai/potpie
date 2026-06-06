"""Build Graphiti episode payloads from Linear issue payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def build_linear_issue_episode(
    issue: dict[str, Any],
    comments: list[dict[str, Any]],
) -> dict[str, Any]:
    identifier = issue.get("identifier") or issue.get("id") or "unknown"
    title = issue.get("title") or ""
    description = (issue.get("description") or "").strip()
    url = issue.get("url") or ""
    state = issue.get("state") or {}
    state_name = state.get("name") if isinstance(state, dict) else str(state)
    assignee = issue.get("assignee") or {}
    assignee_name = ""
    if isinstance(assignee, dict):
        assignee_name = assignee.get("name") or assignee.get("email") or ""

    labels = issue.get("labels") or {}
    label_nodes = []
    if isinstance(labels, dict):
        label_nodes = labels.get("nodes") or labels.get("edges") or []
    if isinstance(label_nodes, list):
        label_names = [
            (n.get("name") if isinstance(n, dict) else str(n))
            for n in label_nodes
            if n
        ]
    else:
        label_names = []

    comment_lines = []
    for c in comments:
        if not isinstance(c, dict):
            continue
        body = (c.get("body") or "").strip()
        user = c.get("user") or {}
        who = user.get("name") if isinstance(user, dict) else "unknown"
        comment_lines.append(f"- {who}: {body[:2000]}")

    comments_section = "\n".join(comment_lines) if comment_lines else "- None"

    episode_body = (
        f"Linear issue {identifier}: {title}\n\n"
        f"URL: {url}\n"
        f"State: {state_name}\n"
        f"Assignee: {assignee_name or 'None'}\n"
        f"Updated at: {issue.get('updatedAt', '')}\n\n"
        "DESCRIPTION:\n"
        f"{description or 'No description.'}\n\n"
        "LABELS:\n"
        f"{', '.join(label_names) if label_names else 'None'}\n\n"
        "COMMENTS:\n"
        f"{comments_section}\n\n"
        "LINKS (for cross-repo correlation):\n"
        f"{description}\n{comments_section}\n"
    )

    sid = str(issue.get("id") or identifier).replace("-", "_")
    return {
        "name": f"linear_issue_{sid}",
        "episode_body": episode_body,
        "source_description": f"Linear Issue {identifier}",
        "source_id": f"linear_issue_{issue.get('id', identifier)}",
        "reference_time": _parse_dt(issue.get("updatedAt") or issue.get("createdAt")),
    }
