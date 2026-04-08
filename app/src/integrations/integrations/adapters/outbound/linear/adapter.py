"""IssueTrackerPort implementation for Linear (GraphQL)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator

from integrations.adapters.outbound.linear.graphql_client import linear_graphql


@dataclass
class _IssueRef:
    id: str
    identifier: str
    updated_at: datetime | None


_ISSUES_PAGE = """
query TeamIssues($teamId: ID!, $after: String) {
  issues(
    filter: { team: { id: { eq: $teamId } } }
    first: 50
    after: $after
    includeArchived: false
  ) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id
      identifier
      title
      updatedAt
    }
  }
}
"""

_ISSUE_DETAIL = """
query IssueDetail($id: ID!) {
  issue(id: $id) {
    id
    identifier
    title
    description
    url
    createdAt
    updatedAt
    state { name }
    assignee { id name email }
    labels { nodes { id name } }
    comments {
      nodes {
        id
        body
        createdAt
        user { id name }
      }
    }
  }
}
"""


class LinearIssueTrackerAdapter:
    """Linear GraphQL backed adapter (not a Protocol subclass at runtime; structurally matches)."""

    def __init__(self, access_token: str) -> None:
        self._token = access_token

    def iter_issues(
        self,
        *,
        scope: dict[str, Any],
        updated_after: datetime | None = None,
    ) -> Iterator[Any]:
        team_id = scope.get("team_id")
        if not team_id:
            return
        after: str | None = None
        while True:
            data = linear_graphql(
                self._token, _ISSUES_PAGE, {"teamId": team_id, "after": after}
            )
            conn = (data or {}).get("issues") or {}
            nodes = conn.get("nodes") or []
            for n in nodes:
                if not isinstance(n, dict):
                    continue
                raw_updated = n.get("updatedAt")
                updated = None
                if isinstance(raw_updated, str):
                    try:
                        updated = datetime.fromisoformat(
                            raw_updated.replace("Z", "+00:00")
                        )
                    except ValueError:
                        updated = None
                if updated_after and updated:
                    ua = updated_after if updated_after.tzinfo else updated_after.replace(tzinfo=updated.tzinfo)
                    if updated <= ua:
                        continue
                node_id = n.get("id")
                if not node_id:
                    continue
                yield _IssueRef(
                    id=node_id,
                    identifier=n.get("identifier") or node_id,
                    updated_at=updated,
                )
            page = conn.get("pageInfo") or {}
            if not page.get("hasNextPage"):
                break
            after = page.get("endCursor")
            if not after:
                break

    def get_issue(self, *, scope: dict[str, Any], issue_id: str) -> dict[str, Any]:
        _ = scope
        data = linear_graphql(self._token, _ISSUE_DETAIL, {"id": issue_id})
        issue = (data or {}).get("issue")
        if not isinstance(issue, dict):
            raise LookupError(f"Linear issue not found: {issue_id}")
        return issue

    def get_issue_comments(
        self,
        *,
        scope: dict[str, Any],
        issue_id: str,
    ) -> list[dict[str, Any]]:
        issue = self.get_issue(scope=scope, issue_id=issue_id)
        comments = issue.get("comments") or {}
        nodes = comments.get("nodes") if isinstance(comments, dict) else None
        if isinstance(nodes, list):
            return [n for n in nodes if isinstance(n, dict)]
        return []
