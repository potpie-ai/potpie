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


@dataclass
class _ProjectRef:
    id: str
    name: str
    updated_at: datetime | None


@dataclass
class _DocumentRef:
    id: str
    title: str
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

# Connection fields (`labels`, `comments`) require pagination args on Linear's schema;
# omitting `first`/`last` can yield HTTP 400 from api.linear.app/graphql.
_ISSUE_DETAIL = """
query IssueDetail($id: String!) {
  issue(id: $id) {
    id
    identifier
    title
    description
    url
    priority
    createdAt
    updatedAt
    completedAt
    canceledAt
    state { id name type }
    team { id name }
    project { id name }
    creator { id name email }
    assignee { id name email }
    labels(first: 50) {
      nodes { id name }
    }
    comments(first: 100) {
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


# Projects/documents scope to a team via Linear's project filter idiom
# (mirrors the proven ``team: { id: { eq } }`` issue filter). Documents in
# Linear hang off projects, so they're reached through the project's
# accessible-teams edge; standalone/initiative docs are out of team scope by
# design (project docs are the bulk and the context-relevant set).
_PROJECTS_PAGE = """
query TeamProjects($teamId: ID!, $after: String) {
  projects(
    filter: { accessibleTeams: { some: { id: { eq: $teamId } } } }
    first: 50
    after: $after
  ) {
    pageInfo { hasNextPage endCursor }
    nodes { id name updatedAt }
  }
}
"""

_DOCUMENTS_PAGE = """
query TeamDocuments($teamId: ID!, $after: String) {
  documents(
    filter: { project: { accessibleTeams: { some: { id: { eq: $teamId } } } } }
    first: 50
    after: $after
  ) {
    pageInfo { hasNextPage endCursor }
    nodes { id title updatedAt }
  }
}
"""

_PROJECT_DETAIL = """
query ProjectDetail($id: String!) {
  project(id: $id) {
    id
    name
    description
    url
    state
    createdAt
    updatedAt
    startDate
    targetDate
    completedAt
    canceledAt
    lead { id name email }
    teams(first: 20) { nodes { id name } }
  }
}
"""

_DOCUMENT_DETAIL = """
query DocumentDetail($id: String!) {
  document(id: $id) {
    id
    title
    content
    url
    createdAt
    updatedAt
    creator { id name email }
    project { id name }
  }
}
"""


def _parse_iso(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _after_cutoff(updated: datetime | None, updated_after: datetime | None) -> bool:
    """True when ``updated`` is strictly newer than the ``updated_after`` cutoff.

    Mirrors ``iter_issues``' tz-reconciliation: a naive cutoff is read in the
    node's tz so aware/naive never collide.
    """
    if not updated_after or not updated:
        return True
    ua = (
        updated_after
        if updated_after.tzinfo
        else updated_after.replace(tzinfo=updated.tzinfo)
    )
    return updated > ua


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
                    ua = (
                        updated_after
                        if updated_after.tzinfo
                        else updated_after.replace(tzinfo=updated.tzinfo)
                    )
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

    def iter_projects(
        self,
        *,
        scope: dict[str, Any],
        updated_after: datetime | None = None,
    ) -> Iterator[Any]:
        """Enumerate the team's projects (cursor-paginated, same idiom as issues)."""
        team_id = scope.get("team_id")
        if not team_id:
            return
        after: str | None = None
        while True:
            data = linear_graphql(
                self._token, _PROJECTS_PAGE, {"teamId": team_id, "after": after}
            )
            conn = (data or {}).get("projects") or {}
            for n in conn.get("nodes") or []:
                if not isinstance(n, dict):
                    continue
                node_id = n.get("id")
                if not node_id:
                    continue
                updated = _parse_iso(n.get("updatedAt"))
                if not _after_cutoff(updated, updated_after):
                    continue
                yield _ProjectRef(
                    id=node_id,
                    name=n.get("name") or node_id,
                    updated_at=updated,
                )
            page = conn.get("pageInfo") or {}
            if not page.get("hasNextPage"):
                break
            after = page.get("endCursor")
            if not after:
                break

    def get_project(self, *, scope: dict[str, Any], project_id: str) -> dict[str, Any]:
        _ = scope
        data = linear_graphql(self._token, _PROJECT_DETAIL, {"id": project_id})
        project = (data or {}).get("project")
        if not isinstance(project, dict):
            raise LookupError(f"Linear project not found: {project_id}")
        return project

    def iter_documents(
        self,
        *,
        scope: dict[str, Any],
        updated_after: datetime | None = None,
    ) -> Iterator[Any]:
        """Enumerate the team's project documents (cursor-paginated)."""
        team_id = scope.get("team_id")
        if not team_id:
            return
        after: str | None = None
        while True:
            data = linear_graphql(
                self._token, _DOCUMENTS_PAGE, {"teamId": team_id, "after": after}
            )
            conn = (data or {}).get("documents") or {}
            for n in conn.get("nodes") or []:
                if not isinstance(n, dict):
                    continue
                node_id = n.get("id")
                if not node_id:
                    continue
                updated = _parse_iso(n.get("updatedAt"))
                if not _after_cutoff(updated, updated_after):
                    continue
                yield _DocumentRef(
                    id=node_id,
                    title=n.get("title") or node_id,
                    updated_at=updated,
                )
            page = conn.get("pageInfo") or {}
            if not page.get("hasNextPage"):
                break
            after = page.get("endCursor")
            if not after:
                break

    def get_document(
        self, *, scope: dict[str, Any], document_id: str
    ) -> dict[str, Any]:
        _ = scope
        data = linear_graphql(self._token, _DOCUMENT_DETAIL, {"id": document_id})
        document = (data or {}).get("document")
        if not isinstance(document, dict):
            raise LookupError(f"Linear document not found: {document_id}")
        return document
