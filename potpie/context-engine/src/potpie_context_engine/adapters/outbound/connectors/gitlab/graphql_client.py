"""Optional GraphQL fast path for GitLab merge-request review history.

REST v4 is the connector's primary and complete transport (see
``api_client.py``). This module exists for one narrow win: assembling an
MR's full review history — approvals plus every resolvable discussion
thread with its inline positions — costs three REST round-trips
(``/approvals`` + ``/discussions`` + ``/notes``) but only one GraphQL
query.

GitLab's GraphQL schema evolves between releases, so this path is
strictly best-effort: :meth:`merge_request_review_history` returns
``None`` (or raises, which the caller swallows) whenever the response
does not have the expected shape, and
:meth:`GitLabRestSourceControl.get_merge_request_review_history` falls
back to REST. Nothing in the connector depends on GraphQL being
available.

Set ``CONTEXT_ENGINE_GITLAB_GRAPHQL=0`` to disable the fast path.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from potpie_context_engine.adapters.outbound.cli_auth.http import (
    AuthHttpClient,
    AuthHttpError,
    HttpClient,
)

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 30.0

# One query, one round-trip. ``iid`` is a String in GitLab's schema even
# though it is an integer in REST — passing an int is a hard schema error.
_REVIEW_HISTORY_QUERY = """
query PotpieMergeRequestReviewHistory($fullPath: ID!, $iid: String!) {
  project(fullPath: $fullPath) {
    mergeRequest(iid: $iid) {
      iid
      title
      state
      mergedAt
      approved
      approvedBy { nodes { username } }
      discussions(first: 100) {
        nodes {
          id
          resolved
          resolvable
          notes(first: 100) {
            nodes {
              id
              body
              system
              createdAt
              resolvable
              resolved
              author { username }
              position { newPath oldPath newLine oldLine }
            }
          }
        }
      }
    }
  }
}
"""


def graphql_enabled() -> bool:
    """True unless the operator turned the fast path off."""
    raw = os.getenv("CONTEXT_ENGINE_GITLAB_GRAPHQL", "").strip().lower()
    return raw not in ("0", "false", "no", "off")


class GitLabGraphQLClient:
    """Minimal GraphQL caller for a single GitLab instance."""

    def __init__(
        self,
        instance_url: str,
        token: str,
        *,
        http: HttpClient | None = None,
    ) -> None:
        self._endpoint = f"{instance_url.rstrip('/')}/api/graphql"
        self._headers = {
            # A PAT is accepted as a bearer token on the GraphQL endpoint;
            # PRIVATE-TOKEN is REST-only.
            "Authorization": f"Bearer {token.strip()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._http = http or AuthHttpClient(timeout=_HTTP_TIMEOUT)
        self._owns_http = http is None

    def close(self) -> None:
        if self._owns_http:
            close = getattr(self._http, "close", None)
            if callable(close):
                close()

    def _execute(self, query: str, variables: dict[str, Any]) -> dict[str, Any] | None:
        try:
            response = self._http.post(
                self._endpoint,
                headers=self._headers,
                json={"query": query, "variables": variables},
            )
        except AuthHttpError as exc:
            raise RuntimeError(f"GitLab GraphQL request failed: {exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(f"GitLab GraphQL returned HTTP {response.status_code}")
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        if payload.get("errors"):
            raise RuntimeError("GitLab GraphQL returned errors")
        data = payload.get("data")
        return data if isinstance(data, dict) else None

    def merge_request_review_history(
        self, project: str, iid: int
    ) -> dict[str, Any] | None:
        """Approvals + discussion threads for one MR, or ``None`` if unusable."""
        if not graphql_enabled():
            return None
        data = self._execute(
            _REVIEW_HISTORY_QUERY,
            {"fullPath": str(project).strip().strip("/"), "iid": str(int(iid))},
        )
        project_node = (data or {}).get("project")
        if not isinstance(project_node, dict):
            return None
        mr = project_node.get("mergeRequest")
        if not isinstance(mr, dict):
            return None
        return {
            "approvals": _map_approvals(mr),
            "discussions": _map_discussions(mr),
            "notes": _map_system_notes(mr),
        }


def _nodes(container: Any) -> list[dict[str, Any]]:
    if not isinstance(container, dict):
        return []
    nodes = container.get("nodes")
    if not isinstance(nodes, list):
        return []
    return [n for n in nodes if isinstance(n, dict)]


def _author(note: dict[str, Any]) -> str | None:
    author = note.get("author")
    if isinstance(author, dict) and author.get("username"):
        return str(author["username"])
    return None


def _map_approvals(mr: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": True,
        "approved": bool(mr.get("approved")),
        "approvals_required": None,
        "approvals_left": None,
        "approved_by": [
            str(n["username"])
            for n in _nodes(mr.get("approvedBy"))
            if n.get("username")
        ],
    }


def _map_discussions(mr: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten discussion threads into the REST ``discussions`` note shape."""
    out: list[dict[str, Any]] = []
    for discussion in _nodes(mr.get("discussions")):
        discussion_id = discussion.get("id")
        for note in _nodes(discussion.get("notes")):
            if note.get("system"):
                continue
            position = note.get("position")
            position = position if isinstance(position, dict) else {}
            out.append(
                {
                    "id": note.get("id"),
                    "discussion_id": discussion_id,
                    "type": "DiffNote" if position else None,
                    "body": note.get("body"),
                    "user": {"login": _author(note)} if _author(note) else None,
                    "path": position.get("newPath") or position.get("oldPath"),
                    "line": position.get("newLine") or position.get("oldLine"),
                    "resolvable": note.get("resolvable"),
                    "resolved": note.get("resolved"),
                    "resolved_by": None,
                    "created_at": note.get("createdAt"),
                }
            )
    return out


def _map_system_notes(mr: dict[str, Any]) -> list[dict[str, Any]]:
    """System notes carry the review-action trail on GitLab CE."""
    out: list[dict[str, Any]] = []
    for discussion in _nodes(mr.get("discussions")):
        for note in _nodes(discussion.get("notes")):
            out.append(
                {
                    "id": note.get("id"),
                    "body": note.get("body"),
                    "user": {"login": _author(note)} if _author(note) else None,
                    "system": bool(note.get("system")),
                    "created_at": note.get("createdAt"),
                    "updated_at": None,
                    "resolvable": note.get("resolvable"),
                    "resolved": note.get("resolved"),
                }
            )
    return out


__all__ = ["GitLabGraphQLClient", "graphql_enabled"]
