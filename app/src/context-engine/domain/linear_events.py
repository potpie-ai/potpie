"""Normalized Linear issue + comment models for context-engine ingestion.

Linear's GraphQL payloads and webhook payloads overlap but are not identical
(webhooks drop some nested connections, pad a few timestamps). The normalized
types here are the lingua franca the deterministic planner and resolver
consume, so a GraphQL sync, a webhook, and a test fixture all drive the same
downstream code.

Keep these flat and serializable — no ORM coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable


LINEAR_ISSUE_ACTIONS = frozenset(
    {
        "create",
        "update",
        "state_change",
        "comment_added",
        "remove",
    }
)
"""Canonical actions the planner understands.

Linear native webhook actions (``create``, ``update``, ``remove``) map 1:1
except that we synthesize ``state_change`` when an ``update`` changed
``stateId``, and ``comment_added`` when the event is a comment create.
"""


@dataclass(slots=True)
class LinearPerson:
    """A Linear user (assignee, creator, comment author)."""

    id: str
    name: str
    email: str | None = None


@dataclass(slots=True)
class LinearLabel:
    """A Linear label attached to an issue."""

    id: str
    name: str


@dataclass(slots=True)
class LinearState:
    """A Linear workflow state: display name plus a coarse state type."""

    name: str
    type: str | None = None  # "backlog" | "unstarted" | "started" | "completed" | "canceled"


@dataclass(slots=True)
class LinearComment:
    """One comment on a Linear issue."""

    id: str
    body: str
    created_at: datetime | None = None
    author: LinearPerson | None = None


@dataclass(slots=True)
class LinearIssue:
    """Normalized Linear issue snapshot."""

    id: str
    identifier: str  # e.g. "ENG-123"
    title: str
    description: str = ""
    url: str | None = None
    state: LinearState | None = None
    priority: int | None = None
    team_id: str | None = None
    project_id: str | None = None
    creator: LinearPerson | None = None
    assignee: LinearPerson | None = None
    labels: list[LinearLabel] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    canceled_at: datetime | None = None
    comments: list[LinearComment] = field(default_factory=list)


@dataclass(slots=True)
class LinearIssueEvent:
    """A normalized issue-centric event (create, update, state_change, comment)."""

    action: str  # one of LINEAR_ISSUE_ACTIONS
    issue: LinearIssue
    comment: LinearComment | None = None
    previous_state: LinearState | None = None
    occurred_at: datetime | None = None


def parse_linear_datetime(value: Any) -> datetime | None:
    """Parse Linear ISO8601 timestamps (accepts ``Z`` and naive strings)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _person(value: Any) -> LinearPerson | None:
    if not isinstance(value, dict):
        return None
    pid = value.get("id")
    name = value.get("name") or value.get("displayName") or value.get("email")
    if not pid or not name:
        return None
    return LinearPerson(id=str(pid), name=str(name), email=value.get("email"))


def _state(value: Any) -> LinearState | None:
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    if not name:
        return None
    return LinearState(name=str(name), type=value.get("type"))


def _labels(value: Any) -> list[LinearLabel]:
    nodes: Iterable[Any]
    if isinstance(value, dict) and isinstance(value.get("nodes"), list):
        nodes = value["nodes"]
    elif isinstance(value, list):
        nodes = value
    else:
        return []
    out: list[LinearLabel] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        lid = node.get("id")
        name = node.get("name")
        if not lid or not name:
            continue
        out.append(LinearLabel(id=str(lid), name=str(name)))
    return out


def _comments(value: Any) -> list[LinearComment]:
    nodes: Iterable[Any]
    if isinstance(value, dict) and isinstance(value.get("nodes"), list):
        nodes = value["nodes"]
    elif isinstance(value, list):
        nodes = value
    else:
        return []
    out: list[LinearComment] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        cid = node.get("id")
        body = node.get("body") or ""
        if not cid:
            continue
        out.append(
            LinearComment(
                id=str(cid),
                body=str(body),
                created_at=parse_linear_datetime(node.get("createdAt")),
                author=_person(node.get("user") or node.get("author")),
            )
        )
    return out


def linear_issue_from_payload(payload: dict[str, Any]) -> LinearIssue:
    """Build a :class:`LinearIssue` from either a GraphQL or webhook payload."""
    return LinearIssue(
        id=str(payload.get("id") or ""),
        identifier=str(payload.get("identifier") or ""),
        title=str(payload.get("title") or ""),
        description=str(payload.get("description") or ""),
        url=payload.get("url") or None,
        state=_state(payload.get("state")),
        priority=(
            int(payload["priority"])
            if isinstance(payload.get("priority"), (int, float))
            else None
        ),
        team_id=(
            str(payload["teamId"])
            if payload.get("teamId")
            else (
                str(payload["team"]["id"])
                if isinstance(payload.get("team"), dict) and payload["team"].get("id")
                else None
            )
        ),
        project_id=(
            str(payload["projectId"])
            if payload.get("projectId")
            else (
                str(payload["project"]["id"])
                if isinstance(payload.get("project"), dict) and payload["project"].get("id")
                else None
            )
        ),
        creator=_person(payload.get("creator")),
        assignee=_person(payload.get("assignee")),
        labels=_labels(payload.get("labels")),
        created_at=parse_linear_datetime(payload.get("createdAt")),
        updated_at=parse_linear_datetime(payload.get("updatedAt")),
        completed_at=parse_linear_datetime(payload.get("completedAt")),
        canceled_at=parse_linear_datetime(payload.get("canceledAt")),
        comments=_comments(payload.get("comments")),
    )
