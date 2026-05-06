"""Deterministic Linear issue planner using generic graph mutations.

Analog of :mod:`adapters.outbound.reconciliation.github_pr_plan`: the context
event payload ``{"action": ..., "issue": ..., "comment": ..., "previous_state":
...}`` is compiled into a :class:`ReconciliationPlan` with :class:`Issue`,
:class:`Person`, and :class:`Label` entities plus ``ASSIGNED_TO``,
``CREATED_BY``, ``HAS_LABEL``, ``HAS_COMMENT``, ``BELONGS_TO_TEAM``,
``EVIDENCED_BY`` edges.

Entity/edge shapes deliberately mirror the GitHub planner so downstream
ontology validation treats Linear issues as first-class ``Issue`` nodes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from domain.context_events import EventRef
from domain.graph_mutations import EdgeUpsert, EntityUpsert
from domain.linear_events import (
    LinearComment,
    LinearIssue,
    LinearIssueEvent,
    linear_issue_from_payload,
)
from domain.reconciliation import (
    EpisodeDraft,
    EvidenceRef,
    ReconciliationPlan,
    ReconciliationRequest,
)

from adapters.outbound.reconciliation.timeline_plan import (
    VERB_ASSIGNED,
    VERB_COMMENTED,
    VERB_OPENED_ISSUE,
    VERB_PERFORMED,
    VERB_STATE_CHANGED,
    build_timeline_mutations,
)


def build_linear_issue_plan(
    *,
    event_ref: EventRef,
    event: LinearIssueEvent,
) -> ReconciliationPlan:
    """Compile a Linear issue event into a canonical reconciliation plan."""
    issue = event.issue
    team_id = issue.team_id or "unknown"
    source_key = f"source-ref:linear:issue:{issue.identifier or issue.id}"
    issue_key = f"linear:issue:{issue.identifier or issue.id}"
    team_key = f"linear:team:{team_id}"
    observed_at = _iso(event.occurred_at or issue.updated_at or issue.created_at)
    title = issue.title or issue.identifier or "Linear issue"

    draft = EpisodeDraft(
        name=_episode_name(event),
        episode_body=_episode_body(event),
        source_description=f"Linear issue {issue.identifier}".strip(),
        reference_time=event.occurred_at or issue.updated_at or issue.created_at or datetime.now(timezone.utc),
    )

    entities: list[EntityUpsert] = [
        EntityUpsert(
            entity_key=team_key,
            labels=("Entity", "Team"),
            properties={
                "name": f"Linear team {team_id}",
                "provider": "linear",
                "provider_host": "linear.app",
                "team_id": team_id,
                "source_ref": source_key,
            },
        ),
        EntityUpsert(
            entity_key=source_key,
            labels=("Entity", "SourceReference"),
            properties={
                "name": f"Linear issue {issue.identifier}",
                "source_system": "linear",
                "source_kind": "linear_issue",
                "source_type": "linear_issue",
                "ref_type": "linear_issue",
                "external_id": issue.identifier or issue.id,
                "retrieval_uri": issue.url or "",
                "uri": issue.url or "",
                "source_ref": source_key,
                "observed_at": observed_at,
                "verification_state": "unverified",
                "sync_status": "needs_resync",
            },
        ),
        EntityUpsert(
            entity_key=issue_key,
            labels=("Entity", "Issue"),
            properties={
                "name": title,
                "title": title,
                "summary": issue.description or "",
                "description": issue.description or "",
                "identifier": issue.identifier,
                "issue_id": issue.id,
                "status": _status_value(issue),
                "state_name": issue.state.name if issue.state else None,
                "state_type": issue.state.type if issue.state else None,
                "priority": issue.priority,
                "team_id": team_id,
                "team_key": team_key,
                "url": issue.url or "",
                "source_ref": source_key,
                "observed_at": observed_at,
                "created_at": _iso(issue.created_at),
                "updated_at": _iso(issue.updated_at),
                "completed_at": _iso(issue.completed_at),
                "canceled_at": _iso(issue.canceled_at),
            },
        ),
    ]
    edges: list[EdgeUpsert] = [
        EdgeUpsert("EVIDENCED_BY", issue_key, source_key, {"source_ref": source_key}),
        EdgeUpsert(
            "BELONGS_TO_TEAM",
            issue_key,
            team_key,
            {"source_ref": source_key, "valid_from": observed_at},
        ),
    ]

    if issue.creator:
        creator_key = _person_key(issue.creator.id)
        entities.append(_person_entity(issue.creator, source_key, observed_at))
        edges.append(
            EdgeUpsert(
                "CREATED_BY",
                issue_key,
                creator_key,
                {"source_ref": source_key, "valid_from": observed_at},
            )
        )
    if issue.assignee:
        assignee_key = _person_key(issue.assignee.id)
        entities.append(_person_entity(issue.assignee, source_key, observed_at))
        edges.append(
            EdgeUpsert(
                "ASSIGNED_TO",
                issue_key,
                assignee_key,
                {"source_ref": source_key, "valid_from": observed_at},
            )
        )
    for label in issue.labels:
        label_key = f"linear:label:{team_id}:{label.id}"
        entities.append(
            EntityUpsert(
                entity_key=label_key,
                labels=("Entity", "Label"),
                properties={
                    "name": label.name,
                    "label_id": label.id,
                    "team_id": team_id,
                    "source_ref": source_key,
                    "observed_at": observed_at,
                },
            )
        )
        edges.append(
            EdgeUpsert(
                "HAS_LABEL",
                issue_key,
                label_key,
                {"source_ref": source_key, "valid_from": observed_at},
            )
        )

    if event.action == "comment_added" and event.comment is not None:
        comment_entity, comment_edges = _comment_entity(
            event.comment, issue_key, source_key, observed_at
        )
        entities.append(comment_entity)
        edges.extend(comment_edges)

    # --- Timeline subgraph -----------------------------------------------
    verb, actor_key, summary, activity_suffix = _timeline_params_for_event(
        event, issue_key, team_key
    )
    touched = [issue_key, team_key]
    if event.action == "comment_added" and event.comment is not None:
        touched.append(f"linear:comment:{event.comment.id}")
    t_entities, t_edges = build_timeline_mutations(
        pot_id=event_ref.pot_id,
        verb=verb,
        occurred_at=observed_at,
        summary=summary[:300],
        source_ref_key=source_key,
        actor_key=actor_key,
        touched_entity_keys=touched,
        activity_suffix=activity_suffix,
        extra_properties={
            "provider": "linear",
            "issue_identifier": issue.identifier,
            "action": event.action,
        },
        confidence=0.9,
    )
    entities.extend(t_entities)
    edges.extend(t_edges)

    return ReconciliationPlan(
        event_ref=event_ref,
        summary=_plan_summary(event),
        episodes=[draft],
        entity_upserts=_dedupe_entities(entities),
        edge_upserts=_dedupe_edges(edges),
        evidence=[
            EvidenceRef(
                kind="source_ref",
                ref=source_key,
                metadata={
                    "provider": "linear",
                    "identifier": issue.identifier,
                    "action": event.action,
                },
            )
        ],
        confidence=0.9,
    )


class LinearIssuePlannerAgent:
    """Deterministic reconciliation agent for Linear issue events.

    The event payload must match
    :func:`normalize_linear_webhook.payload_from_webhook` — an ``action`` string
    plus an ``issue`` dict in Linear's native shape, an optional ``comment``
    dict, and an optional ``previous_state``.
    """

    def run_reconciliation(self, request: ReconciliationRequest) -> ReconciliationPlan:
        p = request.event.payload or {}
        action = str(p.get("action") or request.event.action or "update").lower()
        issue_payload = p.get("issue") or {}
        if not isinstance(issue_payload, dict):
            raise ValueError("linear planner: payload.issue must be a dict")
        issue = linear_issue_from_payload(issue_payload)

        comment_payload = p.get("comment")
        comment = None
        if isinstance(comment_payload, dict) and comment_payload.get("id"):
            from domain.linear_events import _comments  # local import keeps helper internal
            parsed = _comments([comment_payload])
            comment = parsed[0] if parsed else None

        prev_state = None
        prev_payload = p.get("previous_state")
        if isinstance(prev_payload, dict):
            from domain.linear_events import _state

            prev_state = _state(prev_payload)

        event = LinearIssueEvent(
            action=action,
            issue=issue,
            comment=comment,
            previous_state=prev_state,
            occurred_at=request.event.occurred_at,
        )
        ref = EventRef(
            event_id=request.event.event_id,
            source_system=request.event.source_system,
            pot_id=request.pot_id,
        )
        return build_linear_issue_plan(event_ref=ref, event=event)

    def capability_metadata(self) -> dict[str, Any]:
        return {"agent": "linear_issue_planner", "version": "1"}


def _timeline_params_for_event(
    event: LinearIssueEvent,
    issue_key: str,
    team_key: str,
) -> tuple[str, str | None, str, str]:
    """Map a LinearIssueEvent.action to (verb, actor_key, summary, suffix)."""
    _ = issue_key, team_key
    issue = event.issue
    ident = issue.identifier or issue.id or "unknown"
    action = (event.action or "").lower()
    if action == "comment_added" and event.comment is not None:
        author = event.comment.author.name if event.comment.author else "someone"
        actor_key = (
            _person_key(event.comment.author.id) if event.comment.author else None
        )
        return (
            VERB_COMMENTED,
            actor_key,
            f"{author} commented on Linear {ident}",
            f"linear:{ident}:comment:{event.comment.id}",
        )
    if action == "state_change":
        prev = event.previous_state.name if event.previous_state else "?"
        cur = issue.state.name if issue.state else "?"
        actor_key = _person_key(issue.assignee.id) if issue.assignee else None
        actor_name = issue.assignee.name if issue.assignee else "Linear"
        return (
            VERB_STATE_CHANGED,
            actor_key,
            f"{actor_name}: Linear {ident} moved {prev} -> {cur}",
            f"linear:{ident}:state:{cur}",
        )
    if action in {"assigned", "reassigned"}:
        actor_key = _person_key(issue.assignee.id) if issue.assignee else None
        actor_name = issue.assignee.name if issue.assignee else "someone"
        return (
            VERB_ASSIGNED,
            actor_key,
            f"Linear {ident} assigned to {actor_name}",
            f"linear:{ident}:assign",
        )
    if action in {"create", "created", "opened"}:
        actor_key = _person_key(issue.creator.id) if issue.creator else None
        actor_name = issue.creator.name if issue.creator else "someone"
        return (
            VERB_OPENED_ISSUE,
            actor_key,
            f"{actor_name} opened Linear {ident}: {issue.title or ''}".strip(),
            f"linear:{ident}:open",
        )
    actor_key = _person_key(issue.assignee.id) if issue.assignee else None
    return (
        VERB_PERFORMED,
        actor_key,
        f"Linear {ident} action {action}",
        f"linear:{ident}:{action}",
    )


def _plan_summary(event: LinearIssueEvent) -> str:
    issue = event.issue
    ident = issue.identifier or issue.id or "unknown"
    if event.action == "comment_added":
        return f"linear comment on {ident}"
    if event.action == "state_change":
        prev = event.previous_state.name if event.previous_state else "?"
        cur = issue.state.name if issue.state else "?"
        return f"linear {ident} state {prev} → {cur}"
    return f"linear issue {event.action} {ident}"


def _episode_name(event: LinearIssueEvent) -> str:
    ident = event.issue.identifier or event.issue.id or "unknown"
    return f"linear_{ident}_{event.action}"


def _episode_body(event: LinearIssueEvent) -> str:
    issue = event.issue
    lines = [
        f"Linear issue {issue.identifier}: {issue.title}",
        f"State: {issue.state.name if issue.state else 'unknown'}",
    ]
    if issue.assignee:
        lines.append(f"Assignee: {issue.assignee.name}")
    if issue.labels:
        lines.append("Labels: " + ", ".join(lbl.name for lbl in issue.labels))
    if event.action == "state_change" and event.previous_state:
        lines.append(
            f"State change: {event.previous_state.name} -> "
            f"{issue.state.name if issue.state else 'unknown'}"
        )
    if event.action == "comment_added" and event.comment:
        author = event.comment.author.name if event.comment.author else "unknown"
        lines.append(f"Comment by {author}: {event.comment.body}")
    if issue.description:
        lines.append("")
        lines.append(issue.description)
    return "\n".join(lines)


def _status_value(issue: LinearIssue) -> str:
    if issue.state is None:
        return "unknown"
    type_ = (issue.state.type or "").lower()
    if type_ == "completed":
        return "closed"
    if type_ == "canceled":
        return "closed"
    if type_ == "started":
        return "open"
    if type_ in {"backlog", "unstarted"}:
        return "open"
    return "open"


def _comment_entity(
    comment: LinearComment,
    issue_key: str,
    source_key: str,
    observed_at: str,
) -> tuple[EntityUpsert, list[EdgeUpsert]]:
    author = comment.author.name if comment.author else "unknown"
    comment_key = f"linear:comment:{comment.id}"
    entity = EntityUpsert(
        entity_key=comment_key,
        labels=("Entity", "Comment"),
        properties={
            "name": f"Linear comment by {author}",
            "summary": comment.body[:16000],
            "body": comment.body[:16000],
            "author": author,
            "comment_id": comment.id,
            "source_ref": source_key,
            "observed_at": observed_at,
            "created_at": _iso(comment.created_at),
        },
    )
    edges = [
        EdgeUpsert(
            "HAS_COMMENT",
            issue_key,
            comment_key,
            {"source_ref": source_key, "valid_from": observed_at},
        ),
    ]
    if comment.author:
        edges.append(
            EdgeUpsert(
                "AUTHORED_BY",
                comment_key,
                _person_key(comment.author.id),
                {"source_ref": source_key, "valid_from": observed_at},
            )
        )
    return entity, edges


def _person_entity(person: Any, source_key: str, observed_at: str) -> EntityUpsert:
    key = _person_key(person.id)
    props: dict[str, Any] = {
        "name": person.name,
        "linear_user_id": person.id,
        "source_ref": source_key,
        "observed_at": observed_at,
    }
    if getattr(person, "email", None):
        props["email"] = person.email
    return EntityUpsert(entity_key=key, labels=("Entity", "Person"), properties=props)


def _person_key(user_id: str) -> str:
    return f"linear:user:{user_id}"


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return datetime.now(timezone.utc).isoformat()
    else:
        return datetime.now(timezone.utc).isoformat()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _dedupe_entities(items: list[EntityUpsert]) -> list[EntityUpsert]:
    out: dict[str, EntityUpsert] = {}
    for item in items:
        if item.entity_key in out:
            prior = out[item.entity_key]
            labels = tuple(dict.fromkeys([*prior.labels, *item.labels]))
            props = {**prior.properties, **item.properties}
            out[item.entity_key] = EntityUpsert(item.entity_key, labels, props)
        else:
            out[item.entity_key] = item
    return list(out.values())


def _dedupe_edges(items: list[EdgeUpsert]) -> list[EdgeUpsert]:
    out: dict[tuple[str, str, str], EdgeUpsert] = {}
    for item in items:
        out[(item.edge_type, item.from_entity_key, item.to_entity_key)] = item
    return list(out.values())
