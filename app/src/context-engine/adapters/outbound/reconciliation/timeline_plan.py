"""Shared timeline-subgraph mutation builder.

Every ingestion-time plan builder calls :func:`build_timeline_mutations` to
emit the Activity + Period + edges that make up the timeline subgraph. The
output is appended to the existing ``ReconciliationPlan.entity_upserts`` and
``edge_upserts`` lists, so timeline nodes flow through the same validation,
split, and apply pipeline as every other mutation.

Schema emitted:

    (Person|Agent|Team) -[PERFORMED]-> (Activity) -[TOUCHED]-> (Entity ...)
                                           |
                                           `-[IN_PERIOD]-> (Period, daily)

See docs/context-graph/timeline.md.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Iterable

from domain.graph_mutations import EdgeUpsert, EntityUpsert


# Canonical verb vocabulary. Not enforced by the ontology — verbs are free-form
# strings so new sources can introduce new verbs without a schema change — but
# these are the values plan builders and the deep agent should prefer so
# queries like "show me everyone's recent merges" stay stable.
VERB_OPENED_PR = "opened_pr"
VERB_MERGED_PR = "merged_pr"
VERB_CLOSED_PR = "closed_pr"
VERB_REVIEWED_PR = "reviewed_pr"
VERB_AUTHORED_COMMIT = "authored_commit"
VERB_OPENED_ISSUE = "opened_issue"
VERB_STATE_CHANGED = "state_changed"
VERB_COMMENTED = "commented"
VERB_ASSIGNED = "assigned"
VERB_DEPLOYED = "deployed"
VERB_DECIDED = "decided"
VERB_DECLARED_PROGRESS = "declared_progress"
VERB_DECLARED_COMPLETED = "declared_completed"
VERB_PERFORMED = "performed"  # last-resort fallback


def build_timeline_mutations(
    *,
    pot_id: str,
    verb: str,
    occurred_at: str,
    summary: str,
    source_ref_key: str,
    actor_key: str | None,
    actor_labels: tuple[str, ...] | None = None,
    actor_properties: dict[str, object] | None = None,
    touched_entity_keys: Iterable[str] = (),
    activity_suffix: str | None = None,
    branch: str | None = None,
    environment: str | None = None,
    confidence: float | None = None,
    extra_properties: dict[str, object] | None = None,
) -> tuple[list[EntityUpsert], list[EdgeUpsert]]:
    """Build the Activity + Period + edges for one timeline happening.

    ``actor_key``/``actor_labels``/``actor_properties`` are optional: events
    without a known actor (e.g. bot-authored commits with an unknown handle)
    still produce an Activity, just without a ``PERFORMED`` edge.

    The Activity's ``entity_key`` is deterministic over
    ``(verb, source_ref_key, activity_suffix)`` so re-ingestion of the same
    event idempotently upserts the same Activity rather than creating duplicates.
    """
    entities: list[EntityUpsert] = []
    edges: list[EdgeUpsert] = []

    activity_key = _activity_key(verb, source_ref_key, activity_suffix)
    iso_when = _normalize_iso(occurred_at)
    day_bucket = iso_when[:10]
    period_key = _period_key(pot_id, day_bucket)

    activity_props: dict[str, object] = {
        "name": summary[:300] if summary else verb,
        "verb": verb,
        "occurred_at": iso_when,
        "summary": summary or verb,
        "source_ref": source_ref_key,
        "pot_id": pot_id,
        "lifecycle_state": "completed",
    }
    if branch:
        activity_props["branch"] = branch
    if environment:
        activity_props["environment"] = environment
    if actor_key:
        activity_props["actor_key"] = actor_key
    if confidence is not None:
        activity_props["confidence"] = float(confidence)
    if extra_properties:
        for k, v in extra_properties.items():
            if k not in activity_props and v is not None:
                activity_props[k] = v

    entities.append(
        EntityUpsert(
            entity_key=activity_key,
            labels=("Entity", "Activity"),
            properties=activity_props,
        )
    )

    entities.append(
        EntityUpsert(
            entity_key=period_key,
            labels=("Entity", "Period"),
            properties={
                "name": f"Timeline period {day_bucket} ({pot_id})",
                "label": day_bucket,
                "period_kind": "daily",
                "opened_at": f"{day_bucket}T00:00:00+00:00",
                "pot_id": pot_id,
                "lifecycle_state": "open",
            },
        )
    )

    edges.append(
        EdgeUpsert(
            "IN_PERIOD",
            activity_key,
            period_key,
            {"source_ref": source_ref_key, "valid_from": iso_when},
        )
    )

    if actor_key:
        if actor_labels and actor_properties is not None:
            # Caller didn't emit the actor entity itself — make sure it exists
            # (dedupe in the planner will merge with any existing upsert for
            # the same key).
            entities.append(
                EntityUpsert(
                    entity_key=actor_key,
                    labels=actor_labels,
                    properties=dict(actor_properties),
                )
            )
        edges.append(
            EdgeUpsert(
                "PERFORMED",
                actor_key,
                activity_key,
                {"source_ref": source_ref_key, "valid_from": iso_when},
            )
        )

    for touched in touched_entity_keys:
        if not touched or touched == activity_key:
            continue
        edges.append(
            EdgeUpsert(
                "TOUCHED",
                activity_key,
                touched,
                {"source_ref": source_ref_key, "valid_from": iso_when},
            )
        )

    return entities, edges


def _activity_key(verb: str, source_ref_key: str, suffix: str | None) -> str:
    raw = f"{verb}|{source_ref_key}|{suffix or ''}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"timeline:activity:{verb}:{digest}"


def _period_key(pot_id: str, day_bucket: str) -> str:
    return f"timeline:period:daily:{pot_id}:{day_bucket}"


def _normalize_iso(value: str | datetime | None) -> str:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


__all__ = [
    "VERB_OPENED_PR",
    "VERB_MERGED_PR",
    "VERB_CLOSED_PR",
    "VERB_REVIEWED_PR",
    "VERB_AUTHORED_COMMIT",
    "VERB_OPENED_ISSUE",
    "VERB_STATE_CHANGED",
    "VERB_COMMENTED",
    "VERB_ASSIGNED",
    "VERB_DEPLOYED",
    "VERB_DECIDED",
    "VERB_DECLARED_PROGRESS",
    "VERB_DECLARED_COMPLETED",
    "VERB_PERFORMED",
    "build_timeline_mutations",
]
