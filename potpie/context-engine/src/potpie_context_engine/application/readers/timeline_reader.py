"""TimelineReader (UC3 / P9).

Inputs: scope + window. Logic: surface :Activity entities that touched
the scope inside the window, ranked by recency + scope-overlap. Uses
``MENTIONS`` provenance (F4) so PR / ticket / deploy / alert activities
all converge: any activity whose body mentioned the scope service has
an outgoing MENTIONS edge linking it back.

This reader is the F4 reader: prior to MENTIONS, an Activity bound to
``person:alice`` (via ``PERFORMED``) was invisible to a "what
activities touched service:auth-svc this week?" query. With MENTIONS,
the activity → service link is queryable.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from potpie_context_engine.application.readers._common import (
    ReadRequest,
    ReadResponse,
    claim_corroboration,
    claim_payload,
    coverage_status_from_count,
    dedupe_claim_rows,
    rank_candidates,
    row_in_anchor_set,
    service_anchor_keys,
)
from potpie_context_engine.domain.ports.claim_query import (
    ClaimQueryFilter,
    ClaimQueryPort,
    ClaimRow,
)
from potpie_context_engine.domain.ranking import Candidate, RankingService


_TIMELINE_PREDICATES: tuple[str, ...] = (
    "MENTIONS",
    "TOUCHED",
    "PERFORMED",
    "AUTHORED",
)


@dataclass(slots=True)
class TimelineReader:
    claim_query: ClaimQueryPort
    ranker: RankingService
    family: str = "timeline"

    def read(self, req: ReadRequest) -> ReadResponse:
        anchor_keys = service_anchor_keys(req.scope)
        window_after, window_before = _resolve_window(req)

        rows = self._activities_for_scope(
            req=req,
            anchor_keys=anchor_keys,
            window_after=window_after,
            window_before=window_before,
        )

        grouped_rows = _dedupe_activity_rows(rows, anchor_keys=anchor_keys)

        candidates: list[Candidate] = []
        for row in grouped_rows:
            event_time = _event_datetime(row)
            overlap = (
                1.0 if (not anchor_keys) or row_in_anchor_set(row, anchor_keys) else 0.4
            )
            sim = row.properties.get("semantic_similarity")
            candidates.append(
                Candidate(
                    candidate_key=_activity_key(row),
                    payload=_payload_from_row(row),
                    strength=row.evidence_strength,
                    valid_at=event_time,
                    scope_overlap=overlap,
                    corroboration_count=claim_corroboration(row),
                    semantic_similarity=float(sim)
                    if isinstance(sim, (int, float))
                    else None,
                )
            )

        # Time-skewed: timeline reads prefer 'fresh' by default if no
        # override is set, so the ranker's recency factor dominates.
        if req.freshness_preference == "balanced" and not req.query:
            req = _with_freshness(req, "fresh")

        ranked = rank_candidates(service=self.ranker, candidates=candidates, req=req)
        return ReadResponse(
            family=self.family,
            items=tuple(ranked),
            coverage_status=coverage_status_from_count(
                found=len(ranked), requested=req.max_items
            ),
            meta={
                "anchor_keys": list(anchor_keys),
                "window_after": window_after.isoformat() if window_after else None,
                "window_before": window_before.isoformat() if window_before else None,
                "candidate_pool": len(rows),
                "activity_pool": len(grouped_rows),
            },
        )

    def _activities_for_scope(
        self,
        *,
        req: ReadRequest,
        anchor_keys: Iterable[str],
        window_after: datetime | None,
        window_before: datetime | None,
    ) -> list[ClaimRow]:
        anchors = tuple(anchor_keys)
        base = ClaimQueryFilter(
            pot_id=req.pot_id,
            predicate_in=_TIMELINE_PREDICATES,
            include_invalidated=req.include_invalidated,
            as_of=req.as_of,
            source_ref_in=req.source_refs,
            # Timeline windows are source-event windows. Older rows may have
            # occurred_at only in properties with valid_at set to ingestion time,
            # so filter after hydration with _event_datetime().
            limit=max(req.max_items * 20, 200),
            fact_query=req.query,
        )
        if not anchors:
            return _filter_window(
                dedupe_claim_rows(self.claim_query.find_claims(base)),
                window_after=window_after,
                window_before=window_before,
            )

        # Activity → anchor (MENTIONS) plus anchor → activity (PERFORMED etc).
        mentioning = self.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=base.pot_id,
                predicate_in=("MENTIONS", "TOUCHED"),
                object_key_in=anchors,
                include_invalidated=base.include_invalidated,
                as_of=base.as_of,
                source_ref_in=base.source_ref_in,
                limit=base.limit,
                fact_query=req.query,
            )
        )
        authored = self.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=base.pot_id,
                predicate_in=("PERFORMED", "AUTHORED"),
                subject_key_in=anchors,
                include_invalidated=base.include_invalidated,
                as_of=base.as_of,
                source_ref_in=base.source_ref_in,
                limit=base.limit,
                fact_query=req.query,
            )
        )
        out = dedupe_claim_rows((*mentioning, *authored))
        return _filter_window(
            out, window_after=window_after, window_before=window_before
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_window(req: ReadRequest) -> tuple[datetime | None, datetime | None]:
    until = req.until
    since = req.since
    if since and until:
        return since, until
    if since:
        return since, until
    if until and not since:
        # Caller asked only for "before this"; pass through.
        return None, until
    return None, None


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    extras = dict(row.properties or {})
    occurred_at = _event_time_iso(row)
    return claim_payload(
        row,
        extra={
            # For every timeline predicate (TOUCHED / PERFORMED / MENTIONS) the
            # Activity is an endpoint. The event kind rides in the claim's extras
            # as ``verb_class``.
            "activity_key": _activity_key(row),
            "occurred_at": occurred_at,
            "properties": extras,
            "verb_class": extras.get("verb_class"),
        },
    )


def _with_freshness(req: ReadRequest, preference: str) -> ReadRequest:
    return ReadRequest(
        pot_id=req.pot_id,
        scope=req.scope,
        query=req.query,
        intent=req.intent,
        as_of=req.as_of,
        since=req.since,
        until=req.until,
        max_items=req.max_items,
        freshness_preference=preference,
        include_invalidated=req.include_invalidated,
        source_refs=req.source_refs,
    )


def _filter_window(
    rows: Iterable[ClaimRow],
    *,
    window_after: datetime | None,
    window_before: datetime | None,
) -> list[ClaimRow]:
    out: list[ClaimRow] = []
    for row in rows:
        event_time = _event_datetime(row)
        if window_after is not None and (
            event_time is None or event_time < window_after
        ):
            continue
        if (
            window_before is not None
            and event_time is not None
            and event_time > window_before
        ):
            continue
        if event_time is not None and event_time != row.valid_at:
            row = dataclasses.replace(row, valid_at=event_time)
        out.append(row)
    return out


def _dedupe_activity_rows(
    rows: Iterable[ClaimRow], *, anchor_keys: Iterable[str]
) -> list[ClaimRow]:
    """Collapse multiple timeline edges for the same Activity to one event row."""
    anchors = set(anchor_keys)
    grouped: dict[str, list[ClaimRow]] = {}
    order: list[str] = []
    for row in rows:
        activity_key = _activity_key(row)
        if activity_key not in grouped:
            order.append(activity_key)
        grouped.setdefault(activity_key, []).append(row)

    return [
        _representative_activity_row(grouped[key], anchors=anchors) for key in order
    ]


def _representative_activity_row(
    rows: list[ClaimRow], *, anchors: set[str]
) -> ClaimRow:
    def sort_key(row: ClaimRow) -> tuple[int, int, float, float]:
        predicate = row.predicate.upper()
        direct_anchor = 1 if (not anchors or row_in_anchor_set(row, anchors)) else 0
        predicate_priority = {
            "TOUCHED": 3,
            "MENTIONS": 2,
            "PERFORMED": 1,
            "AUTHORED": 1,
        }.get(predicate, 0)
        sim = row.properties.get("semantic_similarity")
        similarity = float(sim) if isinstance(sim, (int, float)) else 0.0
        event_time = _event_datetime(row)
        event_ts = event_time.timestamp() if event_time else 0.0
        return (direct_anchor, predicate_priority, similarity, event_ts)

    representative = max(rows, key=sort_key)
    if len(rows) <= 1:
        return representative

    related_edges: list[dict[str, Any]] = []
    source_refs: list[str] = []
    seen_source_refs: set[str] = set()
    for row in rows:
        for ref in row.source_refs:
            if ref not in seen_source_refs:
                seen_source_refs.add(ref)
                source_refs.append(ref)
        related_edges.append(
            {
                "predicate": row.predicate,
                "subject_key": row.subject_key,
                "object_key": row.object_key,
                "claim_key": row.claim_key,
            }
        )

    props = dict(representative.properties or {})
    props["activity_edge_count"] = len(rows)
    props["related_event_edges"] = related_edges
    if source_refs:
        return dataclasses.replace(
            representative, properties=props, source_refs=tuple(source_refs)
        )
    return dataclasses.replace(representative, properties=props)


def _event_datetime(row: ClaimRow) -> datetime | None:
    raw = row.properties.get("occurred_at")
    if isinstance(raw, str) and raw.strip():
        try:
            return datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
        except ValueError:
            pass
    return row.valid_at


def _event_time_iso(row: ClaimRow) -> str | None:
    event_time = _event_datetime(row)
    return event_time.isoformat() if event_time else None


def _activity_key(row: ClaimRow) -> str:
    predicate = row.predicate.upper()
    if predicate in {"PERFORMED", "AUTHORED"}:
        return row.object_key
    return row.subject_key


__all__ = ["TimelineReader"]
