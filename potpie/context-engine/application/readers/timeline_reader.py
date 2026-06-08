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

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping

from application.readers._common import (
    ReadRequest,
    ReadResponse,
    coverage_status_from_count,
    rank_candidates,
)
from domain.ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow
from domain.ranking import Candidate, RankingService


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
        anchor_keys = _scope_to_anchor_keys(req.scope)
        window_after, window_before = _resolve_window(req)

        rows = self._activities_for_scope(
            req=req,
            anchor_keys=anchor_keys,
            window_after=window_after,
            window_before=window_before,
        )

        candidates: list[Candidate] = []
        for row in rows:
            overlap = (
                1.0
                if (not anchor_keys) or _row_in_anchor_set(row, anchor_keys)
                else 0.4
            )
            candidates.append(
                Candidate(
                    candidate_key=_make_candidate_key(row),
                    payload=_payload_from_row(row),
                    strength=row.evidence_strength,
                    valid_at=row.valid_at,
                    scope_overlap=overlap,
                    corroboration_count=_corroboration(row),
                )
            )

        # Time-skewed: timeline reads prefer 'fresh' by default if no
        # override is set, so the ranker's recency factor dominates.
        if req.freshness_preference == "balanced":
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
            valid_at_after=window_after,
            valid_at_before=window_before,
            limit=max(req.max_items * 4, 32),
        )
        if not anchors:
            return self.claim_query.find_claims(base)

        # Activity → anchor (MENTIONS) plus anchor → activity (PERFORMED etc).
        mentioning = self.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=base.pot_id,
                predicate_in=("MENTIONS", "TOUCHED"),
                object_key_in=anchors,
                include_invalidated=base.include_invalidated,
                as_of=base.as_of,
                valid_at_after=window_after,
                valid_at_before=window_before,
                limit=base.limit,
            )
        )
        authored = self.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=base.pot_id,
                predicate_in=("PERFORMED", "AUTHORED"),
                subject_key_in=anchors,
                include_invalidated=base.include_invalidated,
                as_of=base.as_of,
                valid_at_after=window_after,
                valid_at_before=window_before,
                limit=base.limit,
            )
        )
        seen: set[str] = set()
        out: list[ClaimRow] = []
        for row in (*mentioning, *authored):
            key = _make_candidate_key(row)
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scope_to_anchor_keys(scope: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    services = scope.get("services") or scope.get("service")
    if isinstance(services, str):
        services = [services]
    if isinstance(services, list):
        for s in services:
            if isinstance(s, str) and s.strip():
                keys.append(f"service:{s.strip().lower()}")
    return keys


def _row_in_anchor_set(row: ClaimRow, anchor_keys: Iterable[str]) -> bool:
    anchors = set(anchor_keys)
    return row.subject_key in anchors or row.object_key in anchors


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


def _make_candidate_key(row: ClaimRow) -> str:
    return f"{row.predicate}:{row.subject_key}:{row.object_key}:{row.source_ref or '-'}"


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    extras = dict(row.properties or {})
    return {
        "predicate": row.predicate,
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        # For every timeline predicate (TOUCHED / PERFORMED / MENTIONS) the
        # Activity is the subject, so subject_key is the event identity. The
        # event kind rides in the claim's extras as ``verb_class``.
        "activity_key": row.subject_key,
        "verb_class": extras.get("verb_class"),
        "fact": row.fact,
        "source_ref": row.source_ref,
        "source_system": row.source_system,
        "valid_at": row.valid_at.isoformat() if row.valid_at else None,
        "evidence_strength": row.evidence_strength,
        "properties": extras,
    }


def _corroboration(row: ClaimRow) -> int:
    count = row.properties.get("corroboration_count")
    if isinstance(count, int) and count > 0:
        return count
    return 1


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
    )


__all__ = ["TimelineReader"]
