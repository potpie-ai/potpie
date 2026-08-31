"""Aggregate-Cypher analytics for the FalkorDB claim store.

:class:`ClaimQueryAnalytics` derives ``counts``/``freshness``/``quality`` by
pulling every claim row through the claim-query port — a full scan per call,
and ``graph read``/``graph status`` make several such calls per request. On
FalkorDB the same numbers are one aggregate query each, so this adapter keeps
those calls O(1)-ish in claim count and only falls back to the scan-based
implementation when the aggregate query fails (e.g. injected test graphs that
stub ``query``). ``repair`` always delegates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timezone
from typing import Any, Callable, Mapping, Sequence

from potpie_context_engine.adapters.outbound.graph.backends.claim_query_analytics import (
    ClaimQueryAnalytics,
)
from potpie_context_engine.adapters.outbound.graph.canonical_claim_query import parse_dt
from potpie_context_engine.core.ports.graph.analytics import RepairReport

logger = logging.getLogger(__name__)

_COUNTS_CYPHER = """
MATCH (a:Entity {group_id: $gid})-[r:RELATES_TO {group_id: $gid}]->(b:Entity {group_id: $gid})
RETURN count(r) AS claims,
       count(DISTINCT r.name) AS predicates,
       count(CASE WHEN r.invalid_at IS NOT NULL THEN 1 END) AS invalidated
"""

_ENTITY_COUNT_CYPHER = """
MATCH (a:Entity {group_id: $gid})-[r:RELATES_TO {group_id: $gid}]->(b:Entity {group_id: $gid})
WITH collect(DISTINCT r.subject_key) AS subjects, collect(DISTINCT r.object_key) AS objects
RETURN size(subjects) + size([key IN objects WHERE NOT key IN subjects]) AS entities
"""

# ``min``/``max`` compare the stored ISO-8601 strings lexicographically, which
# matches chronological order because the writers stamp zero-padded UTC
# timestamps.
_FRESHNESS_CYPHER = """
MATCH (a:Entity {group_id: $gid})-[r:RELATES_TO {group_id: $gid}]->(b:Entity {group_id: $gid})
WHERE r.valid_at IS NOT NULL
RETURN min(r.valid_at) AS oldest, max(r.valid_at) AS newest, count(r) AS stamped
"""


def _iso_utc(value: Any) -> str | None:
    dt = parse_dt(value)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


@dataclass(slots=True)
class FalkorDBAnalytics:
    """``GraphAnalyticsPort`` computed with aggregate Cypher instead of row scans."""

    graph_provider: Callable[[], Any]
    fallback: ClaimQueryAnalytics

    def _rows(self, cypher: str, pot_id: str) -> list[Any]:
        result = self.graph_provider().query(cypher, params={"gid": pot_id})
        return list(getattr(result, "result_set", []) or [])

    def counts(self, pot_id: str) -> Mapping[str, int]:
        try:
            claims, predicates, invalidated = self._rows(_COUNTS_CYPHER, pot_id)[0]
            (entities,) = self._rows(_ENTITY_COUNT_CYPHER, pot_id)[0]
        except Exception:  # noqa: BLE001
            logger.debug("aggregate counts failed; falling back to scan", exc_info=True)
            return self.fallback.counts(pot_id)
        return {
            "claims": int(claims or 0),
            "entities": int(entities or 0),
            "predicates": int(predicates or 0),
            "invalidated": int(invalidated or 0),
        }

    def freshness(self, pot_id: str) -> Mapping[str, Any]:
        try:
            oldest, newest, stamped = self._rows(_FRESHNESS_CYPHER, pot_id)[0]
        except Exception:  # noqa: BLE001
            logger.debug(
                "aggregate freshness failed; falling back to scan", exc_info=True
            )
            return self.fallback.freshness(pot_id)
        return {
            "oldest": _iso_utc(oldest),
            "newest": _iso_utc(newest),
            "stamped_claims": int(stamped or 0),
        }

    def quality(self, pot_id: str) -> Mapping[str, Any]:
        claim_count = int(self.counts(pot_id).get("claims", 0))
        return {
            "status": "ok" if claim_count else "empty",
            "open_conflicts": 0,
            "claim_count": claim_count,
        }

    def repair(self, pot_id: str, *, targets: Sequence[str] = ()) -> RepairReport:
        return self.fallback.repair(pot_id, targets=targets)


__all__ = ["FalkorDBAnalytics"]
