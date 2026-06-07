"""Analytics computed from the canonical ``ClaimQueryPort`` alone.

``counts``/``freshness``/``quality`` are derivable from the claim rows, so any
backend with a real ``claim_query`` (in_memory, embedded, neo4j) gets a working
analytics projection without a backend-specific rebuild. ``repair`` is a no-op
here because these projections are computed on read, never materialised.

This keeps ``graph status`` / ``data_plane_status`` honest on the neo4j profile
while the richer native projections (vector semantic, cypher inspection,
portable snapshot) remain backend-specific TODOs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from domain.ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow
from domain.ports.graph.analytics import RepairReport

# Pull a generous page; analytics over a single pot's claim set. Backends that
# need true streaming aggregation can override this adapter.
_SCAN_LIMIT = 100_000


@dataclass(slots=True)
class ClaimQueryAnalytics:
    """``GraphAnalyticsPort`` over a backend's canonical claim store."""

    claim_query: ClaimQueryPort

    def _rows(self, pot_id: str) -> list[ClaimRow]:
        return list(
            self.claim_query.find_claims(
                ClaimQueryFilter(
                    pot_id=pot_id, include_invalidated=True, limit=_SCAN_LIMIT
                )
            )
        )

    def counts(self, pot_id: str) -> Mapping[str, int]:
        rows = self._rows(pot_id)
        entities = {k for r in rows for k in (r.subject_key, r.object_key)}
        predicates = {r.predicate for r in rows}
        return {
            "claims": len(rows),
            "entities": len(entities),
            "predicates": len(predicates),
            "invalidated": sum(1 for r in rows if r.invalid_at is not None),
        }

    def freshness(self, pot_id: str) -> Mapping[str, Any]:
        stamps = [r.valid_at for r in self._rows(pot_id) if r.valid_at is not None]
        return {
            "oldest": min(stamps).isoformat() if stamps else None,
            "newest": max(stamps).isoformat() if stamps else None,
            "stamped_claims": len(stamps),
        }

    def quality(self, pot_id: str) -> Mapping[str, Any]:
        rows = self._rows(pot_id)
        return {
            "status": "ok" if rows else "empty",
            "open_conflicts": 0,
            "claim_count": len(rows),
        }

    def repair(self, pot_id: str, *, targets: Sequence[str] = ()) -> RepairReport:
        return RepairReport(
            pot_id=pot_id,
            targets=tuple(targets),
            repaired={},
            detail="analytics are computed on read from claim_query; nothing to rebuild",
        )


__all__ = ["ClaimQueryAnalytics"]
