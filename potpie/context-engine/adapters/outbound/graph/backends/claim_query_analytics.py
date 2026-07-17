"""Analytics computed from the canonical ``ClaimQueryPort`` alone.

``counts``/``freshness``/``quality`` are derivable from the claim rows, so any
backend with a real ``claim_query`` (in_memory, embedded, neo4j) gets a working
analytics projection without a backend-specific rebuild. ``repair`` is usually a
no-op because these projections are computed on read, but writable backends can
inject a narrow entity-summary repair callback.

This keeps ``graph status`` / ``data_plane_status`` honest on the neo4j profile
while the richer native projections (vector semantic, cypher inspection,
portable snapshot) remain backend-specific TODOs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from adapters.outbound.graph.entity_summary_repair import (
    ENTITY_SUMMARY_TARGET,
    wants_entity_summary_repair,
)
from adapters.outbound.graph.semantic_index_repair import (
    SEMANTIC_INDEX_TARGET,
    wants_semantic_index_repair,
)
from domain.ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow
from domain.ports.graph.analytics import RepairReport

# Pull a generous page; analytics over a single pot's claim set. Backends that
# need true streaming aggregation can override this adapter.
_SCAN_LIMIT = 100_000


@dataclass(slots=True)
class ClaimQueryAnalytics:
    """``GraphAnalyticsPort`` over a backend's canonical claim store."""

    claim_query: ClaimQueryPort
    entity_summary_repair: Callable[[str], int] | None = None
    semantic_index_repair: Callable[[str], Mapping[str, Any]] | None = None

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
        repaired: dict[str, int] = {}
        details: list[str] = []
        if self.semantic_index_repair is not None and wants_semantic_index_repair(
            targets
        ):
            report = self.semantic_index_repair(pot_id)
            repaired[SEMANTIC_INDEX_TARGET] = int(report.get("repaired") or 0)
            failed = int(report.get("failed") or 0)
            if failed:
                repaired[f"{SEMANTIC_INDEX_TARGET}_failed"] = failed
            detail = report.get("detail")
            if isinstance(detail, str) and detail:
                details.append(detail)
        if self.entity_summary_repair is not None and wants_entity_summary_repair(
            targets
        ):
            count = self.entity_summary_repair(pot_id)
            repaired[ENTITY_SUMMARY_TARGET] = count
            details.append(f"repaired {count} entity summaries")
        if repaired or details:
            return RepairReport(
                pot_id=pot_id,
                targets=tuple(targets),
                repaired=repaired,
                detail="; ".join(details) or None,
            )
        return RepairReport(
            pot_id=pot_id,
            targets=tuple(targets),
            repaired={},
            detail="analytics are computed on read from claim_query; nothing to rebuild",
        )


__all__ = ["ClaimQueryAnalytics"]
