"""Aggregate analytics for the Ladybug Claim store."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timezone
from typing import Any, Callable, Mapping

from potpie_context_engine.adapters.outbound.graph.backends.claim_query_analytics import (
    ClaimQueryAnalytics,
)
from potpie_context_engine.adapters.outbound.graph.canonical_claim_query import parse_dt
from potpie_context_engine.adapters.outbound.graph.ladybug_writer import (
    _records_from_result,
)
from potpie_context_engine.core.ports.graph.analytics import RepairReport

logger = logging.getLogger(__name__)


def _iso_utc(value: Any) -> str | None:
    dt = parse_dt(value)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


@dataclass(slots=True)
class LadybugAnalytics:
    conn_provider: Callable[[], Any]
    fallback: ClaimQueryAnalytics

    def _row(self, cypher: str, pot_id: str) -> dict[str, Any] | None:
        rows = _records_from_result(
            self.conn_provider().execute(cypher, {"gid": pot_id})
        )
        return rows[0] if rows else None

    def counts(self, pot_id: str) -> Mapping[str, int]:
        try:
            crow = (
                self._row(
                    """
                MATCH (c:Claim {group_id: $gid})
                RETURN count(c) AS claims,
                       count(DISTINCT c.name) AS predicates,
                       count(CASE WHEN c.invalid_at IS NOT NULL THEN 1 END) AS invalidated
                """,
                    pot_id,
                )
                or {}
            )
            erow = (
                self._row(
                    """
                MATCH (e:Entity {group_id: $gid})
                RETURN count(e) AS entities
                """,
                    pot_id,
                )
                or {}
            )
            return {
                "claims": int(crow.get("claims") or 0),
                "entities": int(erow.get("entities") or 0),
                "predicates": int(crow.get("predicates") or 0),
                "invalidated": int(crow.get("invalidated") or 0),
            }
        except Exception:  # noqa: BLE001
            logger.debug("ladybug aggregate counts failed", exc_info=True)
            return self.fallback.counts(pot_id)

    def freshness(self, pot_id: str) -> Mapping[str, Any]:
        try:
            row = (
                self._row(
                    """
                MATCH (c:Claim {group_id: $gid})
                WHERE c.valid_at IS NOT NULL
                RETURN min(c.valid_at) AS oldest, max(c.valid_at) AS newest,
                       count(c) AS stamped
                """,
                    pot_id,
                )
                or {}
            )
            return {
                "oldest": _iso_utc(row.get("oldest")),
                "newest": _iso_utc(row.get("newest")),
                "stamped_claims": int(row.get("stamped") or 0),
            }
        except Exception:  # noqa: BLE001
            logger.debug("ladybug aggregate freshness failed", exc_info=True)
            return self.fallback.freshness(pot_id)

    def quality(self, pot_id: str) -> Mapping[str, Any]:
        claim_count = int(self.counts(pot_id).get("claims", 0))
        return {
            "claim_count": claim_count,
            "ok": claim_count >= 0,
        }

    def repair(self, pot_id: str) -> RepairReport:
        return self.fallback.repair(pot_id)


__all__ = ["LadybugAnalytics"]
