"""InfraTopologyReader (UC2 / P9).

Inputs: scope (service / env / file). Logic: bounded neighbourhood
traversal over claim edges with infra predicates (``DEPLOYED_TO``,
``OF_SERVICE``, ``DEPENDS_ON``, ``STORED_IN``, ``USES``, ``EXPOSES``,
``CONFIGURED_BY``), environment-filtered via the ``environment`` edge
property. Supports blast-radius (incoming ``DEPENDS_ON`` traversal
with depth). Supports ``as_of`` via the bitemporal predicate.

This reader is the F1 reader: with the Kubernetes scanner emitting
``Deployment OF_SERVICE Service`` and ``Deployment DEPLOYED_TO Env``,
the question "what env runs auth-svc?" returns a real edge instead of
0% coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from application.readers._common import (
    ReadRequest,
    ReadResponse,
    coverage_status_from_count,
    rank_candidates,
)
from domain.ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow
from domain.ranking import Candidate, RankingService


_INFRA_PREDICATES: tuple[str, ...] = (
    "DEPLOYED_TO",
    "OF_SERVICE",
    "DEPENDS_ON",
    "STORED_IN",
    "USES",
    "EXPOSES",
    "CONFIGURED_BY",
    "OWNED_BY",
    "PRIMARY_STORE",
    "CURRENT_VERSION",
)


@dataclass(slots=True)
class InfraTopologyReader:
    claim_query: ClaimQueryPort
    ranker: RankingService
    family: str = "infra_topology"
    max_blast_radius_depth: int = 2

    def read(self, req: ReadRequest) -> ReadResponse:
        anchor_keys = _anchor_entity_keys(req.scope)
        rows = self._traverse(req, anchor_keys=anchor_keys)
        environment_filter = _string_or_none(req.scope.get("environment"))

        candidates: list[Candidate] = []
        for row in rows:
            row_env = row.properties.get("environment")
            if environment_filter and isinstance(row_env, str) and row_env != environment_filter:
                continue

            overlap = _scope_overlap(row, anchor_keys=anchor_keys, environment=environment_filter)
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

        ranked = rank_candidates(service=self.ranker, candidates=candidates, req=req)
        return ReadResponse(
            family=self.family,
            items=tuple(ranked),
            coverage_status=coverage_status_from_count(
                found=len(ranked), requested=req.max_items
            ),
            meta={
                "anchor_keys": list(anchor_keys),
                "environment": environment_filter,
                "candidate_pool": len(rows),
            },
        )

    def _traverse(
        self, req: ReadRequest, *, anchor_keys: Iterable[str]
    ) -> list[ClaimRow]:
        """Bounded neighbourhood traversal (depth-limited BFS).

        Each hop fans out from the current anchor frontier in both
        directions (outgoing + incoming). The depth bound is set on
        the reader; depth=2 lets queries like "what env runs service X"
        traverse Service ← OF_SERVICE — Deployment → DEPLOYED_TO → Env
        without paging through irrelevant subgraphs.
        """
        if not anchor_keys:
            return self.claim_query.find_claims(
                ClaimQueryFilter(
                    pot_id=req.pot_id,
                    predicate_in=_INFRA_PREDICATES,
                    include_invalidated=req.include_invalidated,
                    as_of=req.as_of,
                    limit=max(req.max_items * 4, 16),
                )
            )

        seen_rows: dict[str, ClaimRow] = {}
        frontier: set[str] = set(anchor_keys)
        visited_anchors: set[str] = set()
        limit_per_hop = max(req.max_items * 4, 16)

        for _ in range(self.max_blast_radius_depth):
            if not frontier:
                break
            current = tuple(sorted(frontier - visited_anchors))
            if not current:
                break
            visited_anchors.update(current)

            outgoing = self.claim_query.find_claims(
                ClaimQueryFilter(
                    pot_id=req.pot_id,
                    predicate_in=_INFRA_PREDICATES,
                    subject_key_in=current,
                    include_invalidated=req.include_invalidated,
                    as_of=req.as_of,
                    limit=limit_per_hop,
                )
            )
            incoming = self.claim_query.find_claims(
                ClaimQueryFilter(
                    pot_id=req.pot_id,
                    predicate_in=_INFRA_PREDICATES,
                    object_key_in=current,
                    include_invalidated=req.include_invalidated,
                    as_of=req.as_of,
                    limit=limit_per_hop,
                )
            )

            next_frontier: set[str] = set()
            for row in (*outgoing, *incoming):
                key = _make_candidate_key(row)
                if key in seen_rows:
                    continue
                seen_rows[key] = row
                # Expand frontier to opposite endpoint so the next hop
                # can pick up downstream edges (e.g. Deployment →
                # DEPLOYED_TO after Service ← OF_SERVICE → Deployment).
                if row.subject_key in current:
                    next_frontier.add(row.object_key)
                if row.object_key in current:
                    next_frontier.add(row.subject_key)
            frontier = next_frontier - visited_anchors

        return list(seen_rows.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _anchor_entity_keys(scope: Mapping[str, Any]) -> list[str]:
    """Turn scope into the entity_keys we anchor the traversal on."""
    keys: list[str] = []
    services = scope.get("services") or scope.get("service")
    if isinstance(services, str):
        services = [services]
    if isinstance(services, list):
        for s in services:
            if isinstance(s, str) and s.strip():
                keys.append(f"service:{s.strip().lower()}")
    if not keys:
        # Allow caller to pass a raw entity_key directly
        anchor = scope.get("anchor_entity_key")
        if isinstance(anchor, str) and anchor.strip():
            keys.append(anchor.strip())
    return keys


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _scope_overlap(
    row: ClaimRow,
    *,
    anchor_keys: Iterable[str],
    environment: str | None,
) -> float:
    score = 0.0
    bumps = 0
    if any(k == row.subject_key or k == row.object_key for k in anchor_keys):
        score += 1.0
        bumps += 1
    row_env = row.properties.get("environment")
    if environment and isinstance(row_env, str) and row_env == environment:
        score += 1.0
        bumps += 1
    if bumps == 0:
        return 0.5  # neutral: unscoped result
    return min(1.0, score / max(bumps, 1))


def _corroboration(row: ClaimRow) -> int:
    count = row.properties.get("corroboration_count")
    if isinstance(count, int) and count > 0:
        return count
    return 1


def _make_candidate_key(row: ClaimRow) -> str:
    return f"{row.predicate}:{row.subject_key}:{row.object_key}:{row.source_ref or '-'}"


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    return {
        "predicate": row.predicate,
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        "fact": row.fact,
        "environment": row.properties.get("environment"),
        "source_ref": row.source_ref,
        "source_system": row.source_system,
        "valid_at": row.valid_at.isoformat() if row.valid_at else None,
        "evidence_strength": row.evidence_strength,
    }


__all__ = ["InfraTopologyReader"]
