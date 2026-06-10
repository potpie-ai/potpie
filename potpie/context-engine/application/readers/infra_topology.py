"""InfraTopologyReader (topology core).

Inputs: scope (service / env / file). Logic: bounded neighbourhood
traversal over claim edges with topology predicates (``DEFINED_IN``,
``DEPLOYED_TO``, ``DEPENDS_ON``, ``USES``, ``HOSTED_ON``, ``OWNED_BY``,
``PROVIDES``, ``IMPLEMENTED_IN``),
environment-filtered via the ``environment`` edge property. Supports
blast-radius (incoming ``DEPENDS_ON`` traversal with depth). Supports
``as_of`` via the bitemporal predicate.

When a harness records ``Service DEPLOYED_TO Environment`` with the
environment stamped on the edge, the question "what env runs auth-svc?"
returns a real edge instead of 0% coverage.
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
    "DEFINED_IN",
    "DEPLOYED_TO",
    "DEPENDS_ON",
    "USES",
    "HOSTED_ON",
    "OWNED_BY",
    "PROVIDES",
    "IMPLEMENTED_IN",
)


_MAX_TRAVERSAL_DEPTH = 4


@dataclass(slots=True)
class InfraTopologyReader:
    claim_query: ClaimQueryPort
    ranker: RankingService
    family: str = "infra_topology"
    max_blast_radius_depth: int = 2

    def read(self, req: ReadRequest) -> ReadResponse:
        anchor_keys = _anchor_entity_keys(req.scope)
        environment_filter = _normalise_environment(req.scope.get("environment"))
        include_unqualified_environment = _scope_bool(
            req.scope, "include_unqualified_environment"
        )
        rows = self._traverse(
            req,
            anchor_keys=anchor_keys,
            environment_filter=environment_filter,
            include_unqualified_environment=include_unqualified_environment,
        )

        candidates: list[Candidate] = []
        for row in rows:
            if not _matches_environment(
                row,
                environment_filter=environment_filter,
                include_unqualified_environment=include_unqualified_environment,
            ):
                continue

            overlap = _scope_overlap(
                row, anchor_keys=anchor_keys, environment=environment_filter
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
                "include_unqualified_environment": include_unqualified_environment,
                "candidate_pool": len(rows),
            },
        )

    def _traverse(
        self,
        req: ReadRequest,
        *,
        anchor_keys: Iterable[str],
        environment_filter: str | None,
        include_unqualified_environment: bool,
    ) -> list[ClaimRow]:
        """Bounded neighbourhood traversal (depth-limited BFS).

        Each hop fans out from the current anchor frontier in both
        directions (outgoing + incoming). The depth bound is set on
        the reader; depth=2 lets queries like "what env runs service X"
        (Service → DEPLOYED_TO → Env, one hop) plus a second hop for
        blast-radius (Service ← DEPENDS_ON — caller, or Service → USES →
        DataStore) without paging through irrelevant subgraphs.
        """
        if not anchor_keys:
            rows = self.claim_query.find_claims(
                ClaimQueryFilter(
                    pot_id=req.pot_id,
                    predicate_in=_INFRA_PREDICATES,
                    include_invalidated=req.include_invalidated,
                    as_of=req.as_of,
                    limit=max(req.max_items * 4, 16),
                )
            )
            return _filter_environment(
                rows,
                environment_filter=environment_filter,
                include_unqualified_environment=include_unqualified_environment,
            )

        # Traverse-axis controls: bounded depth and direction-aware walk.
        depth = self.max_blast_radius_depth
        if isinstance(req.depth, int) and req.depth > 0:
            depth = min(req.depth, _MAX_TRAVERSAL_DEPTH)
        direction = (req.direction or "both").lower()
        walk_out = direction in ("out", "both")
        walk_in = direction in ("in", "both")

        seen_rows: dict[str, ClaimRow] = {}
        frontier: set[str] = set(anchor_keys)
        visited_anchors: set[str] = set()
        limit_per_hop = max(req.max_items * 4, 16)

        for _ in range(depth):
            if not frontier:
                break
            current = tuple(sorted(frontier - visited_anchors))
            if not current:
                break
            visited_anchors.update(current)

            hop_rows: list[ClaimRow] = []
            if walk_out:
                hop_rows.extend(
                    self.claim_query.find_claims(
                        ClaimQueryFilter(
                            pot_id=req.pot_id,
                            predicate_in=_INFRA_PREDICATES,
                            subject_key_in=current,
                            include_invalidated=req.include_invalidated,
                            as_of=req.as_of,
                            limit=limit_per_hop,
                        )
                    )
                )
            if walk_in:
                hop_rows.extend(
                    self.claim_query.find_claims(
                        ClaimQueryFilter(
                            pot_id=req.pot_id,
                            predicate_in=_INFRA_PREDICATES,
                            object_key_in=current,
                            include_invalidated=req.include_invalidated,
                            as_of=req.as_of,
                            limit=limit_per_hop,
                        )
                    )
                )

            hop_rows = _filter_environment(
                hop_rows,
                environment_filter=environment_filter,
                include_unqualified_environment=include_unqualified_environment,
            )
            next_frontier: set[str] = set()
            for row in hop_rows:
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


def _normalise_environment(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _scope_bool(scope: Mapping[str, Any], key: str) -> bool:
    value = scope.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _filter_environment(
    rows: Iterable[ClaimRow],
    *,
    environment_filter: str | None,
    include_unqualified_environment: bool,
) -> list[ClaimRow]:
    return [
        row
        for row in rows
        if _matches_environment(
            row,
            environment_filter=environment_filter,
            include_unqualified_environment=include_unqualified_environment,
        )
    ]


def _matches_environment(
    row: ClaimRow,
    *,
    environment_filter: str | None,
    include_unqualified_environment: bool,
) -> bool:
    if environment_filter is None:
        return True
    row_env = _row_environment(row)
    if row_env == environment_filter:
        return True
    return include_unqualified_environment and row_env is None


def _row_environment(row: ClaimRow) -> str | None:
    env = row.environment
    if isinstance(env, str) and env.strip():
        return env.strip().lower()
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
    if environment and _row_environment(row) == environment:
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
    return row.claim_key or f"{row.predicate}:{row.subject_key}:{row.object_key}"


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    return {
        "predicate": row.predicate,
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        "claim_key": row.claim_key,
        "subgraph": row.subgraph,
        "truth": row.truth,
        "fact": row.fact,
        "environment": _row_environment(row),
        "source_refs": list(row.source_refs),
        "source_system": row.source_system,
        "valid_at": row.valid_at.isoformat() if row.valid_at else None,
        "valid_until": row.valid_until.isoformat() if row.valid_until else None,
        "observed_at": row.observed_at.isoformat() if row.observed_at else None,
        "evidence_strength": row.evidence_strength,
    }


__all__ = ["InfraTopologyReader"]
