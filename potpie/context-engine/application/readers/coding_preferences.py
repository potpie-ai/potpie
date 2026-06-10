"""CodingPreferencesReader (UC1 / P9).

Inputs: task scope (language, framework, repo, service, file_path).
Logic: query claim edges with ``predicate='POLICY_APPLIES_TO'`` whose
``code_scope`` overlaps the task scope. Optional semantic similarity
hop via the claim-query port's ``fact_query`` for fuzzy task phrasing.
Output: ranked preferences via :class:`RankingService` with the
strength tier surfaced in each item's payload.

The reader is intentionally thin — scope intersection lives here
(use-case specific), but ranking is delegated to P7. Per the plan:
"each reader contains no logic specific to other readers."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from application.readers._common import (
    ReadRequest,
    ReadResponse,
    coverage_status_from_count,
    rank_candidates,
)
from domain.ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow
from domain.ranking import Candidate, RankingService
from domain.scope_match import hierarchical_scope_overlap


@dataclass(slots=True)
class CodingPreferencesReader:
    """Reader for the coding-preferences use case."""

    claim_query: ClaimQueryPort
    ranker: RankingService
    family: str = "coding_preferences"
    predicate: str = "POLICY_APPLIES_TO"

    def read(self, req: ReadRequest) -> ReadResponse:
        rows = self.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=req.pot_id,
                predicate_in=(self.predicate,),
                include_invalidated=req.include_invalidated,
                as_of=req.as_of,
                fact_query=req.query,
                limit=max(req.max_items * 4, 16),
            )
        )

        scope_keys = _normalise_scope_for_overlap(req.scope)
        candidates: list[Candidate] = []
        for row in rows:
            overlap = _scope_overlap(row, scope_keys)
            if overlap == 0.0 and scope_keys:
                # Hard zero on overlap: skip — readers should not surface
                # rules that demonstrably don't apply.
                continue
            sim = row.properties.get("semantic_similarity")
            candidates.append(
                Candidate(
                    candidate_key=_make_candidate_key(row),
                    payload=_payload_from_row(row),
                    strength=row.evidence_strength,
                    valid_at=row.valid_at,
                    corroboration_count=_corroboration(row),
                    scope_overlap=overlap if scope_keys else None,
                    semantic_similarity=float(sim)
                    if isinstance(sim, (int, float))
                    else None,
                )
            )

        ranked = rank_candidates(service=self.ranker, candidates=candidates, req=req)
        return ReadResponse(
            family=self.family,
            items=tuple(ranked),
            coverage_status=coverage_status_from_count(
                found=len(ranked), requested=req.max_items
            ),
            meta={"candidate_pool": len(rows)},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_scope_for_overlap(scope: Mapping[str, Any]) -> dict[str, str]:
    """Pick the scope keys preferences care about; lowercase + strip."""
    interesting = (
        "language",
        "framework",
        "repo",
        "service",
        "file_path",
        "path",
        "symbol",
        "function_name",
        "audience",
        "environment",
    )
    out: dict[str, str] = {}
    for key in interesting:
        val = scope.get(key)
        if isinstance(val, str) and val.strip():
            out[key] = val.strip().lower()
    return out


def _scope_overlap(row: ClaimRow, task_scope: Mapping[str, str]) -> float:
    """Hierarchical task-scope ↔ rule-scope overlap in [0, 1] (R4).

    The rule's scope is ``row.properties['code_scope']``. Matching is by
    containment, not flat equality: a repo-wide rule applies to a file in that
    repo, and a ``src/payments/**`` rule applies to ``src/payments/client.py``.
    A rule with no scope is global (0.5).
    """
    rule_scope_raw = row.properties.get("code_scope")
    if not isinstance(rule_scope_raw, Mapping):
        return 0.5 if not task_scope else 0.3
    rule_scope = {
        k: v for k, v in rule_scope_raw.items() if isinstance(v, str) and v.strip()
    }
    if not rule_scope:
        return 0.5
    return hierarchical_scope_overlap(task_scope, rule_scope)


def _corroboration(row: ClaimRow) -> int:
    count = row.properties.get("corroboration_count")
    if isinstance(count, int) and count > 0:
        return count
    return 1


def _make_candidate_key(row: ClaimRow) -> str:
    return row.claim_key or f"{row.predicate}:{row.subject_key}:{row.object_key}"


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        "claim_key": row.claim_key,
        "subgraph": row.subgraph,
        "truth": row.truth,
        "fact": row.fact,
        "source_refs": list(row.source_refs),
        "source_system": row.source_system,
        "valid_at": row.valid_at.isoformat() if row.valid_at else None,
        "valid_until": row.valid_until.isoformat() if row.valid_until else None,
        "observed_at": row.observed_at.isoformat() if row.observed_at else None,
        "evidence_strength": row.evidence_strength,
    }
    # Surface common preference fields the agent will want to see
    for key in ("policy_kind", "code_scope", "strength", "audience", "prescription"):
        if key in row.properties:
            payload[key] = row.properties[key]
    return payload


__all__ = ["CodingPreferencesReader"]
