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

from potpie_context_engine.application.readers._common import (
    ReadRequest,
    ReadResponse,
    claim_candidate_key,
    claim_corroboration,
    claim_payload,
    claim_semantic_similarity,
    code_scope_conflicts,
    coverage_status_from_count,
    dedupe_claim_rows,
    graph_read_scope,
    rank_candidates,
    row_matches_query,
)
from potpie_context_core.ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow
from potpie_context_engine.domain.ranking import Candidate, RankingService
from potpie_context_engine.domain.scope_match import hierarchical_scope_overlap


@dataclass(slots=True)
class CodingPreferencesReader:
    """Reader for the coding-preferences use case."""

    claim_query: ClaimQueryPort
    ranker: RankingService
    family: str = "coding_preferences"
    predicate: str = "POLICY_APPLIES_TO"

    def read(self, req: ReadRequest) -> ReadResponse:
        rows = dedupe_claim_rows(
            self.claim_query.find_claims(
                ClaimQueryFilter(
                    pot_id=req.pot_id,
                    predicate_in=(self.predicate,),
                    include_invalidated=req.include_invalidated,
                    as_of=req.as_of,
                    source_ref_in=req.source_refs,
                    fact_query=req.query,
                    limit=max(req.max_items * 4, 16),
                )
            )
        )

        scope_keys = _normalise_scope_for_overlap(req.scope)
        hard_scope = graph_read_scope(req.scope)
        candidates: list[Candidate] = []
        for row in rows:
            if not row_matches_query(row, req.query, threshold=req.query_threshold):
                continue
            if _scope_conflicts(row, hard_scope):
                continue
            overlap = _scope_overlap(row, scope_keys)
            if overlap == 0.0 and scope_keys:
                # Hard zero on overlap: skip — readers should not surface
                # rules that demonstrably don't apply.
                continue
            candidates.append(
                Candidate(
                    candidate_key=claim_candidate_key(row),
                    payload=_payload_from_row(row),
                    strength=row.evidence_strength,
                    valid_at=row.valid_at,
                    corroboration_count=claim_corroboration(row),
                    scope_overlap=overlap if scope_keys else None,
                    semantic_similarity=claim_semantic_similarity(row),
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


def _scope_conflicts(row: ClaimRow, task_scope: Mapping[str, str]) -> bool:
    if not task_scope:
        return False
    rule_scope_raw = row.properties.get("code_scope")
    if not isinstance(rule_scope_raw, Mapping):
        return False
    return code_scope_conflicts(task_scope, rule_scope_raw)


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    payload = claim_payload(row, extra={"properties": dict(row.properties or {})})
    # Surface common preference fields the agent will want to see
    for key in ("policy_kind", "code_scope", "strength", "audience", "prescription"):
        if key in row.properties:
            payload[key] = row.properties[key]
    return payload


__all__ = ["CodingPreferencesReader"]
