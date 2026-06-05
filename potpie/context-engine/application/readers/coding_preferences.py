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
    interesting = ("language", "framework", "repo", "service", "file_path", "audience")
    out: dict[str, str] = {}
    for key in interesting:
        val = scope.get(key)
        if isinstance(val, str) and val.strip():
            out[key] = val.strip().lower()
    return out


def _scope_overlap(row: ClaimRow, task_scope: Mapping[str, str]) -> float:
    """Compute task-scope ↔ rule-scope overlap as a [0, 1] score.

    The rule's code_scope is taken from ``row.properties['code_scope']``
    (per the P3/P6 ontology). Score = (overlap_count / max(task, rule)).
    A rule with no scope at all is treated as global (overlap = 0.5).
    """
    rule_scope_raw = row.properties.get("code_scope")
    if not isinstance(rule_scope_raw, Mapping):
        return 0.5 if not task_scope else 0.3
    rule_scope: dict[str, str] = {
        k: v.lower().strip()
        for k, v in rule_scope_raw.items()
        if isinstance(k, str) and isinstance(v, str) and v.strip()
    }
    if not rule_scope:
        return 0.5
    matching = sum(
        1 for key, value in rule_scope.items() if task_scope.get(key) == value
    )
    denominator = max(len(rule_scope), len(task_scope)) or 1
    return matching / denominator


def _corroboration(row: ClaimRow) -> int:
    count = row.properties.get("corroboration_count")
    if isinstance(count, int) and count > 0:
        return count
    return 1


def _make_candidate_key(row: ClaimRow) -> str:
    return f"{row.predicate}:{row.subject_key}:{row.object_key}:{row.source_ref or '-'}"


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        "fact": row.fact,
        "source_ref": row.source_ref,
        "source_system": row.source_system,
        "valid_at": row.valid_at.isoformat() if row.valid_at else None,
        "evidence_strength": row.evidence_strength,
    }
    # Surface common preference fields the agent will want to see
    for key in ("policy_kind", "code_scope", "strength", "audience", "prescription"):
        if key in row.properties:
            payload[key] = row.properties[key]
    return payload


__all__ = ["CodingPreferencesReader"]
