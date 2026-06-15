"""PriorBugsReader (UC4 / P9).

Inputs: symptom signature (free-text or structured) + scope. Logic:
direct native-vector query (via the claim-query port's ``fact_query``)
over ``BugPattern``-related claim edges filtered to live edges and the
agent's scope-or-broader. Rank by ``verification × scope-overlap ×
recency × corroboration`` (the ranker's standard formula, with the
verification signal mapped onto the corroboration factor).

Surfaces both worked fixes (``RESOLVED`` claims) and attempted-failed
fixes (``ATTEMPTED_FIX_FAILED`` claims), the latter labeled so the
agent doesn't repeat them. Hides narrower-scope bugs from broader-
scope queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from application.readers._common import (
    ReadRequest,
    ReadResponse,
    claim_candidate_key,
    claim_payload,
    coverage_status_from_count,
    rank_candidates,
    row_in_anchor_set,
    service_anchor_keys,
)
from domain.ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow
from domain.ranking import Candidate, RankingService


_BUG_PREDICATES: tuple[str, ...] = (
    "RESOLVED",
    "ATTEMPTED_FIX_FAILED",
    "VERIFIED",
    "REPRODUCES",
)


@dataclass(slots=True)
class PriorBugsReader:
    claim_query: ClaimQueryPort
    ranker: RankingService
    family: str = "prior_bugs"

    def read(self, req: ReadRequest) -> ReadResponse:
        anchor_keys = service_anchor_keys(req.scope)

        rows = self.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=req.pot_id,
                predicate_in=_BUG_PREDICATES,
                include_invalidated=req.include_invalidated,
                as_of=req.as_of,
                fact_query=req.query,
                limit=max(req.max_items * 4, 24),
            )
        )
        rows = self._expand_bug_neighborhood(req, rows=rows, anchor_keys=anchor_keys)

        # Verification-count lookup: a Fix with two VERIFIED claims
        # (different sources) beats a Fix with none. Rather than re-
        # querying, we count VERIFIED rows in the same result set and
        # roll the count into the corroboration_count of the related
        # RESOLVED row.
        verification_counts = _count_verifications(rows)

        candidates: list[Candidate] = []
        bug_scope = _bug_scope_overlaps(rows, anchor_keys=anchor_keys)
        fix_bug = _fix_bug_index(rows)
        for row in rows:
            if row.predicate == "VERIFIED":
                # Verifications fold into their target Fix's score; not
                # surfaced as standalone candidates.
                continue
            overlap = _scope_overlap(
                row,
                anchor_keys=anchor_keys,
                bug_scope=bug_scope,
                fix_bug=fix_bug,
            )
            if anchor_keys and overlap == 0.0:
                continue
            sim = row.properties.get("semantic_similarity")
            fix_key = _fix_key_for_row(row)
            verification_boost = verification_counts.get(fix_key, 0) if fix_key else 0
            candidates.append(
                Candidate(
                    candidate_key=claim_candidate_key(row),
                    payload=_payload_from_row(row, verifications=verification_boost),
                    strength=row.evidence_strength,
                    valid_at=row.valid_at,
                    corroboration_count=1 + verification_boost,
                    scope_overlap=overlap if anchor_keys else None,
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
            meta={
                "anchor_keys": list(anchor_keys),
                "candidate_pool": len(rows),
            },
        )

    def _expand_bug_neighborhood(
        self,
        req: ReadRequest,
        *,
        rows: list[ClaimRow],
        anchor_keys: Iterable[str],
    ) -> list[ClaimRow]:
        """Add bug/fix neighbor rows for matched symptom rows.

        Symptom search often matches only the ``REPRODUCES`` row. The agent
        still needs the fix and failed attempts, so expand by BugPattern key
        before ranking and inline-relation assembly.
        """
        if not rows:
            return rows
        bug_keys = set(_bug_keys(rows))
        fix_keys = set(_fix_keys(rows))
        expanded: list[ClaimRow] = list(rows)

        if bug_keys:
            expanded.extend(
                self.claim_query.find_claims(
                    ClaimQueryFilter(
                        pot_id=req.pot_id,
                        predicate_in=("REPRODUCES",),
                        subject_key_in=tuple(sorted(bug_keys)),
                        include_invalidated=req.include_invalidated,
                        as_of=req.as_of,
                        limit=max(len(bug_keys) * 8, 32),
                    )
                )
            )

        scope_rows = [row for row in expanded if row.predicate == "REPRODUCES"]
        if anchor_keys:
            scoped_bug_keys = {
                row.subject_key
                for row in scope_rows
                if _scope_overlap(row, anchor_keys=anchor_keys) > 0.0
            }
        else:
            scoped_bug_keys = {row.subject_key for row in scope_rows}
        if not scope_rows:
            scoped_bug_keys = set(bug_keys)

        if scoped_bug_keys:
            fix_rows = self.claim_query.find_claims(
                ClaimQueryFilter(
                    pot_id=req.pot_id,
                    predicate_in=("RESOLVED", "ATTEMPTED_FIX_FAILED"),
                    object_key_in=tuple(sorted(scoped_bug_keys)),
                    include_invalidated=req.include_invalidated,
                    as_of=req.as_of,
                    limit=max(len(scoped_bug_keys) * 8, 32),
                )
            )
            expanded.extend(fix_rows)
            fix_keys.update(_fix_keys(fix_rows))

        if fix_keys:
            # Canonical VERIFIED is actor/activity -> Fix. Some legacy/test rows
            # used Fix -> actor; read both so old memory still contributes.
            expanded.extend(
                self.claim_query.find_claims(
                    ClaimQueryFilter(
                        pot_id=req.pot_id,
                        predicate_in=("VERIFIED",),
                        object_key_in=tuple(sorted(fix_keys)),
                        include_invalidated=req.include_invalidated,
                        as_of=req.as_of,
                        limit=max(len(fix_keys) * 8, 32),
                    )
                )
            )
            expanded.extend(
                self.claim_query.find_claims(
                    ClaimQueryFilter(
                        pot_id=req.pot_id,
                        predicate_in=("VERIFIED",),
                        subject_key_in=tuple(sorted(fix_keys)),
                        include_invalidated=req.include_invalidated,
                        as_of=req.as_of,
                        limit=max(len(fix_keys) * 8, 32),
                    )
                )
            )

        return _dedupe_rows(expanded)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scope_overlap(
    row: ClaimRow,
    *,
    anchor_keys: Iterable[str],
    bug_scope: Mapping[str, float] | None = None,
    fix_bug: Mapping[str, str] | None = None,
) -> float:
    if not anchor_keys:
        return 0.5
    anchors = set(anchor_keys)
    row_scope = row.properties.get("scope_keys")
    if isinstance(row_scope, list):
        hits = sum(1 for k in row_scope if k in anchors)
        if hits:
            return min(1.0, hits / max(len(anchors), 1))
    if row_in_anchor_set(row, anchors):
        return 1.0
    if bug_scope:
        if row.object_key in bug_scope:
            return bug_scope[row.object_key]
        bug_key = fix_bug.get(row.subject_key) if fix_bug else None
        if bug_key and bug_key in bug_scope:
            return bug_scope[bug_key]
    return 0.0


def _count_verifications(rows: Iterable[ClaimRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.predicate == "VERIFIED":
            fix_key = _fix_key_for_row(row)
            if fix_key:
                counts[fix_key] = counts.get(fix_key, 0) + 1
    return counts


def _bug_scope_overlaps(
    rows: Iterable[ClaimRow], *, anchor_keys: Iterable[str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        if row.predicate != "REPRODUCES":
            continue
        out[row.subject_key] = max(
            out.get(row.subject_key, 0.0),
            _scope_overlap(row, anchor_keys=anchor_keys),
        )
    return out


def _fix_bug_index(rows: Iterable[ClaimRow]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        if row.predicate in {"RESOLVED", "ATTEMPTED_FIX_FAILED"}:
            out[row.subject_key] = row.object_key
    return out


def _bug_keys(rows: Iterable[ClaimRow]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        if row.predicate == "REPRODUCES":
            keys.add(row.subject_key)
        elif row.predicate in {"RESOLVED", "ATTEMPTED_FIX_FAILED"}:
            keys.add(row.object_key)
    return sorted(keys)


def _fix_keys(rows: Iterable[ClaimRow]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        fix_key = _fix_key_for_row(row)
        if fix_key:
            keys.add(fix_key)
    return sorted(keys)


def _fix_key_for_row(row: ClaimRow) -> str | None:
    if row.predicate in {"RESOLVED", "ATTEMPTED_FIX_FAILED"}:
        return row.subject_key
    if row.predicate == "VERIFIED":
        if row.object_key.startswith("fix:"):
            return row.object_key
        if row.subject_key.startswith("fix:"):
            return row.subject_key
    return None


def _dedupe_rows(rows: Iterable[ClaimRow]) -> list[ClaimRow]:
    seen: set[str] = set()
    out: list[ClaimRow] = []
    for row in rows:
        key = claim_candidate_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _payload_from_row(row: ClaimRow, *, verifications: int) -> dict[str, Any]:
    return claim_payload(
        row,
        extra={
            "is_attempted_failed_fix": row.predicate == "ATTEMPTED_FIX_FAILED",
            "verification_count": verifications,
        },
    )


__all__ = ["PriorBugsReader"]
