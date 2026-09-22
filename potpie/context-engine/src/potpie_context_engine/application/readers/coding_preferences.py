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
    EXCLUDE_KNOWLEDGE_SUBGRAPH,
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
from potpie_context_core.ports.claim_query import (
    ClaimQueryFilter,
    ClaimQueryPort,
    ClaimRow,
)
from potpie_context_core.repository_identity import normalize_repo_ref
from potpie_context_core.identity import _slugify
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
        all_rows = dedupe_claim_rows(
            self.claim_query.find_claims(
                ClaimQueryFilter(
                    pot_id=req.pot_id,
                    predicate_in=(self.predicate,),
                    include_invalidated=req.include_invalidated,
                    as_of=req.as_of,
                    source_ref_in=req.source_refs,
                    subgraph_not_in=EXCLUDE_KNOWLEDGE_SUBGRAPH,
                    # Applicability is a hard filter, so fetch the family before
                    # asking a vector backend for top-k semantic matches.
                    limit=None,
                )
            )
        )

        scope_keys = _normalise_scope_for_overlap(req.scope)
        hard_scope = graph_read_scope(req.scope)
        if "project" in scope_keys:
            hard_scope["project"] = scope_keys["project"]
        if "repo" in hard_scope:
            hard_scope["repo"] = normalize_repo_ref(hard_scope["repo"]) or ""
        scoped_rows = [
            row
            for row in all_rows
            if _row_applies(row, hard_scope=hard_scope, scope_keys=scope_keys)
        ]
        rows = self._query_within_scope(req, scoped_rows)
        candidates: list[Candidate] = []
        for row in rows:
            if not row_matches_query(
                row, req.query, threshold=req.query_threshold,
                semantic_only=req.query_threshold is not None,
            ):
                continue
            rule_scope = _rule_scope(row)
            overlap = _scope_overlap(rule_scope, scope_keys)
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
            meta={
                "ranking_omitted": max(0, len(candidates) - len(ranked)),
                "candidate_pool_unit": "claims",
                "candidate_pool": len(all_rows),
                "scoped_candidate_pool": len(scoped_rows),
            },
        )

    def _query_within_scope(
        self, req: ReadRequest, scoped_rows: list[ClaimRow]
    ) -> list[ClaimRow]:
        if not req.query or not scoped_rows:
            return scoped_rows

        keyed = {row.claim_key: row for row in scoped_rows if row.claim_key}
        semantic_rows: list[ClaimRow] = []
        if keyed:
            semantic_rows = self.claim_query.find_claims(
                ClaimQueryFilter(
                    pot_id=req.pot_id,
                    predicate_in=(self.predicate,),
                    claim_key_in=tuple(keyed),
                    include_invalidated=req.include_invalidated,
                    as_of=req.as_of,
                    source_ref_in=req.source_refs,
                    subgraph_not_in=EXCLUDE_KNOWLEDGE_SUBGRAPH,
                    fact_query=req.query,
                    limit=max(req.max_items * 4, 16),
                )
            )

        # Legacy rows without claim keys cannot be constrained in the query
        # port. Keep them eligible for lexical matching without reopening the
        # semantic search to unrelated scopes.
        legacy_rows = [row for row in scoped_rows if not row.claim_key]
        return dedupe_claim_rows([*semantic_rows, *legacy_rows])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_scope_for_overlap(scope: Mapping[str, Any]) -> dict[str, str]:
    """Pick the scope keys preferences care about; lowercase + strip."""
    interesting = (
        "language",
        "framework",
        "project",
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
            normalized = val.strip().lower()
            if key == "repo":
                normalized = normalize_repo_ref(val)
            elif key == "project":
                normalized = _slugify(normalized.removeprefix("project:"))
            elif key == "service":
                normalized = _slugify(normalized.removeprefix("service:"))
            out[key] = normalized
    return out


def _scope_overlap(
    rule_scope: Mapping[str, str] | None, task_scope: Mapping[str, str]
) -> float:
    """Hierarchical task-scope ↔ rule-scope overlap in [0, 1] (R4).

    The rule's scope combines its canonical relation target with explicit
    refinements. Matching is by containment, not flat equality: a repo-wide
    rule applies to a file in that repo, and a ``src/payments/**`` rule applies
    to ``src/payments/client.py``. A rule with no scope is global (0.5).
    """
    if rule_scope is None:
        return 0.5 if not task_scope else 0.3
    if not rule_scope:
        return 0.5
    return hierarchical_scope_overlap(task_scope, rule_scope)


def _row_applies(
    row: ClaimRow,
    *,
    hard_scope: Mapping[str, str],
    scope_keys: Mapping[str, str],
) -> bool:
    rule_scope = _rule_scope(row)
    if _scope_conflicts(row, hard_scope, rule_scope=rule_scope):
        return False
    if hard_scope and rule_scope is None:
        # The relation is scoped, but neither its canonical target nor explicit
        # metadata says where. Treating it as global would guess applicability.
        return False
    return not (scope_keys and _scope_overlap(rule_scope, scope_keys) == 0.0)


def _scope_conflicts(
    row: ClaimRow,
    task_scope: Mapping[str, str],
    *,
    rule_scope: Mapping[str, str] | None,
) -> bool:
    if not task_scope:
        return False
    if (
        rule_scope is not None
        and "project" in task_scope
        and "project" in rule_scope
        and task_scope["project"] != rule_scope["project"]
    ):
        return True
    return rule_scope is not None and code_scope_conflicts(task_scope, rule_scope)


def _rule_scope(row: ClaimRow) -> dict[str, str] | None:
    """Return the rule's authoritative target plus explicit refinements.

    ``POLICY_APPLIES_TO`` already names the repository, service, or code asset
    governed by a rule.  ``code_scope`` adds dimensions such as language and
    environment; it is not a second, competing source of target identity.
    Contradictory duplicate metadata is left in the merged scope so normal
    conflict checking rejects it for either target rather than guessing.
    """
    target_scope = _scope_from_target(row.object_key)
    raw = row.properties.get("code_scope")
    raw_items = raw.items() if isinstance(raw, Mapping) else ()
    explicit = {
        str(key): (
            normalize_repo_ref(value) if str(key) == "repo" else value.strip().lower()
        )
        for key, value in raw_items
        if isinstance(value, str) and value.strip()
    }
    if row.environment:
        explicit.setdefault("environment", row.environment.strip().lower())
    if "project" in explicit:
        explicit["project"] = _slugify(explicit["project"].removeprefix("project:"))
    if "service" in explicit:
        explicit["service"] = _slugify(explicit["service"].removeprefix("service:"))

    if target_scope is None:
        return explicit or None
    if not target_scope:  # scope:any is an explicitly shared rule.
        return explicit

    merged = dict(target_scope)
    for key, value in explicit.items():
        if key not in merged:
            merged[key] = value
        elif key in {"file_path", "path"} and _equivalent_paths(merged[key], value):
            # CodeAsset keys can be relative to a package while explicit scope
            # is relative to the repository root.  Keep the more specific
            # spelling when one is a complete trailing path of the other.
            merged[key] = max((merged[key], value), key=len)
        elif merged[key] != value:
            # Preserve both facts as an impossible value.  This ensures the
            # rule cannot match either contradictory scope accidentally.
            merged[key] = f"{merged[key]}\0{value}"
    return merged


def _equivalent_paths(left: str, right: str) -> bool:
    left = left.strip("/")
    right = right.strip("/")
    return left == right or left.endswith(f"/{right}") or right.endswith(f"/{left}")


def _scope_from_target(target: str) -> dict[str, str] | None:
    value = target.strip().lower()
    if value == "scope:any":
        return {}
    for key in ("project", "repo", "service"):
        prefix = f"{key}:"
        if value.startswith(prefix) and len(value) > len(prefix):
            scoped_value = value[len(prefix) :]
            if key == "repo":
                scoped_value = normalize_repo_ref(scoped_value)
            elif key in {"project", "service"}:
                scoped_value = _slugify(scoped_value)
            return {key: scoped_value}
    if not value.startswith("code:"):
        return None

    body = value[5:]
    if body.startswith("project:") and len(body) > len("project:"):
        return {"project": _slugify(body[len("project:") :])}
    if body.startswith("service:"):
        parts = body.split(":", 2)
        if len(parts) == 3 and all(parts[1:]):
            return {"service": parts[1], "file_path": parts[2]}
        return None
    if body.startswith("repo:"):
        body = body[5:]
    if ":" not in body:
        return None
    repo, path = body.split(":", 1)
    if not repo or not path:
        return None
    return {"repo": normalize_repo_ref(repo), "file_path": path}


def _payload_from_row(row: ClaimRow) -> dict[str, Any]:
    payload = claim_payload(row, extra={"properties": dict(row.properties or {})})
    # Surface common preference fields the agent will want to see
    for key in ("policy_kind", "code_scope", "strength", "audience", "prescription"):
        if key in row.properties:
            payload[key] = row.properties[key]
    return payload


__all__ = ["CodingPreferencesReader"]
