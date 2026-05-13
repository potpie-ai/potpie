"""Evaluator for the ingestion / ontology axis.

Inputs: the post-ingest graph snapshot + ingestion outcomes from the
real reconciliation pipeline. Asserts that the right entities and edges
exist with canonical labels and that reconciliation completed cleanly.

Scoring:
- Each entity assertion is worth 100 / N_entity_assertions points (capped).
- Each edge assertion is worth 100 / N_edge_assertions points (capped).
- Reconciliation health is a multiplier in [0, 1] applied to the total.
- ``no_orphan_entities`` is a hard gate: if violated, score halves.

Pass: score >= 75 AND no reconciliation errors that exceeded the budget.
"""

from __future__ import annotations

import re
from typing import Any

from benchmarks.core.graph_inspect import GraphSnapshot
from benchmarks.core.ingestion import IngestionOutcome
from benchmarks.core.scenario import (
    EdgeAssertion,
    EntityAssertion,
    PostIngestAssertions,
)
from benchmarks.evaluators.base import EvaluationResult


def _entity_matches(entity: object, assertion: EntityAssertion) -> bool:
    label = getattr(entity, "label", "")
    key = getattr(entity, "key", "")
    props: dict[str, Any] = getattr(entity, "properties", {}) or {}
    if label != assertion.label:
        return False
    if assertion.key_pattern and not re.search(assertion.key_pattern, key or ""):
        return False
    for prop_key, expected in assertion.where.items():
        actual = props.get(prop_key)
        if isinstance(expected, dict):
            # Single-key match expressions: { contains: "...", regex: "...", equals: ... }.
            if "contains" in expected:
                if not isinstance(actual, str) or expected["contains"] not in actual:
                    return False
            if "regex" in expected:
                if not isinstance(actual, str) or not re.search(expected["regex"], actual):
                    return False
            if "equals" in expected and actual != expected["equals"]:
                return False
        elif prop_key.endswith("_contains"):
            base = prop_key[: -len("_contains")]
            actual = props.get(base)
            if not isinstance(actual, str) or expected not in actual:
                return False
        else:
            if actual != expected:
                return False
    return True


def _count_entity_matches(snapshot: GraphSnapshot, assertion: EntityAssertion) -> int:
    return sum(1 for e in snapshot.entities if _entity_matches(e, assertion))


def _count_edge_matches(snapshot: GraphSnapshot, assertion: EdgeAssertion) -> int:
    return sum(
        1
        for edge in snapshot.edges
        if edge.type == assertion.type
        and edge.from_label == assertion.from_label
        and edge.to_label == assertion.to_label
    )


def _orphan_count(snapshot: GraphSnapshot) -> int:
    referenced: set[tuple[str, str]] = set()
    for edge in snapshot.edges:
        referenced.add((edge.from_label, edge.from_key))
        referenced.add((edge.to_label, edge.to_key))
    return sum(1 for e in snapshot.entities if (e.label, e.key) not in referenced)


def evaluate_ingestion_quality(
    *,
    snapshot: GraphSnapshot,
    outcomes: list[IngestionOutcome],
    assertions: PostIngestAssertions,
) -> EvaluationResult:
    errors: list[str] = []
    reco_failed = sum(1 for o in outcomes if o.error is not None)
    reco_downgrades = sum(o.soft_downgrades for o in outcomes)
    details: dict[str, object] = {
        "entities_total": len(snapshot.entities),
        "edges_total": len(snapshot.edges),
        "events_total": len(outcomes),
        "events_failed": reco_failed,
        "soft_downgrades_total": reco_downgrades,
    }

    over_failed = max(0, reco_failed - assertions.reconciliation.failed_events_max)
    over_downgrades = max(0, reco_downgrades - assertions.reconciliation.soft_downgrades_max)
    if over_failed:
        errors.append(
            f"reconciliation: {reco_failed} failed events (budget {assertions.reconciliation.failed_events_max})"
        )
        for o in outcomes:
            if o.error:
                errors.append(f"  {o.fixture_path}: {o.error}")
    if over_downgrades:
        errors.append(
            f"reconciliation: {reco_downgrades} soft downgrades (budget {assertions.reconciliation.soft_downgrades_max})"
        )
    reco_multiplier = 1.0
    if over_failed:
        reco_multiplier *= max(0.0, 1.0 - 0.25 * over_failed)
    if over_downgrades:
        reco_multiplier *= max(0.0, 1.0 - 0.10 * over_downgrades)

    entity_assertions = assertions.graph_must_contain_entities
    edge_assertions = assertions.graph_must_contain_edges
    n_struct = len(entity_assertions) + len(edge_assertions)
    if n_struct == 0:
        # No structural assertions — score is 100 unless reconciliation was over budget.
        structural_score = 100.0
    else:
        per_assertion = 100.0 / n_struct
        structural_score = 0.0
        for ea in entity_assertions:
            count = _count_entity_matches(snapshot, ea)
            if count >= ea.min_count:
                structural_score += per_assertion
            else:
                errors.append(
                    f"entity missing: label={ea.label} "
                    f"key_pattern={ea.key_pattern} where={ea.where} "
                    f"required>={ea.min_count} got={count}"
                )
        for edge in edge_assertions:
            count = _count_edge_matches(snapshot, edge)
            if count >= edge.min_count:
                structural_score += per_assertion
            else:
                errors.append(
                    f"edge missing: ({edge.from_label})-[{edge.type}]->({edge.to_label}) "
                    f"required>={edge.min_count} got={count}"
                )

    if assertions.no_orphan_entities:
        orphans = _orphan_count(snapshot)
        details["orphan_entities"] = orphans
        if orphans:
            errors.append(f"orphan entities present: {orphans}")
            structural_score *= 0.5

    score = max(0.0, min(100.0, structural_score * reco_multiplier))
    passed = score >= 75 and not over_failed
    return EvaluationResult(score=score, passed=passed, details=details, errors=errors)
