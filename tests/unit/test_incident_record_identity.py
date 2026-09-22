"""W3 regression tests for incident occurrence and repository identity."""

# ruff: noqa: S101

from __future__ import annotations

from potpie_context_core.api import ClaimQueryFilter
from potpie_context_core.ports.agent_context import RecordRequest
from potpie_context_engine.testing import build_test_graph_runtime


POT_ID = "w3-identity-test"


def _record_fix(runtime, *, service: str, source: str, root_cause: str) -> None:
    receipt = runtime.record(
        RecordRequest(
            pot_id=POT_ID,
            record_type="fix",
            summary="Connection timeout",
            details={"root_cause": root_cause, "fix_steps": [f"repair {service}"]},
            source_refs=(source,),
            scope={"service": service},
        )
    )
    assert receipt.accepted


def test_same_titled_incidents_keep_separate_fixes_causes_and_evidence() -> None:
    runtime = build_test_graph_runtime()
    _record_fix(
        runtime,
        service="payments",
        source="incident:payments-42",
        root_cause="Payments leaked pooled connections",
    )
    _record_fix(
        runtime,
        service="inventory",
        source="incident:inventory-77",
        root_cause="Inventory held a write lock",
    )

    claims = runtime.backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT_ID))
    resolved = [claim for claim in claims if claim.predicate == "RESOLVED"]
    assert len(resolved) == 2
    assert len({claim.subject_key for claim in resolved}) == 2

    properties = {
        claim.subject_key: runtime.backend.claim_query.entity_properties(
            pot_id=POT_ID, entity_key=claim.subject_key
        )
        for claim in resolved
    }
    assert {props["root_cause"] for props in properties.values()} == {
        "Payments leaked pooled connections",
        "Inventory held a write lock",
    }
    assert {tuple(claim.source_refs) for claim in resolved} == {
        ("incident:payments-42",),
        ("incident:inventory-77",),
    }


def test_same_occurrence_replay_does_not_create_another_fix() -> None:
    runtime = build_test_graph_runtime()
    for _ in range(2):
        _record_fix(
            runtime,
            service="payments",
            source="incident:payments-42",
            root_cause="Payments leaked pooled connections",
        )

    claims = runtime.backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT_ID))
    resolved = [claim for claim in claims if claim.predicate == "RESOLVED"]
    assert len(resolved) == 1


def test_explicit_shared_bug_pattern_keeps_occurrence_fixes_distinct() -> None:
    runtime = build_test_graph_runtime()
    for service, fix_id, source in (
        ("payments", "fix:payments-42", "incident:payments-42"),
        ("inventory", "fix:inventory-77", "incident:inventory-77"),
    ):
        receipt = runtime.record(
            RecordRequest(
                pot_id=POT_ID,
                record_type="fix",
                summary="Connection timeout",
                details={
                    "fix_id": fix_id,
                    "bug_pattern_id": "bug_pattern:connection-timeout",
                    "fix_steps": [f"repair {service}"],
                },
                source_refs=(source,),
                scope={"service": service},
            )
        )
        assert receipt.accepted

    claims = runtime.backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT_ID))
    resolved = [claim for claim in claims if claim.predicate == "RESOLVED"]
    assert {claim.subject_key for claim in resolved} == {
        "fix:payments-42",
        "fix:inventory-77",
    }
    assert {claim.object_key for claim in resolved} == {
        "bug_pattern:connection-timeout"
    }
