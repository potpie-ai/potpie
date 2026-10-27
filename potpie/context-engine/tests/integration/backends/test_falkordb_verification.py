# ruff: noqa: F811 - imported pytest fixtures are injected as test parameters.
"""Native server/Lite acceptance for exact-incarnation workbench verification."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.core import workbench_service
from potpie_context_engine.core.errors import CapabilityNotImplemented
from potpie_context_engine.core.graph_mutations import EntityUpsert, ProvenanceContext
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.reconciliation import MutationBatch
from potpie_context_engine.core.workbench_service import GraphWorkbenchService

from tests.integration.mutations.test_falkordb_claim_lifecycle import store as store
from tests.integration.backends.test_falkordb_reliability import (
    graph as graph,
    snapshot,
)

pytestmark = pytest.mark.integration
JAN = datetime(2020, 1, 1, tzinfo=timezone.utc)
FEB = datetime(2020, 2, 1, tzinfo=timezone.utc)
MAR = datetime(2020, 3, 1, tzinfo=timezone.utc)
SOURCE = "service:payments-api"
TARGET = "service:ledger-api"


@pytest.fixture()
def workbench(store, tmp_path):
    return GraphWorkbenchService(
        backend=store,
        plan_store=LocalJsonGraphPlanStore(home=tmp_path / "plans"),
    )


def assertion(point=JAN):
    return {
        "operations": [
            {
                "op": "link_entities",
                "subgraph": "infra_topology",
                "subject": {"key": SOURCE, "type": "Service"},
                "predicate": "DEPENDS_ON",
                "object": {"key": TARGET, "type": "Service"},
                "truth": "source_observation",
                "evidence": [{"source_ref": "repo:manifest"}],
                "description": "payments depends on ledger",
                "valid_from": point.isoformat(),
            }
        ]
    }


def closure(point=FEB, *, target=TARGET):
    return {
        "operations": [
            {
                "op": "end_relation_validity",
                "subgraph": "infra_topology",
                "subject": {"key": SOURCE, "type": "Service"},
                "predicate": "DEPENDS_ON",
                "object": {"key": target, "type": "Service"},
                "reason": "dependency removed",
                "valid_until": point.isoformat(),
            }
        ]
    }


def commit(workbench, payload, **kwargs):
    proposal = workbench.propose(payload, pot_id="p")
    assert proposal.ok, proposal.to_dict()
    result = workbench.commit(
        proposal.plan_id, pot_id="p", approved_by="native-verification-test", **kwargs
    )
    assert result.ok and result.status == "committed", result.to_dict()
    return proposal, result


def all_claims(store):
    return store.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p", include_invalidated=True, include_retired=True)
    )


def receipt(graph, mutation_id):
    rows = graph.ro_query(
        "MATCH (r:PotpieMutationReceipt {pot_id:'p',mutation_id:$mid}) "
        "RETURN properties(r)",
        params={"mid": mutation_id},
    ).result_set
    assert len(rows) == 1
    return rows[0][0]


def revision(graph):
    return graph.ro_query(
        "MATCH (v:PotpieRevision {pot_id:'p'}) RETURN v.version"
    ).result_set[0][0]


def assert_report(report, *, outcome, checked_revision):
    assert report.state == "completed", report.to_dict()
    assert report.outcome == outcome, report.to_dict()
    assert report.checked_revision == checked_revision
    assert report.verified_at is not None
    assert report.verified_at.utcoffset() == timedelta(0)


def test_old_incarnation_verifies_as_retracted_after_close_and_same_key_reassertion(
    workbench, store, graph
):
    first, original = commit(workbench, assertion(), verify=True)
    assert original.verification_state == "completed"
    assert_report(
        original.verification, outcome="matching", checked_revision=revision(graph)
    )
    old = all_claims(store)[0]
    persisted_receipt = receipt(graph, original.mutation_id)
    original_execution = original.execution.to_dict()
    commit(workbench, closure(), verify=True)
    current_proposal, current_commit = commit(workbench, assertion(MAR), verify=True)
    current = next(row for row in all_claims(store) if row.record_id != old.record_id)
    assert current.claim_key == old.claim_key
    report = workbench.verify_commit(first.plan_id, pot_id="p")
    assert_report(
        report, outcome="superseded_retracted", checked_revision=revision(graph)
    )
    outcomes = report.content_readback["claim_outcomes"]
    assert [(item["record_id"], item["outcome"]) for item in outcomes] == [
        (old.record_id, "superseded_retracted")
    ]
    assert outcomes[0]["lifecycle"]["state"] == "closed"
    assert_report(
        workbench.verify_commit(current_proposal.plan_id, pot_id="p"),
        outcome="matching",
        checked_revision=revision(graph),
    )
    assert original.execution.to_dict() == original_execution
    assert receipt(graph, original.mutation_id) == persisted_receipt
    assert original.execution.version_after < current_commit.execution.version_after


def test_delete_only_and_frozen_empty_corrections_verify_their_exact_selection(
    workbench, store, graph
):
    commit(workbench, assertion(), verify=False)
    old = all_claims(store)[0]
    removed, result = commit(workbench, closure(), verify=True)
    record = workbench.plan_store.get(pot_id="p", plan_id=removed.plan_id)
    assert record.lowered_batch.edge_upserts == []
    assert result.verification_state == "completed"
    assert_report(
        result.verification, outcome="matching", checked_revision=revision(graph)
    )
    assert result.verification.content_readback["checked_claim_record_ids"] == (
        old.record_id,
    )
    assert (
        result.verification.content_readback["claim_outcomes"][0]["outcome"]
        == "matching"
    )
    commit(
        workbench,
        {
            "operations": [
                {
                    "op": "upsert_entity",
                    "subject": {"key": "service:unused", "type": "Service"},
                }
            ]
        },
        verify=False,
    )
    empty, empty_result = commit(
        workbench,
        {
            "operations": [
                {
                    "op": "retract_claim",
                    "subgraph": "infra_topology",
                    "subject": {"key": "service:unused", "type": "Service"},
                    "reason": "empty reviewed entity selection",
                    "valid_until": MAR.isoformat(),
                }
            ]
        },
        verify=True,
    )
    empty_record = workbench.plan_store.get(pot_id="p", plan_id=empty.plan_id)
    assert empty_record.lowered_batch.invalidations[0].closures == ()
    assert_report(
        empty_result.verification, outcome="matching", checked_revision=revision(graph)
    )
    assert empty_result.verification.content_readback["claim_outcomes"] == ()
    assert empty_result.verification.content_readback["checked_claim_record_ids"] == ()


def test_later_scheduled_closure_reports_observed_future_lifecycle(workbench, graph):
    proposed, result = commit(workbench, assertion(), verify=False)
    original = receipt(graph, result.mutation_id)
    endpoint = datetime.now(timezone.utc) + timedelta(days=2)
    commit(workbench, closure(endpoint), verify=False)
    report = workbench.verify_commit(proposed.plan_id, pot_id="p")
    assert_report(
        report, outcome="superseded_retracted", checked_revision=revision(graph)
    )
    lifecycle = report.content_readback["claim_outcomes"][0]["lifecycle"]
    assert lifecycle["state"] == "scheduled_close"
    assert datetime.fromisoformat(lifecycle["invalid_at"]) == endpoint
    assert report.verified_at < endpoint
    assert receipt(graph, result.mutation_id) == original


@pytest.mark.parametrize("field", ("source_system", "source_ref", "valid_at"))
def test_scalar_source_or_original_start_corruption_is_a_mismatch(
    workbench, store, graph, field
):
    proposed, result = commit(workbench, assertion(), verify=True)
    row = all_claims(store)[0]
    persisted_receipt = receipt(graph, result.mutation_id)
    patch = (
        {"valid_at": MAR.isoformat(), "valid_from": MAR.isoformat()}
        if field == "valid_at"
        else {field: "unexpected-source"}
    )
    graph.query(
        "MATCH ()-[r:RELATES_TO {group_id:'p',record_id:$rid}]->() SET r+=$patch",
        params={"rid": row.record_id, "patch": patch},
    )
    report = workbench.verify_commit(proposed.plan_id, pot_id="p")
    assert_report(report, outcome="mismatched", checked_revision=revision(graph))
    assert report.ok is False
    assert report.content_readback["claim_outcomes"][0]["record_id"] == row.record_id
    assert field in report.content_readback["mismatches"][0]["fields"]
    assert receipt(graph, result.mutation_id) == persisted_receipt


@pytest.mark.parametrize(
    "patch,diagnostic_fields",
    (
        ({"invalid_at": "not-a-time"}, ("invalid_at",)),
        ({"valid_until": "not-a-time"}, ("valid_until",)),
        ({"invalid_at": "2020-02-01T00:00:00"}, ("invalid_at",)),
        ({"valid_until": "2020-02-01T00:00:00"}, ("valid_until",)),
        ({"valid_at": MAR.isoformat()}, ("valid_at", "valid_from")),
        ({"valid_until": "2019-12-31T00:00:00Z"}, ("valid_until",)),
    ),
    ids=(
        "malformed-invalid-at",
        "malformed-valid-until",
        "naive-invalid-at",
        "naive-valid-until",
        "contradictory-start-aliases",
        "nonpositive-interval",
    ),
)
@pytest.mark.parametrize(
    "content_mismatch", (False, True), ids=("temporal-only", "source-also-changed")
)
def test_unprovable_stored_times_are_inconclusive_unless_content_also_mismatches(
    workbench, store, graph, patch, diagnostic_fields, content_mismatch
):
    proposed, result = commit(workbench, assertion(), verify=True)
    row = all_claims(store)[0]
    original = receipt(graph, result.mutation_id)
    corruption = {
        **patch,
        **({"source_ref": "unexpected-source"} if content_mismatch else {}),
    }
    graph.query(
        "MATCH ()-[r:RELATES_TO {group_id:'p',record_id:$rid}]->() SET r+=$patch",
        params={"rid": row.record_id, "patch": corruption},
    )
    evidence = all_claims(store)[0]
    assert set(diagnostic_fields).issubset(evidence.temporal_diagnostics)
    before_verification = snapshot(graph)
    report = workbench.verify_commit(proposed.plan_id, pot_id="p")
    assert report.outcome == ("mismatched" if content_mismatch else "inconclusive"), (
        report.to_dict()
    )
    assert report.state == ("completed" if content_mismatch else "deferred")
    assert report.ok is False
    assert report.checked_revision == revision(graph)
    outcome = report.content_readback["claim_outcomes"][0]
    assert outcome["record_id"] == row.record_id
    assert outcome["outcome"] == report.outcome
    lifecycle = outcome["lifecycle"]
    assert lifecycle["state"] == "unknown"
    assert set(diagnostic_fields).issubset(lifecycle["temporal_diagnostics"])
    assert not {"valid_at", "invalid_at", "valid_until"}.intersection(lifecycle)
    if content_mismatch:
        assert "source_ref" in report.content_readback["mismatches"][0]["fields"]
    else:
        assert report.content_readback["mismatches"] == ()
    assert snapshot(graph) == before_verification
    assert receipt(graph, result.mutation_id) == original


@pytest.mark.parametrize(
    "stored_field,reported_field,value",
    (
        ("name", "predicate", "OWNED_BY"),
        ("subject_key", "subject_key", "service:unexpected-subject"),
        ("object_key", "object_key", "service:unexpected-object"),
    ),
)
def test_correction_verification_checks_frozen_predicate_and_endpoints(
    workbench, store, graph, stored_field, reported_field, value
):
    commit(workbench, assertion(), verify=False)
    row = all_claims(store)[0]
    proposed, result = commit(workbench, closure(), verify=True)
    record = workbench.plan_store.get(pot_id="p", plan_id=proposed.plan_id)
    assert record.lowered_batch.invalidations[0].target_edge == (
        "DEPENDS_ON",
        SOURCE,
        TARGET,
    )
    original = receipt(graph, result.mutation_id)
    graph.query(
        "MATCH ()-[r:RELATES_TO {group_id:'p',record_id:$rid}]->() SET r+=$patch",
        params={"rid": row.record_id, "patch": {stored_field: value}},
    )
    before_verification = snapshot(graph)
    report = workbench.verify_commit(proposed.plan_id, pot_id="p")
    assert_report(report, outcome="mismatched", checked_revision=revision(graph))
    assert report.ok is False
    assert report.content_readback["checked_claim_record_ids"] == (row.record_id,)
    assert reported_field in report.content_readback["mismatches"][0]["fields"]
    assert snapshot(graph) == before_verification
    assert receipt(graph, result.mutation_id) == original


def test_new_same_key_incarnation_cannot_satisfy_missing_old_record(
    workbench, store, graph
):
    first, result = commit(workbench, assertion(), verify=False)
    old = all_claims(store)[0]
    persisted_receipt = receipt(graph, result.mutation_id)
    commit(workbench, closure(), verify=False)
    newer, _ = commit(workbench, assertion(MAR), verify=False)
    graph.query(
        "MATCH ()-[r:RELATES_TO {group_id:'p',record_id:$rid}]->() DELETE r",
        params={"rid": old.record_id},
    )
    report = workbench.verify_commit(first.plan_id, pot_id="p")
    assert_report(report, outcome="mismatched", checked_revision=revision(graph))
    assert report.content_readback["claim_outcomes"][0]["record_id"] == old.record_id
    assert (
        report.content_readback["mismatches"][0]["fields"]["incarnation"]["actual"] == 0
    )
    assert_report(
        workbench.verify_commit(newer.plan_id, pot_id="p"),
        outcome="matching",
        checked_revision=revision(graph),
    )
    assert receipt(graph, result.mutation_id) == persisted_receipt


def test_revision_change_during_readback_defers_instead_of_claiming_coherence(
    workbench, store, graph, monkeypatch
):
    proposed, result = commit(workbench, assertion(), verify=False)
    persisted_receipt = receipt(graph, result.mutation_id)
    find = store.claim_query.find_claims
    changed = False

    def concurrent_change(filter_):
        nonlocal changed
        rows = find(filter_)
        if filter_.record_id_in and not changed:
            changed = True
            store.mutation.apply(
                MutationBatch(
                    entity_upserts=[EntityUpsert("service:concurrent", ("Service",))]
                ),
                expected_pot_id="p",
                provenance_context=ProvenanceContext(
                    mutation_id="concurrent-verification"
                ),
            )
        return rows

    monkeypatch.setattr(store.claim_query, "find_claims", concurrent_change)
    report = workbench.verify_commit(proposed.plan_id, pot_id="p")
    assert changed
    assert report.state == "deferred" and report.outcome == "inconclusive"
    assert report.checked_revision is None
    assert revision(graph) == result.execution.version_after + 1
    assert receipt(graph, result.mutation_id) == persisted_receipt


@pytest.mark.parametrize("failure", (RuntimeError, asyncio.CancelledError))
def test_postreceipt_verification_failure_preserves_successful_native_execution(
    workbench, graph, monkeypatch, failure
):
    def unavailable(*args, **kwargs):
        raise failure("injected post-receipt verification failure")

    monkeypatch.setattr(workbench_service, "_verify_ingestion_commit", unavailable)
    proposed, result = commit(workbench, assertion(), verify=True)
    assert result.verification_state == "deferred"
    assert result.verification.state == "deferred"
    assert result.verification.outcome == "inconclusive"
    assert revision(graph) == result.execution.version_after
    original = receipt(graph, result.mutation_id)
    status = workbench.commit_status(proposed.plan_id, pot_id="p")
    assert status.ok and status.execution == result.execution
    assert receipt(graph, result.mutation_id) == original


@pytest.mark.parametrize(
    "verify,deferred,state", ((False, False, "not_requested"), (True, True, "deferred"))
)
def test_commit_verification_intent_is_separate_from_receipt(
    workbench, graph, verify, deferred, state
):
    _, result = commit(
        workbench, assertion(), verify=verify, defer_verification=deferred
    )
    assert result.verification_state == state
    assert result.verification is None
    assert result.execution.version_after == revision(graph)
    assert result.to_dict()["verification_state"] == state


def test_unsupported_exact_read_is_reported_once_without_reapplying(
    workbench, store, graph, monkeypatch
):
    proposed, result = commit(workbench, assertion(), verify=False)
    before = snapshot(graph)
    find = store.claim_query.find_claims
    calls = 0

    def unsupported(filter_):
        nonlocal calls
        if filter_.record_id_in:
            calls += 1
            raise CapabilityNotImplemented("test.exact_verification_read")
        return find(filter_)

    monkeypatch.setattr(store.claim_query, "find_claims", unsupported)
    report = workbench.verify_commit(proposed.plan_id, pot_id="p")
    assert calls == 1
    assert report.state == "unsupported" and report.outcome == "inconclusive"
    assert report.checked_revision is None
    assert result.execution.version_after == revision(graph)
    assert snapshot(graph) == before
