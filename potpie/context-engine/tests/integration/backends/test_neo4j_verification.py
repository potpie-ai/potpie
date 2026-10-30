"""Workbench verification observes native history without rewriting execution."""

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import os
from types import SimpleNamespace
import uuid

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.neo4j_backend import (
    Neo4jGraphBackend,
)
from potpie_context_engine.core.errors import CapabilityNotImplemented
from potpie_context_engine.core.graph_mutations import EdgeDelete
from potpie_context_engine.core.reconciliation import MutationBatch
from potpie_context_engine.core.workbench_service import GraphWorkbenchService
from potpie_context_engine.testing import InMemoryGraphPlanStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("PROTOCOL_TEST_NEO4J_URI"),
        reason="PROTOCOL_TEST_NEO4J_URI must name a disposable test server",
    ),
]
JAN = datetime(2020, 1, 1, tzinfo=timezone.utc)
FEB = datetime(2020, 2, 1, tzinfo=timezone.utc)
MAR = datetime(2020, 3, 1, tzinfo=timezone.utc)


@pytest.fixture
def graph():
    from neo4j import GraphDatabase

    settings = SimpleNamespace(
        is_enabled=lambda: True,
        neo4j_uri=lambda: os.environ["PROTOCOL_TEST_NEO4J_URI"],
        neo4j_user=lambda: os.environ.get("PROTOCOL_TEST_NEO4J_USERNAME", "neo4j"),
        neo4j_password=lambda: os.environ.get("PROTOCOL_TEST_NEO4J_PASSWORD", ""),
    )
    backend = Neo4jGraphBackend(settings)
    pot = "verification-" + uuid.uuid4().hex
    store = InMemoryGraphPlanStore()
    workbench = GraphWorkbenchService(backend=backend, plan_store=store)
    with GraphDatabase.driver(
        settings.neo4j_uri(), auth=(settings.neo4j_user(), settings.neo4j_password())
    ) as driver:
        try:
            yield pot, backend, driver, workbench, store
        finally:
            backend.claim_query.close()
            _run(
                driver,
                "MATCH (n) WHERE n.group_id=$pot OR n.pot_id=$pot DETACH DELETE n",
                pot=pot,
            )


def _run(driver, query, **params):
    with driver.session() as session:
        return [dict(row) for row in session.run(query, **params)]


def _link(*, point=JAN):
    return {
        "operations": [
            {
                "op": "link_entities",
                "subgraph": "infra_topology",
                "environment": "prod",
                "subject": {"key": "service:api", "type": "Service"},
                "predicate": "DEPENDS_ON",
                "object": {"key": "service:ledger", "type": "Service"},
                "truth": "source_observation",
                "evidence": [{"source_ref": "repo:dependencies"}],
                "description": "API depends on ledger",
                "valid_from": point.isoformat(),
            }
        ]
    }


def _close():
    operation = deepcopy(_link()["operations"][0])
    operation.update(
        op="end_relation_validity",
        valid_until=FEB.isoformat(),
        reason="dependency removed",
    )
    operation.pop("valid_from")
    return {"operations": [operation]}


def _commit(graph, payload=None, **options):
    pot, _, _, workbench, _ = graph
    proposal = workbench.propose(payload or _link(), pot_id=pot)
    assert proposal.ok, proposal.to_dict()
    result = workbench.commit(
        proposal.plan_id, pot_id=pot, approved_by="test:reviewer", **options
    )
    assert result.ok, result.to_dict()
    return proposal, result


def _receipts(graph):
    pot, _, driver, _, _ = graph
    return _run(
        driver,
        "MATCH (r:PotpieMutationReceipt {pot_id:$pot}) "
        "RETURN r.mutation_id AS mutation_id,r.payload AS payload ORDER BY mutation_id",
        pot=pot,
    )


def _outcomes(verification):
    return {
        row["record_id"]: row for row in verification.content_readback["claim_outcomes"]
    }


def test_later_close_and_reassert_verify_old_incarnation_without_changing_receipt(
    graph,
):
    pot, backend, _, workbench, store = graph
    first, committed = _commit(graph, verify=True)
    original_id = committed.execution.affected_claim_record_ids[0]
    assert committed.verification_state == committed.verification.state
    assert committed.to_dict()["verification_state"] == committed.verification.state
    assert committed.verification.state == "unsupported"
    assert committed.verification.outcome == "matching"
    assert committed.verification.checked_revision == 1
    original_receipt = next(
        row for row in _receipts(graph) if row["mutation_id"] == committed.mutation_id
    )
    _commit(graph, _close())
    reasserted, newer = _commit(graph, _link(point=MAR), verify=True)
    new_id = newer.execution.affected_claim_record_ids[0]
    assert new_id != original_id
    assert newer.verification.outcome == "matching"
    assert newer.verification.checked_revision == 3

    checked = workbench.verify_commit(first.plan_id, pot_id=pot)
    assert checked.outcome == "superseded_retracted"
    assert checked.checked_revision == backend.mutation.current_version(pot) == 3
    assert set(_outcomes(checked)) == {original_id}
    assert _outcomes(checked)[original_id]["outcome"] == "superseded_retracted"
    assert _outcomes(checked)[original_id]["lifecycle"]["state"] == "closed"
    assert workbench.verify_commit(reasserted.plan_id, pot_id=pot).outcome == "matching"
    assert original_receipt in _receipts(graph)
    replay = workbench.commit(first.plan_id, pot_id=pot)
    assert replay.execution == committed.execution
    assert backend.mutation.current_version(pot) == 3
    stored = store.get(pot_id=pot, plan_id=first.plan_id)
    assert stored.execution == committed.execution


@pytest.mark.parametrize("field", ["source_system", "source_ref", "valid_at"])
def test_physical_source_or_start_corruption_is_content_mismatch(graph, field):
    pot, backend, driver, workbench, _ = graph
    proposal, committed = _commit(graph)
    original_id = committed.execution.affected_claim_record_ids[0]
    receipts = _receipts(graph)
    if field == "valid_at":
        assignment, value = "r.valid_at=$value,r.valid_from=$value", FEB.isoformat()
    else:
        assignment, value = f"r.{field}=$value", "unrelated-source"
    _run(
        driver,
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() "
        "WHERE coalesce(r.record_id,r.uuid)=$record_id SET " + assignment,
        pot=pot,
        record_id=original_id,
        value=value,
    )
    checked = workbench.verify_commit(proposal.plan_id, pot_id=pot)
    assert not checked.ok
    assert checked.outcome == "mismatched"
    assert checked.checked_revision == 1
    mismatch = checked.content_readback["mismatches"][0]
    assert mismatch["record_id"] == original_id
    assert field in mismatch["fields"]
    assert _receipts(graph) == receipts
    assert backend.mutation.current_version(pot) == 1


@pytest.mark.parametrize(
    "assignment,value,diagnostic_field",
    [
        ("r.invalid_at=$value", "not-a-time", "invalid_at"),
        ("r.valid_until=$value", "not-a-time", "valid_until"),
        ("r.invalid_at=$value", "2020-02-01T00:00:00", "invalid_at"),
        ("r.valid_until=$value", "2020-02-01T00:00:00", "valid_until"),
        ("r.valid_from=$value", FEB.isoformat(), "valid_from"),
        (
            "r.valid_at=datetime($value),r.valid_from=datetime($value)",
            "2020-01-01T00:00:00.000000001Z",
            "valid_at",
        ),
    ],
    ids=[
        "malformed-invalid-at",
        "malformed-valid-until",
        "naive-invalid-at",
        "naive-valid-until",
        "contradictory-start-alias",
        "native-nanosecond",
    ],
)
def test_ambiguous_persisted_times_are_inconclusive_with_unknown_lifecycle(
    graph, assignment, value, diagnostic_field
):
    pot, backend, driver, workbench, _ = graph
    proposal, committed = _commit(graph)
    original_id = committed.execution.affected_claim_record_ids[0]
    receipts = _receipts(graph)
    _run(
        driver,
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() "
        "WHERE coalesce(r.record_id,r.uuid)=$record_id SET " + assignment,
        pot=pot,
        record_id=original_id,
        value=value,
    )
    before = _run(
        driver,
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() RETURN properties(r) AS props",
        pot=pot,
    )
    checked = workbench.verify_commit(proposal.plan_id, pot_id=pot)
    assert not checked.ok
    assert checked.state == "deferred"
    assert checked.outcome == "inconclusive"
    assert checked.checked_revision == 1
    outcome = _outcomes(checked)[original_id]
    assert outcome["outcome"] == "inconclusive"
    assert outcome["lifecycle"]["state"] == "unknown"
    assert diagnostic_field in outcome["lifecycle"]["temporal_diagnostics"]
    assert checked.content_readback["mismatches"] == ()
    assert (
        _run(
            driver,
            "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() RETURN properties(r) AS props",
            pot=pot,
        )
        == before
    )
    assert _receipts(graph) == receipts
    assert backend.mutation.current_version(pot) == 1
    assert (
        workbench.commit_status(proposal.plan_id, pot_id=pot).execution
        == committed.execution
    )


def test_source_content_mismatch_wins_over_malformed_interval_uncertainty(graph):
    pot, backend, driver, workbench, _ = graph
    proposal, committed = _commit(graph)
    original_id = committed.execution.affected_claim_record_ids[0]
    receipts = _receipts(graph)
    _run(
        driver,
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() "
        "WHERE coalesce(r.record_id,r.uuid)=$record_id "
        "SET r.invalid_at='not-a-time',r.source_ref='repo:changed'",
        pot=pot,
        record_id=original_id,
    )
    checked = workbench.verify_commit(proposal.plan_id, pot_id=pot)
    assert not checked.ok
    assert checked.outcome == "mismatched"
    assert checked.checked_revision == 1
    outcome = _outcomes(checked)[original_id]
    assert outcome["outcome"] == "mismatched"
    assert outcome["lifecycle"]["state"] == "unknown"
    assert "invalid_at" in outcome["lifecycle"]["temporal_diagnostics"]
    assert "source_ref" in checked.content_readback["mismatches"][0]["fields"]
    assert _receipts(graph) == receipts
    assert backend.mutation.current_version(pot) == 1


@pytest.mark.parametrize(
    "stored_field,expected_field",
    [
        ("name", "predicate"),
        ("subject_key", "subject_key"),
        ("object_key", "object_key"),
    ],
)
def test_correction_verifies_its_known_frozen_predicate_and_endpoints(
    graph, stored_field, expected_field
):
    pot, backend, driver, workbench, store = graph
    _, asserted = _commit(graph)
    proposal, corrected = _commit(graph, _close(), verify=True)
    assert corrected.verification.outcome == "matching"
    prepared = store.get(pot_id=pot, plan_id=proposal.plan_id)
    assert not prepared.lowered_batch.edge_upserts
    assert prepared.lowered_batch.invalidations[0].target_edge == (
        "DEPENDS_ON",
        "service:api",
        "service:ledger",
    )
    original_id = asserted.execution.affected_claim_record_ids[0]
    receipts = _receipts(graph)
    _run(
        driver,
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() "
        "WHERE coalesce(r.record_id,r.uuid)=$record_id "
        f"SET r.{stored_field}=$value",
        pot=pot,
        record_id=original_id,
        value="unrelated:target",
    )
    checked = workbench.verify_commit(proposal.plan_id, pot_id=pot)
    assert not checked.ok
    assert checked.outcome == "mismatched"
    assert checked.checked_revision == 2
    assert _outcomes(checked)[original_id]["outcome"] == "mismatched"
    mismatch = checked.content_readback["mismatches"][0]
    assert mismatch["record_id"] == original_id
    assert expected_field in mismatch["fields"]
    assert _receipts(graph) == receipts
    assert backend.mutation.current_version(pot) == 2


@pytest.mark.parametrize("keyless", [False, True], ids=["keyed", "keyless"])
def test_delete_only_verification_checks_exact_frozen_physical_target(graph, keyless):
    pot, backend, driver, _, store = graph
    props = {
        "group_id": pot,
        "record_id": "legacy:dependency",
        "name": "DEPENDS_ON",
        "subject_key": "service:api",
        "object_key": "service:ledger",
        "subgraph": "infra_topology",
        "environment": "prod",
        "valid_at": JAN.isoformat(),
        "truth": "source_observation",
    }
    if not keyless:
        props["claim_key"] = "claim:legacy:dependency"
    _run(
        driver,
        "CREATE (a:Entity:Service {group_id:$pot,entity_key:'service:api'}),"
        "(b:Entity:Service {group_id:$pot,entity_key:'service:ledger'}),"
        "(a)-[r:RELATES_TO]->(b) SET r=$props",
        pot=pot,
        props=props,
    )

    def lower_delete(_request, plan):
        plan.batch = MutationBatch(
            edge_deletes=[EdgeDelete("DEPENDS_ON", "service:api", "service:ledger")]
        )

    workbench = GraphWorkbenchService(
        backend=backend, plan_store=store, lowerer=lower_delete
    )
    # The public semantic correction policy requires a logical key. The
    # generic batch lowerer remains able to freeze an exact keyless delete.
    proposal = workbench.propose(_link(), pot_id=pot)
    assert proposal.ok, proposal.to_dict()
    prepared = store.get(pot_id=pot, plan_id=proposal.plan_id)
    assert not prepared.lowered_batch.edge_upserts
    assert (
        prepared.lowered_batch.edge_deletes[0].closures[0].target.record_id
        == props["record_id"]
    )
    committed = workbench.commit(
        proposal.plan_id, pot_id=pot, approved_by="test:reviewer", verify=True
    )
    assert committed.ok, committed.to_dict()
    checked = committed.verification
    assert committed.verification_state == checked.state
    assert checked.outcome == "matching"
    assert checked.checked_revision == 1
    assert set(_outcomes(checked)) == {props["record_id"]}
    assert _outcomes(checked)[props["record_id"]]["claim_key"] == props.get("claim_key")
    assert (
        checked.content_readback["checked_correction_targets"][0]["record_id"]
        == props["record_id"]
    )


def test_empty_frozen_correction_never_expands_verification_to_unrelated_claims(
    graph, monkeypatch
):
    pot, backend, _, workbench, store = graph
    _commit(graph)

    def lower_empty(_request, plan):
        plan.batch = MutationBatch(
            edge_deletes=[EdgeDelete("DEPENDS_ON", "service:absent", "service:none")]
        )

    workbench = GraphWorkbenchService(
        backend=backend, plan_store=store, lowerer=lower_empty
    )
    proposal = workbench.propose(_close(), pot_id=pot)
    assert proposal.ok, proposal.to_dict()
    prepared = store.get(pot_id=pot, plan_id=proposal.plan_id)
    assert prepared.lowered_batch.edge_deletes[0].closures == ()
    committed = workbench.commit(
        proposal.plan_id, pot_id=pot, approved_by="test:reviewer"
    )
    assert committed.ok, committed.to_dict()
    queries = []
    original_find = backend.claim_query.find_claims

    def capture(filter_):
        queries.append(filter_)
        return original_find(filter_)

    monkeypatch.setattr(backend.claim_query, "find_claims", capture)
    checked = workbench.verify_commit(proposal.plan_id, pot_id=pot)
    assert checked.outcome == "matching"
    assert checked.content_readback["checked_claim_record_ids"] == ()
    assert checked.content_readback["claim_outcomes"] == ()
    assert checked.readback_count == 0
    assert not any(query.record_id_in for query in queries)
    assert not any(query.include_retired for query in queries)
    assert checked.checked_revision == backend.mutation.current_version(pot)


def test_reset_between_native_reads_defers_verification_without_false_checked_revision(
    graph, monkeypatch
):
    pot, backend, _, workbench, _ = graph
    proposal, committed = _commit(graph)
    receipts = _receipts(graph)
    original_find = backend.claim_query.find_claims
    reset = []

    def reset_after_exact_read(filter_):
        rows = original_find(filter_)
        if filter_.record_id_in and not reset:
            reset.append(
                backend.mutation.reset_pot(pot, mutation_id="verification-reset")
            )
        return rows

    monkeypatch.setattr(backend.claim_query, "find_claims", reset_after_exact_read)
    checked = workbench.verify_commit(proposal.plan_id, pot_id=pot)
    assert reset
    assert checked.state == "deferred"
    assert checked.outcome == "inconclusive"
    assert checked.checked_revision is None
    assert not checked.ok
    assert all(receipt in _receipts(graph) for receipt in receipts)
    assert backend.mutation.current_version(pot) == 2
    assert (
        workbench.commit_status(proposal.plan_id, pot_id=pot).execution
        == committed.execution
    )


def test_unsupported_exact_query_is_attempted_once_and_does_not_hide_receipt(
    graph, monkeypatch
):
    pot, backend, _, workbench, _ = graph
    proposal, committed = _commit(graph)
    receipts = _receipts(graph)
    calls = []

    def unavailable(filter_):
        calls.append(filter_)
        raise CapabilityNotImplemented("claim_query.find_claims")

    monkeypatch.setattr(backend.claim_query, "find_claims", unavailable)
    checked = workbench.verify_commit(proposal.plan_id, pot_id=pot)
    assert len(calls) == 1
    assert calls[0].record_id_in == committed.execution.affected_claim_record_ids
    assert checked.state == "unsupported"
    assert checked.outcome == "inconclusive"
    assert checked.checked_revision is None
    assert _receipts(graph) == receipts
    assert backend.mutation.current_version(pot) == 1


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_postreceipt_verification_failure_preserves_public_execution(
    graph, monkeypatch, error
):
    from potpie_context_engine.core import workbench_service

    pot, backend, _, workbench, _ = graph

    def failed_verification(*_args, **_kwargs):
        raise error("injected postreceipt verification failure")

    monkeypatch.setattr(
        workbench_service, "_verify_ingestion_commit", failed_verification
    )
    proposal, committed = _commit(graph, verify=True)
    assert committed.verification_state == "deferred"
    assert committed.verification.state == "deferred"
    assert committed.verification.outcome == "inconclusive"
    assert committed.verification.checked_revision is None
    receipts = _receipts(graph)
    replay = workbench.commit(proposal.plan_id, pot_id=pot)
    assert replay.ok
    assert replay.execution == committed.execution
    assert _receipts(graph) == receipts
    assert backend.mutation.current_version(pot) == 1


@pytest.mark.parametrize(
    "verify,defer,state", [(False, False, "not_requested"), (True, True, "deferred")]
)
def test_public_commit_distinguishes_unrequested_and_deferred_checks(
    graph, verify, defer, state
):
    _, committed = _commit(graph, verify=verify, defer_verification=defer)
    assert committed.verification_state == state
    assert committed.to_dict()["verification_state"] == state
    assert committed.verification is None


def test_standalone_verification_of_uncommitted_proposal_never_applies(
    graph, monkeypatch
):
    pot, backend, _, workbench, _ = graph
    proposal = workbench.propose(_link(), pot_id=pot)
    assert proposal.ok

    def forbidden_apply(*_args, **_kwargs):
        pytest.fail("verification must never execute an uncommitted proposal")

    monkeypatch.setattr(type(backend.mutation), "compare_and_apply", forbidden_apply)
    checked = workbench.verify_commit(proposal.plan_id, pot_id=pot)
    assert checked.status == "not_committed"
    assert checked.outcome == "inconclusive"
    assert checked.checked_revision is None
    assert backend.mutation.current_version(pot) == 0
    assert _receipts(graph) == []
