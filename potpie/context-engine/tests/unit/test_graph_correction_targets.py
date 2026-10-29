"""Corrections must change exactly the claims shown by their proposal."""

import pytest

from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.workbench_service import GraphWorkbenchService
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)


POT = "correction-targets"
pytestmark = pytest.mark.unit


@pytest.fixture(params=["memory", "embedded"])
def graph(request, tmp_path):
    backend = (
        InMemoryGraphBackend()
        if request.param == "memory"
        else EmbeddedGraphBackend(home=tmp_path / "graph")
    )
    store = LocalJsonGraphPlanStore(home=tmp_path / "plans")
    return GraphWorkbenchService(backend=backend, plan_store=store), backend, store


def _claim(environment="prod", source="incident:original", target="service:ledger"):
    return {
        "op": "link_entities",
        "subgraph": "infra_topology",
        "subject": {"key": "service:payments", "type": "Service"},
        "predicate": "DEPENDS_ON",
        "object": {"key": target, "type": "Service"},
        "environment": environment,
        "truth": "source_observation",
        "valid_from": "2025-01-01T00:00:00+00:00",
        "evidence": [{"source_ref": source}],
        "description": "Payments calls ledger in this environment",
    }


def _seed(workbench):
    plan = workbench.propose(
        {
            "operations": [
                _claim(),
                _claim("staging"),
                _claim(target="service:fraud"),
            ]
        },
        pot_id=POT,
    )
    assert plan.ok
    assert workbench.commit(plan.plan_id, pot_id=POT).ok


def _rows(backend):
    return backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=POT, include_invalidated=True)
    )


def _correction(op):
    result = _claim()
    result.update(
        op=op,
        reason="Production dependency moved",
        valid_until="2026-01-01T00:00:00+00:00",
    )
    if op == "supersede_claim":
        result.pop("valid_from")
        result["superseded_by"] = {"key": "service:ledger-v2", "type": "Service"}
    return result


@pytest.mark.parametrize(
    "op", ["end_relation_validity", "retract_claim", "supersede_claim"]
)
def test_environment_correction_matches_preview_and_preserves_other_claims(graph, op):
    workbench, backend, store = graph
    _seed(workbench)
    before = _rows(backend)
    selected = next(
        row
        for row in before
        if row.environment == "prod" and row.object_key == "service:ledger"
    )
    unaffected = {row.claim_key: row for row in before if row != selected}
    proposal = workbench.propose(
        {"operations": [_correction(op)]}, pot_id=POT, approved_by="test:reviewer"
    )
    assert proposal.ok
    assert proposal.diff.retracted_claim_keys == (selected.claim_key,)
    # Reload the persisted plan to ensure commit uses the previewed selection.
    record = store.get(pot_id=POT, plan_id=proposal.plan_id)
    assert record.lowered_batch.invalidations[0].target_claim_keys == (
        selected.claim_key,
    )
    receipt = workbench.commit(proposal.plan_id, pot_id=POT, verify=True)
    assert receipt.ok
    after = {row.claim_key: row for row in _rows(backend)}
    assert after[selected.claim_key].invalid_at is not None
    assert {key: after[key] for key in unaffected} == unaffected
    if op == "supersede_claim":
        assert any(
            row.object_key == "service:ledger-v2" and row.environment == "prod"
            for row in after.values()
        )


def test_ambiguous_environment_requires_exact_claim_selection(graph):
    workbench, backend, _ = graph
    _seed(workbench)
    op = _correction("end_relation_validity")
    del op["environment"]
    proposal = workbench.propose(
        {"operations": [op]}, pot_id=POT, approved_by="test:reviewer"
    )
    assert not proposal.ok
    assert any(
        issue["code"] == "ambiguous_correction_target" for issue in proposal.issues
    )
    assert all(row.invalid_at is None for row in _rows(backend))


def test_exact_claim_keys_disambiguate_same_environment_sources(graph):
    workbench, backend, _ = graph
    _seed(workbench)
    extra = workbench.propose(
        {"operations": [_claim(source="incident:other")]}, pot_id=POT
    )
    assert workbench.commit(extra.plan_id, pot_id=POT).ok
    op = _correction("retract_claim")
    ambiguous = workbench.propose(
        {"operations": [op]}, pot_id=POT, approved_by="test:reviewer"
    )
    assert not ambiguous.ok
    selected = next(row for row in _rows(backend) if row.source_ref == "incident:other")
    op["target_claim_keys"] = [selected.claim_key]
    proposal = workbench.propose(
        {"operations": [op]}, pot_id=POT, approved_by="test:reviewer"
    )
    assert proposal.ok
    assert workbench.commit(proposal.plan_id, pot_id=POT).ok
    assert [row.claim_key for row in _rows(backend) if row.invalid_at is not None] == [
        selected.claim_key
    ]


def test_explicit_claim_cannot_escape_environment_or_endpoints(graph):
    workbench, backend, _ = graph
    _seed(workbench)
    op = _correction("retract_claim")
    op["target_claim_keys"] = [
        next(row.claim_key for row in _rows(backend) if row.environment == "staging")
    ]
    proposal = workbench.propose(
        {"operations": [op]}, pot_id=POT, approved_by="test:reviewer"
    )
    assert not proposal.ok
    assert all(row.invalid_at is None for row in _rows(backend))


def test_missing_environment_is_its_own_exact_scope(graph):
    workbench, backend, _ = graph
    _seed(workbench)
    proposal = workbench.propose({"operations": [_claim(environment=None)]}, pot_id=POT)
    assert workbench.commit(proposal.plan_id, pot_id=POT).ok
    selected = next(row for row in _rows(backend) if row.environment is None)
    op = _correction("retract_claim")
    op.pop("environment")
    proposed = workbench.propose(
        {"operations": [op]}, pot_id=POT, approved_by="reviewer"
    )
    assert proposed.ok
    assert [closure.target.record_id for closure in proposed.diff.claim_closures] == [
        selected.record_id
    ]
    assert workbench.commit(proposed.plan_id, pot_id=POT).ok
    assert all(
        row.invalid_at is None for row in _rows(backend) if row.environment is not None
    )


def test_explicit_historical_incarnation_and_target_source_match_preview(graph):
    workbench, backend, _ = graph
    _seed(workbench)
    target = next(
        row
        for row in _rows(backend)
        if row.environment == "prod" and row.object_key == "service:ledger"
    )
    first = workbench.propose(
        {"operations": [_correction("retract_claim")]},
        pot_id=POT,
        approved_by="reviewer",
    )
    assert workbench.commit(first.plan_id, pot_id=POT).ok
    op = _correction("retract_claim")
    op.update(
        target_claim_record_ids=[target.record_id],
        target_claim_keys=[target.claim_key],
        target_source_system=target.source_system,
        target_source_ref=target.source_ref,
        valid_until="2026-02-01T00:00:00Z",
    )
    proposal = workbench.propose(
        {"operations": [op]}, pot_id=POT, approved_by="reviewer"
    )
    assert proposal.ok
    assert workbench.commit(proposal.plan_id, pot_id=POT, verify=True).ok
    corrected = next(row for row in _rows(backend) if row.record_id == target.record_id)
    assert corrected.valid_at == target.valid_at
    assert corrected.invalid_at.month == 2
    op["target_source_ref"] = "different-source"
    rejected = workbench.propose(
        {"operations": [op]}, pot_id=POT, approved_by="reviewer"
    )
    assert not rejected.ok


def test_supersede_rejects_conflicting_cutovers(graph):
    workbench, _, _ = graph
    _seed(workbench)
    op = _correction("supersede_claim")
    op["valid_from"] = "2025-01-01T00:00:00Z"
    proposal = workbench.propose(
        {"operations": [op]}, pot_id=POT, approved_by="reviewer"
    )
    assert not proposal.ok
    assert "same cutover" in proposal.detail


def test_each_supersession_binds_its_own_replacement_despite_shared_endpoints(graph):
    workbench, backend, store = graph
    seed = workbench.propose(
        {"operations": [_claim(source="source:a"), _claim(source="source:b")]},
        pot_id=POT,
    )
    assert workbench.commit(seed.plan_id, pot_id=POT).ok
    old = sorted(_rows(backend), key=lambda row: row.source_ref)
    operations = []
    for index, row in enumerate(old):
        op = _correction("supersede_claim")
        op.update(
            target_claim_record_ids=[row.record_id],
            target_claim_keys=[row.claim_key],
            evidence=[{"source_ref": f"new-evidence:{index}"}],
        )
        operations.append(op)
    proposal = workbench.propose(
        {"operations": operations}, pot_id=POT, approved_by="reviewer"
    )
    assert proposal.ok
    batch = store.get(pot_id=POT, plan_id=proposal.plan_id).lowered_batch
    replacements = [
        edge for edge in batch.edge_upserts if edge.edge_type == "DEPENDS_ON"
    ]
    effects = {
        edge.properties["supersedes_record_id"]: edge
        for edge in batch.edge_upserts
        if edge.edge_type == "SUPERSEDES"
    }
    for index, previous in enumerate(old):
        replacement = next(
            edge
            for edge in replacements
            if edge.properties["source_ref"] == f"new-evidence:{index}"
        )
        assert (
            effects[previous.record_id].properties["replacement_record_id"]
            == replacement.resolution.record_id
        )
    assert workbench.commit(proposal.plan_id, pot_id=POT, verify=True).ok


@pytest.mark.parametrize("selector", ["target_claim_keys", "target_claim_record_ids"])
def test_authored_duplicate_correction_identities_are_rejected(graph, selector):
    workbench, backend, _ = graph
    _seed(workbench)
    row = next(
        row
        for row in _rows(backend)
        if row.environment == "prod" and row.object_key == "service:ledger"
    )
    op = _correction("retract_claim")
    value = row.claim_key if selector == "target_claim_keys" else row.record_id
    op[selector] = [value, value]
    proposal = workbench.propose(
        {"operations": [op]}, pot_id=POT, approved_by="reviewer"
    )
    assert not proposal.ok
    assert "must not repeat" in proposal.detail
