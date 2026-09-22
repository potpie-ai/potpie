"""Corrections must change exactly the claims shown by their proposal."""

import pytest

from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_core.workbench_service import GraphWorkbenchService
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
