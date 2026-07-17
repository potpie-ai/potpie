"""GraphBackend capability conformance.

Every backend profile must satisfy the same capability contract. The
``in_memory`` and ``embedded`` profiles implement all six capabilities (the POC
substrates); the ``embedded`` snapshot/persistence path is exercised too. The
``neo4j`` and ``embedded`` profiles that leave projections unbuilt must fail
*closed* with ``CapabilityNotImplemented`` — never a bare ``NotImplementedError``
or a silent wrong answer.

This is the suite the docs' "add a conformance run for every GraphBackend"
hook points at; new backends drop into ``FULL_PROFILES`` / ``PARTIAL_PROFILES``.
"""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.graph.backends import build_backend
from potpie_context_engine.adapters.outbound.graph.inbox_stores.local_json import LocalJsonGraphInboxStore
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import LocalJsonGraphPlanStore
from potpie_context_core.workbench_service import GraphWorkbenchService
from potpie_context_core.context_events import EventRef
from potpie_context_core.errors import CapabilityNotImplemented
from potpie_context_core.graph_mutations import EdgeUpsert, EntityUpsert
from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_core.ports.graph.backend import GraphBackend
from potpie_context_core.reconciliation import ReconciliationPlan

POT = "conformance-pot"

# Profiles that implement all six capabilities end to end.
FULL_PROFILES = ["in_memory", "embedded"]

# Profiles with the canonical source-of-truth ports wired and projections
# deliberately fail-closed until implemented.
PARTIAL_PROFILES = ["neo4j", "falkordb", "falkordb_lite"]


def _build(profile, tmp_path):
    if profile == "embedded":
        from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
            EmbeddedGraphBackend,
        )

        return EmbeddedGraphBackend(home=tmp_path)
    return build_backend(profile)


def _plan(summary="prefers structured logging"):
    return ReconciliationPlan(
        event_ref=EventRef(event_id="e1", source_system="agent", pot_id=POT),
        summary=summary,
        entity_upserts=[
            EntityUpsert(entity_key="pref:logging", labels=("Preference",))
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type="RELATES_TO",
                from_entity_key="pref:logging",
                to_entity_key="svc:api",
                properties={"fact": summary},
            )
        ],
    )


def _semantic_payload() -> dict:
    return {
        "operations": [
            {
                "op": "link_entities",
                "subgraph": "infra_topology",
                "subject": {"key": "service:web", "type": "Service"},
                "predicate": "DEPENDS_ON",
                "object": {"key": "service:api", "type": "Service"},
                "truth": "source_observation",
                "evidence": [{"source_ref": "repo:manifest"}],
                "description": "web service depends on api service",
            }
        ]
    }


@pytest.mark.parametrize("profile", FULL_PROFILES)
def test_backend_satisfies_protocol(profile, tmp_path):
    backend = _build(profile, tmp_path)
    assert isinstance(backend, GraphBackend)
    assert backend.profile == profile


@pytest.mark.parametrize("profile", FULL_PROFILES)
def test_mutation_then_claim_query_round_trip(profile, tmp_path):
    backend = _build(profile, tmp_path)
    result = backend.mutation.apply(_plan(), expected_pot_id=POT)
    assert result.ok
    rows = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))
    assert len(rows) == 1
    assert rows[0].fact == "prefers structured logging"


@pytest.mark.parametrize("profile", FULL_PROFILES)
async def test_apply_async_round_trips(profile, tmp_path):
    # Every backend exposes the async write door; for the sync POC substrates it
    # is parity with apply(), and it is the only safe door for async stores
    # (Neo4j) called from inside an event loop (FastAPI/Celery).
    backend = _build(profile, tmp_path)
    result = await backend.mutation.apply_async(_plan(), expected_pot_id=POT)
    assert result.ok
    rows = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))
    assert len(rows) == 1


@pytest.mark.parametrize("profile", FULL_PROFILES)
def test_workbench_commands_conform_on_runnable_backends(profile, tmp_path):
    backend = _build(profile, tmp_path / "backend")
    workbench = GraphWorkbenchService(
        backend=backend,
        plan_store=LocalJsonGraphPlanStore(home=tmp_path / "workbench"),
        inbox_store=LocalJsonGraphInboxStore(home=tmp_path / "workbench"),
    )

    proposal = workbench.propose(_semantic_payload(), pot_id=POT)
    assert proposal.ok is True
    assert proposal.status == "validated"
    assert backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT)) == []

    committed = workbench.commit(proposal.plan_id, pot_id=POT)
    assert committed.ok is True
    assert committed.status == "committed"
    rows = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))
    assert len(rows) == 1
    assert rows[0].predicate == "DEPENDS_ON"

    history = workbench.history(pot_id=POT, plan_id=proposal.plan_id)
    assert history.ok is True
    assert any(entry.plan_id == proposal.plan_id for entry in history.entries)

    inbox_item = workbench.inbox_add(
        pot_id=POT,
        summary="Review possible graph duplicate",
        suspected_subgraphs=("infra_topology",),
    )
    assert inbox_item.ok is True
    inbox_list = workbench.inbox_list(pot_id=POT, status=("pending",))
    assert len(inbox_list.items) == 1

    quality = workbench.quality(pot_id=POT, report="summary")
    assert quality.ok is True
    assert quality.metrics["counts"]["claims"] == 1


async def test_neo4j_sync_apply_refuses_inside_event_loop():
    # The Neo4j store is async; its sync apply() bridges with asyncio.run, which
    # would corrupt the driver if run on the caller's loop. Inside a loop it must
    # refuse (pointing callers at apply_async) rather than bind to a dead loop —
    # and must not even construct the writer (so no driver is required here).
    from potpie_context_engine.adapters.outbound.graph.backends.neo4j_backend import _Neo4jMutation

    mutation = _Neo4jMutation(settings=object())
    with pytest.raises(RuntimeError, match="event loop"):
        mutation.apply(_plan(), expected_pot_id=POT)


@pytest.mark.parametrize("profile", FULL_PROFILES)
def test_capabilities_match_behavior(profile, tmp_path):
    backend = _build(profile, tmp_path)
    backend.mutation.apply(_plan(), expected_pot_id=POT)
    caps = backend.capabilities()
    assert set(caps.implemented()) == {
        "mutation",
        "claim_query",
        "semantic",
        "inspection",
        "analytics",
        "snapshot",
    }
    # each declared capability actually answers
    assert backend.semantic.search(pot_id=POT, query="logging", k=5)
    assert backend.inspection.neighborhood(pot_id=POT, entity_key="pref:logging").edges
    assert backend.analytics.counts(POT)["claims"] == 1


@pytest.mark.parametrize("profile", FULL_PROFILES)
def test_reset_clears_pot(profile, tmp_path):
    backend = _build(profile, tmp_path)
    backend.mutation.apply(_plan(), expected_pot_id=POT)
    backend.mutation.reset_pot(POT)
    assert backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT)) == []


def test_embedded_persists_across_instances(tmp_path):
    from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import EmbeddedGraphBackend

    EmbeddedGraphBackend(home=tmp_path).mutation.apply(_plan(), expected_pot_id=POT)
    # A fresh backend over the same home must see the persisted claim.
    rows = EmbeddedGraphBackend(home=tmp_path).claim_query.find_claims(
        ClaimQueryFilter(pot_id=POT)
    )
    assert len(rows) == 1


def test_snapshot_export_import_round_trip(tmp_path):
    backend = build_backend("in_memory")
    backend.mutation.apply(_plan(), expected_pot_id=POT)
    dest = str(tmp_path / "snap.json")
    manifest = backend.snapshot.export(pot_id=POT, destination=dest)
    assert manifest.claim_count == 1

    fresh = build_backend("in_memory")
    fresh.snapshot.import_(pot_id=POT, source=dest)
    assert len(fresh.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))) == 1


def test_embedded_unbuilt_profile_fails_closed():
    """A profile that has not built a capability must raise
    CapabilityNotImplemented — proven via the neo4j projections, which are
    derivable stubs (no live driver needed to construct the stub)."""
    from potpie_context_engine.adapters.outbound.graph.backends._unimplemented import UnimplementedSemantic

    stub = UnimplementedSemantic("neo4j")
    with pytest.raises(CapabilityNotImplemented) as exc:
        stub.search(pot_id=POT, query="x")
    assert exc.value.capability == "graph.neo4j.semantic.search"
    assert exc.value.recommended_next_action


@pytest.mark.parametrize("profile", PARTIAL_PROFILES)
def test_partial_backend_profiles_fail_closed_for_unbuilt_projections(
    profile, tmp_path
):
    backend = _build(profile, tmp_path)
    assert isinstance(backend, GraphBackend)
    expected = {
        "neo4j": {"mutation", "claim_query", "semantic", "analytics"},
        # FalkorDB profiles implement structural inspection (graph explorer /
        # ``potpie graph inspect``) over the canonical RELATES_TO edges.
        "falkordb": {"mutation", "claim_query", "semantic", "analytics", "inspection"},
        "falkordb_lite": {
            "mutation",
            "claim_query",
            "semantic",
            "analytics",
            "inspection",
        },
    }[profile]
    assert set(backend.capabilities().implemented()) == expected
    if "inspection" in expected:
        # Implemented: answers without raising (empty pot → empty slice).
        sl = backend.inspection.neighborhood(pot_id=POT, entity_key="pref:logging")
        assert sl.pot_id == POT
    else:
        with pytest.raises(CapabilityNotImplemented):
            backend.inspection.neighborhood(pot_id=POT, entity_key="pref:logging")
