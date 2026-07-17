"""S6 local-profile completion: record→predicate mapping.

These exercise the additive local-profile work: that a recorded preference lands
on its ontology predicate and surfaces in the matching reader.
"""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.claim_query_analytics import ClaimQueryAnalytics
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from potpie.services.host_wiring import build_host_shell
from potpie_context_core.domain.ports.agent_context import RecordRequest, ResolveRequest


@pytest.fixture()
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    return build_host_shell(backend=InMemoryGraphBackend())


def test_recorded_preference_surfaces_in_coding_preferences(host):
    pot = host.pots.create_pot(name="default", use=True)
    receipt = host.agent_context.record(
        RecordRequest(
            pot_id=pot.pot_id,
            record_type="preference",
            summary="Always use ruff for linting",
            details={
                "policy_kind": "style",
                "prescription": "Always use ruff for linting",
                "code_scope": {"language": "python"},
            },
            scope={"language": "python", "service": "context-engine"},
        )
    )
    assert receipt.accepted

    env = host.agent_context.resolve(
        ResolveRequest(
            pot_id=pot.pot_id,
            intent="feature",
            include=("coding_preferences",),
            scope={"language": "python"},
        )
    )
    # The record landed as a POLICY_APPLIES_TO claim, not a generic RELATES_TO,
    # so the coding_preferences reader serves it.
    prefs = [i for i in env.items if i.include == "coding_preferences"]
    assert prefs, "recorded preference did not surface in coding_preferences"
    assert "ruff" in dict(prefs[0].payload).get("fact", "")


def test_free_form_record_falls_back_to_related_to(host):
    pot = host.pots.create_pot(name="default", use=True)
    # ``workflow`` has no emits_predicate → RELATED_TO fallback; still recorded.
    receipt = host.agent_context.record(
        RecordRequest(
            pot_id=pot.pot_id,
            record_type="workflow",
            summary="Run make deploy to ship",
        )
    )
    assert receipt.accepted and receipt.mutations_applied >= 1
    env = host.agent_context.resolve(
        ResolveRequest(pot_id=pot.pot_id, include=("raw_graph",))
    )
    assert any("make deploy" in dict(i.payload).get("fact", "") for i in env.items)


def test_claim_query_analytics_counts_match_backend():
    backend = InMemoryGraphBackend()
    analytics = ClaimQueryAnalytics(backend.claim_query)
    # Empty pot: zero counts, empty quality.
    assert analytics.counts("p1")["claims"] == 0
    assert analytics.quality("p1")["status"] == "empty"
