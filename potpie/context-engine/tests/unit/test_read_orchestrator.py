"""The single read trunk: intent → include routing → P9 readers → envelope."""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.graph.in_memory_reader import InMemoryClaimQueryStore
from potpie_context_engine.application.services.read_orchestrator import ReadOrchestrator
from potpie_context_engine.domain.ports.claim_query import ClaimRow

pytestmark = pytest.mark.unit


def _store_with_pref() -> InMemoryClaimQueryStore:
    store = InMemoryClaimQueryStore()
    store.add(
        ClaimRow(
            pot_id="p1",
            predicate="POLICY_APPLIES_TO",
            subject_key="policy:structlog",
            object_key="scope:global",
            evidence_strength="attested",
            fact="use structlog with kw-args, never f-strings",
            properties={"policy_kind": "logging"},
        )
    )
    return store


def test_resolve_routes_backed_include_to_reader() -> None:
    orch = ReadOrchestrator(claim_query=_store_with_pref())
    env = orch.resolve(pot_id="p1", include=["coding_preferences"])
    assert [i.include for i in env.items] == ["coding_preferences"]
    assert env.items[0].payload["fact"].startswith("use structlog")
    assert env.unsupported_includes == ()


def test_owners_include_is_backed_and_empty_when_no_claims() -> None:
    orch = ReadOrchestrator(claim_query=InMemoryClaimQueryStore())
    env = orch.resolve(pot_id="p1", include=["owners"])
    assert env.unsupported_includes == ()
    assert env.items == ()


def test_unknown_include_is_unsupported_unknown() -> None:
    orch = ReadOrchestrator(claim_query=InMemoryClaimQueryStore())
    env = orch.resolve(pot_id="p1", include=["totally_made_up"])
    names = {u.name: u.reason for u in env.unsupported_includes}
    assert names.get("totally_made_up") == "unknown_include"


def test_raw_graph_returns_generic_related_to_edges() -> None:
    # The semantic readers filter to typed predicates; raw_graph returns the
    # whole partition incl. generic RELATED_TO so the graph explorer can render
    # downgraded data that no UC reader matches.
    store = InMemoryClaimQueryStore()
    store.add(
        ClaimRow(
            pot_id="p1",
            predicate="RELATED_TO",
            subject_key="timeline:activity:repo-added:abc",
            object_key="github:repo:org/name",
            evidence_strength="inferred",
            fact="repo added",
        )
    )
    orch = ReadOrchestrator(claim_query=store)

    raw = orch.resolve(pot_id="p1", include=["raw_graph"])
    assert [i.include for i in raw.items] == ["raw_graph"]
    payload = raw.items[0].payload
    assert payload["predicate"] == "RELATED_TO"
    assert payload["subject_key"] == "timeline:activity:repo-added:abc"
    assert payload["object_key"] == "github:repo:org/name"
    assert raw.unsupported_includes == ()

    # Same data is invisible to a semantic reader (typed predicates only).
    infra = orch.resolve(pot_id="p1", include=["infra_topology"])
    assert infra.items == ()


def test_intent_expands_to_backed_readers() -> None:
    orch = ReadOrchestrator(claim_query=_store_with_pref())
    env = orch.resolve(pot_id="p1", intent="feature")
    # feature default includes coding_preferences plus other backed graph
    # readers; empty backed readers return no items, not unsupported includes.
    assert "coding_preferences" in {i.include for i in env.items}
    assert env.unsupported_includes == ()
