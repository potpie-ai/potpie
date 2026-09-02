"""The single read trunk: intent → include routing → P9 readers → envelope."""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.graph.in_memory_reader import (
    InMemoryClaimQueryStore,
)
from potpie_context_engine.application.services.read_orchestrator import (
    ReadOrchestrator,
)
from potpie_context_core.ports.claim_query import ClaimRow

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


def test_feature_intent_routes_feature_claims_to_features_include() -> None:
    store = InMemoryClaimQueryStore()
    store.add(
        ClaimRow(
            pot_id="p1",
            predicate="PROVIDES",
            subject_key="repo:github.com/acme/widgets",
            object_key="feature:search",
            evidence_strength="attested",
            fact="widgets repo provides search",
            properties={},
        )
    )
    orch = ReadOrchestrator(claim_query=store)

    env = orch.resolve(
        pot_id="p1",
        intent="feature",
        scope={"anchor_entity_key": "repo:github.com/acme/widgets"},
    )

    includes = [item.include for item in env.items]
    assert "features" in includes
    assert "infra_topology" not in includes


def test_infra_topology_excludes_feature_predicates() -> None:
    store = InMemoryClaimQueryStore()
    store.add(
        ClaimRow(
            pot_id="p1",
            predicate="DEFINED_IN",
            subject_key="service:search-api",
            object_key="repo:github.com/acme/widgets",
            evidence_strength="attested",
            fact="search api lives in widgets repo",
            properties={},
        )
    )
    store.add(
        ClaimRow(
            pot_id="p1",
            predicate="PROVIDES",
            subject_key="repo:github.com/acme/widgets",
            object_key="feature:search",
            evidence_strength="attested",
            fact="widgets repo provides search",
            properties={},
        )
    )
    orch = ReadOrchestrator(claim_query=store)

    env = orch.resolve(
        pot_id="p1",
        include=["infra_topology"],
        scope={"anchor_entity_key": "repo:github.com/acme/widgets"},
    )

    assert [item.include for item in env.items] == ["infra_topology"]
    assert env.items[0].payload["predicate"] == "DEFINED_IN"


# --- D10: the timeline family answers a task string ------------------------


def _store_with_activities() -> InMemoryClaimQueryStore:
    store = InMemoryClaimQueryStore()
    store.add(
        ClaimRow(
            pot_id="p1",
            predicate="TOUCHED",
            subject_key="activity:github:pr:42",
            object_key="service:inventory",
            evidence_strength="deterministic",
            fact="PR #42 fix stale stock counts after deploy",
            properties={"verb_class": "change"},
        )
    )
    store.add(
        ClaimRow(
            pot_id="p1",
            predicate="TOUCHED",
            subject_key="activity:github:pr:40",
            object_key="service:checkout",
            evidence_strength="deterministic",
            fact="PR #40 add coupon codes",
            properties={"verb_class": "change"},
        )
    )
    return store


def _timeline_coverage(env):
    return next(c for c in env.coverage if c.include == "timeline")


def test_timeline_answers_a_task_string_through_resolve() -> None:
    """The family was ``candidate_pool 0`` for every task through resolve.

    Two gates a task sentence never cleared: the absolute 0.70 similarity
    threshold no measured score reaches, and a lexical match that wanted
    every token of the sentence in the row. The reader now floors on the
    pool's own best match, the way ``docs`` does.
    """
    orch = ReadOrchestrator(claim_query=_store_with_activities())

    env = orch.resolve(pot_id="p1", intent="debugging", query="why is stock stale")

    keys = [i.candidate_key for i in env.items if i.include == "timeline"]
    assert keys == ["activity:github:pr:42"]
    assert _timeline_coverage(env).candidate_pool == 1


def test_timeline_still_answers_nothing_for_a_task_nothing_matches() -> None:
    orch = ReadOrchestrator(claim_query=_store_with_activities())

    env = orch.resolve(pot_id="p1", intent="debugging", query="does-not-exist-12345")

    assert [i for i in env.items if i.include == "timeline"] == []
    assert _timeline_coverage(env).candidate_pool == 0
