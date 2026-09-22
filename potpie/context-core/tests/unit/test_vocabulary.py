"""Canonical filter vocabulary: exact variants resolve, near misses do not."""

from __future__ import annotations

import pytest

from potpie_context_core.vocabulary import (
    close_candidates,
    local_claim_subgraphs,
    resolve_claim_subgraph,
    resolve_entity_type,
    resolve_predicate,
    vocabulary_from_catalog,
)

pytestmark = pytest.mark.unit


def test_entity_type_case_variants_resolve_to_the_canonical_label() -> None:
    for value in ("repository", "REPOSITORY", " Repository "):
        match = resolve_entity_type(value)
        assert match.canonical == "Repository", value
        assert match.candidates == ()
    assert resolve_entity_type("Repository").changed is False


def test_unknown_entity_type_yields_bounded_candidates_and_no_choice() -> None:
    match = resolve_entity_type("Repositry")
    assert match.canonical is None
    assert match.candidates[0] == "Repository"
    assert 0 < len(match.candidates) <= 6


def test_predicate_spacing_hyphen_and_case_resolve() -> None:
    for value in ("policy_applies_to", "policy applies to", "Policy-Applies-To"):
        assert resolve_predicate(value).canonical == "POLICY_APPLIES_TO", value


def test_unknown_predicate_suggests_its_close_match() -> None:
    match = resolve_predicate("POLICY_APPLIES")
    assert match.canonical is None
    assert "POLICY_APPLIES_TO" in match.candidates


def test_claim_subgraphs_cover_views_and_lowering_slices() -> None:
    names = local_claim_subgraphs()
    assert {"debugging", "decisions", "knowledge", "memory", "admin"} <= set(names)
    assert resolve_claim_subgraph("Debugging").canonical == "debugging"
    assert resolve_claim_subgraph("infra-topology").canonical == "infra_topology"


def test_extension_vocabulary_from_a_catalog_is_honoured() -> None:
    vocab = vocabulary_from_catalog(
        {
            "entity_types": [{"label": "LoadBalancer"}],
            "predicates": [{"name": "ROUTES_TO", "category": "topology"}],
            "views": [{"subgraph": "infra_topology"}],
        }
    )
    assert vocab["entity_types"] == ("LoadBalancer",)
    assert (
        resolve_entity_type("loadbalancer", known=vocab["entity_types"]).canonical
        == "LoadBalancer"
    )
    assert (
        resolve_predicate("routes to", known=vocab["predicates"]).canonical
        == "ROUTES_TO"
    )
    assert set(vocab["subgraphs"]) == {"infra_topology", "topology"}


def test_close_candidates_is_deterministic_and_never_empty_for_a_known_pool() -> None:
    pool = ("Repository", "Service", "Feature")
    assert close_candidates("zzz", pool) == ("Feature", "Repository", "Service")
    assert close_candidates("servce", pool)[0] == "Service"
