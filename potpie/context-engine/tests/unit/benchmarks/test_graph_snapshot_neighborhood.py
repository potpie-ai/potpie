"""Regression tests for the neighborhood-based graph snapshot.

The bench previously snapshotted via ``goal=retrieve / strategy=traversal``,
which the engine answers ``unsupported_context_graph_query`` — silently
producing empty snapshots so every post-ingest assertion graded against an
empty graph. The snapshot now parses ``goal=neighborhood`` (``result.nodes``
/ ``result.edges``) and recovers the semantic edge predicate from the
canonical ``RELATES_TO`` relationship.
"""

from __future__ import annotations

from context_engine.benchmarks.core.graph_inspect import (
    _edge_predicate,
    _parse_neighborhood,
    _pick_label,
)


def _neighborhood_response() -> dict:
    return {
        "kind": "neighborhood",
        "result": {
            "pot_id": "pot-1",
            "nodes": [
                {
                    "entity_key": "service:acme:checkout-api",
                    "labels": ["Entity", "Service"],
                    "properties": {"summary": "checkout"},
                },
                {
                    "entity_key": "service:acme:inventory-svc",
                    "labels": ["Entity", "Service"],
                    "properties": {},
                },
                {
                    "entity_key": "datastore:acme:checkout-prod-db",
                    "labels": ["Entity", "DataStore"],
                    "properties": {},
                },
            ],
            "edges": [
                {
                    "from": "service:acme:checkout-api",
                    "type": "RELATES_TO",
                    "to": "service:acme:inventory-svc",
                    "properties": {
                        "name": "DEPENDS_ON",
                        "fact": "service:acme:checkout-api DEPENDS_ON service:acme:inventory-svc",
                    },
                },
                {
                    "from": "service:acme:checkout-api",
                    "type": "RELATES_TO",
                    "to": "datastore:acme:checkout-prod-db",
                    # No ``name`` — predicate must be recovered from ``fact``.
                    "properties": {
                        "fact": "service:acme:checkout-api USES datastore:acme:checkout-prod-db",
                    },
                },
            ],
        },
    }


def test_pick_label_prefers_specific_over_base() -> None:
    assert _pick_label(["Entity", "Service"]) == "Service"
    assert _pick_label(["Entity"]) == "Entity"
    assert _pick_label("DataStore") == "DataStore"
    assert _pick_label([]) == ""


def test_edge_predicate_prefers_name_then_fact() -> None:
    assert _edge_predicate("a", "b", {"name": "OWNED_BY"}, "RELATES_TO") == "OWNED_BY"
    pred = _edge_predicate(
        "service:acme:checkout-api",
        "datastore:acme:checkout-prod-db",
        {"fact": "service:acme:checkout-api USES datastore:acme:checkout-prod-db"},
        "RELATES_TO",
    )
    assert pred == "USES"
    # Nothing to go on → fall back to the raw relationship type.
    assert _edge_predicate("a", "b", {}, "RELATES_TO") == "RELATES_TO"


def test_parse_neighborhood_maps_entities_and_typed_edges() -> None:
    snap = _parse_neighborhood(_neighborhood_response())

    assert len(snap.entities) == 3
    assert {e.label for e in snap.entities} == {"Service", "DataStore"}
    assert len(snap.entities_by_label("Service")) == 2

    # Canonical RELATES_TO edges surface as their semantic predicate, with
    # endpoint labels resolved from the node list.
    depends = snap.edges_by_type("DEPENDS_ON")
    uses = snap.edges_by_type("USES")
    assert len(depends) == 1
    assert len(uses) == 1
    assert depends[0].from_label == "Service" and depends[0].to_label == "Service"
    assert uses[0].from_label == "Service" and uses[0].to_label == "DataStore"


def test_parse_neighborhood_empty_result_is_empty_snapshot() -> None:
    assert _parse_neighborhood({"result": None}).entities == []
    assert _parse_neighborhood({}).edges == []
