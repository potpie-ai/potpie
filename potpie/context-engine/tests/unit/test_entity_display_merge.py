"""Authored entity summaries survive bare re-references (Stage 1 invariant).

A semantic mutation that merely *references* an existing entity (key + type
only) must never downgrade the node's authored ``name`` / ``summary`` /
``description`` to key-derived placeholders. ``merge_entity_display_properties``
implements the rule for dict-backed stores; the in-memory backend test covers
the full apply path.
"""

from __future__ import annotations

import pytest

from domain.graph_entity_summary import merge_entity_display_properties

pytestmark = pytest.mark.unit


def test_authored_fields_win_over_existing() -> None:
    out = merge_entity_display_properties(
        {"summary": "new summary", "description": "new card"},
        existing={"summary": "old summary", "description": "old card"},
        entity_key="repo:acme/shop",
    )
    assert out["summary"] == "new summary"
    assert out["description"] == "new card"


def test_bare_reference_keeps_existing_authored_fields() -> None:
    out = merge_entity_display_properties(
        {},
        existing={
            "name": "shop",
            "summary": "Next.js storefront",
            "description": "Customer-facing storefront web app.",
        },
        entity_key="repo:acme/shop",
    )
    assert out["name"] == "shop"
    assert out["summary"] == "Next.js storefront"
    assert out["description"] == "Customer-facing storefront web app."


def test_new_node_falls_back_to_entity_key() -> None:
    out = merge_entity_display_properties(
        {}, existing={}, entity_key="repo:acme/shop"
    )
    assert out["name"] == "repo:acme/shop"
    assert out["summary"] == "repo:acme/shop"
    assert out["description"] == "repo:acme/shop"


def test_summary_derives_from_authored_description_not_key() -> None:
    out = merge_entity_display_properties(
        {"description": "Handles refunds and settlement."},
        existing={"summary": "old"},
        entity_key="service:payments",
    )
    assert out["summary"] == "Handles refunds and settlement."


def test_in_memory_backend_apply_preserves_authored_summary() -> None:
    from adapters.outbound.graph.backends.in_memory_backend import (
        InMemoryGraphBackend,
    )
    from domain.graph_mutations import EntityUpsert
    from domain.reconciliation import MutationBatch

    backend = InMemoryGraphBackend()
    pot = "pot-test"

    first = MutationBatch()
    first.entity_upserts = [
        EntityUpsert(
            "repo:acme/shop",
            ("Entity", "Repository"),
            {"summary": "Next.js storefront", "description": "Storefront app."},
        )
    ]
    backend.mutation.apply(first, expected_pot_id=pot)

    second = MutationBatch()
    second.entity_upserts = [
        EntityUpsert("repo:acme/shop", ("Entity", "Repository"), {})
    ]
    backend.mutation.apply(second, expected_pot_id=pot)

    props = backend.claim_query.entity_properties(
        pot_id=pot, entity_key="repo:acme/shop"
    )
    assert props["summary"] == "Next.js storefront"
    assert props["description"] == "Storefront app."
