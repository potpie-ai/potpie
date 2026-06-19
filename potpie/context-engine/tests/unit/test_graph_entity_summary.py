"""Entity summary derivation and repair tests."""

from __future__ import annotations

import pytest

from adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from domain.ports.claim_query import ClaimQueryFilter, ClaimRow

pytestmark = pytest.mark.unit


def test_in_memory_inspection_nodes_include_normalized_entity_summaries() -> None:
    backend = InMemoryGraphBackend()
    backend.store.add(
        ClaimRow(
            pot_id="p",
            predicate="DEPENDS_ON",
            subject_key="service:web",
            object_key="service:auth",
            fact="web depends on auth",
        )
    )
    backend.store.set_entity_properties(
        pot_id="p",
        entity_key="service:web",
        properties={"description": "Web frontend service."},
    )

    sl = backend.inspection.slice(
        pot_id="p",
        filter_=ClaimQueryFilter(pot_id="p"),
    )

    web = next(n for n in sl.nodes if n.key == "service:web")
    assert web.properties["summary"] == "Web frontend service."
    assert web.properties["description"] == "Web frontend service."


def test_in_memory_repair_backfills_existing_entity_summaries() -> None:
    changed = 0

    def on_change() -> None:
        nonlocal changed
        changed += 1

    backend = InMemoryGraphBackend(on_change=on_change)
    backend.store.add(
        ClaimRow(
            pot_id="p",
            predicate="DEPENDS_ON",
            subject_key="service:web",
            object_key="service:auth",
            fact="web depends on auth",
        )
    )
    backend.store.set_entity_properties(
        pot_id="p",
        entity_key="service:web",
        properties={"description": "Web frontend service."},
    )
    backend.store.set_entity_properties(
        pot_id="p",
        entity_key="service:auth",
        properties={"name": "auth", "summary": ""},
    )

    report = backend.analytics.repair(pot_id="p", targets=["entity_summaries"])

    assert report.repaired["entity_summaries"] == 2
    assert changed == 1
    web = backend.store.entity_properties(pot_id="p", entity_key="service:web")
    auth = backend.store.entity_properties(pot_id="p", entity_key="service:auth")
    assert web["summary"] == "Web frontend service."
    assert web["description"] == "Web frontend service."
    assert auth["summary"] == "auth"
    assert auth["description"] == "auth"
