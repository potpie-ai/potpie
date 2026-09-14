"""Ladybug GraphBackend wiring tests (no live DB required)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from potpie_context_engine.adapters.outbound.graph.backends import (
    KNOWN_PROFILES,
    build_backend,
)
from potpie_context_engine.adapters.outbound.graph.backends.ladybug_backend import (
    LadybugGraphBackend,
)
from potpie_context_engine.bootstrap.ingestion_server import build_ingestion_server
from potpie_context_engine.core.context_events import EventRef
from potpie_context_engine.core.graph_mutations import EdgeUpsert, EntityUpsert
from potpie_context_engine.core.ports.graph.backend import GraphBackend
from potpie_context_engine.core.reconciliation import ReconciliationPlan

pytestmark = pytest.mark.unit


class _Settings:
    def is_enabled(self) -> bool:
        return True

    def graph_db_backend(self) -> str:
        return "ladybug"

    def ladybug_path(self) -> str:
        return ".potpie/test/ladybug.lbdb"


class _Writer:
    enabled = True

    def __init__(self) -> None:
        self.entities = []
        self.edges = []

    async def ensure_indexes(self) -> bool:
        return True

    async def upsert_entities(self, pot_id, items, provenance) -> int:
        self.entities.extend(items)
        return len(items)

    async def upsert_edges(self, pot_id, items, provenance) -> int:
        self.edges.extend(items)
        return len(items)

    async def delete_edges(self, pot_id, items, provenance) -> int:
        return len(items)

    async def invalidate(self, pot_id, items, provenance) -> int:
        return len(items)

    async def invalidate_claim_keys(self, pot_id, claim_keys, *, reason=None) -> int:
        return len(tuple(claim_keys or ()))

    async def reset_pot(self, pot_id: str) -> dict:
        return {
            "ok": True,
            "group_id_nodes_before": 0,
            "group_id_nodes_remaining": 0,
        }


def test_build_backend_registers_ladybug_without_connecting() -> None:
    assert "ladybug" in KNOWN_PROFILES
    backend = build_backend("ladybug", settings=_Settings())
    assert isinstance(backend, GraphBackend)
    assert backend.profile == "ladybug"
    assert backend.graph_writer.enabled is True
    assert backend.capabilities().implemented() == (
        "mutation",
        "claim_query",
        "semantic",
        "inspection",
        "analytics",
    )


async def test_ladybug_backend_apply_uses_writer() -> None:
    writer = _Writer()
    backend = LadybugGraphBackend(_Settings(), writer=writer)
    plan = ReconciliationPlan(
        event_ref=EventRef(event_id="e1", source_system="agent", pot_id="p1"),
        entity_upserts=[
            EntityUpsert(entity_key="service:web", labels=("Entity", "Service"))
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type="DEPENDS_ON",
                from_entity_key="service:web",
                to_entity_key="service:auth",
                properties={"fact": "web depends on auth"},
            )
        ],
    )

    result = await backend.mutation.apply_async(plan, expected_pot_id="p1")

    assert result.ok
    assert len(writer.entities) == 1
    assert len(writer.edges) == 1


def test_ladybug_mutation_invalidate_uses_claim_keys() -> None:
    writer = _Writer()
    backend = LadybugGraphBackend(_Settings(), writer=writer)
    n = backend.mutation.invalidate(
        pot_id="p1", claim_keys=["claim:a", "claim:b"], reason="test"
    )
    assert n == 2


def test_ingestion_server_accepts_ladybug_backend() -> None:
    container = build_ingestion_server(
        settings=_Settings(),
        pots=MagicMock(),
    )
    assert container.backend is not None
    assert container.backend.profile == "ladybug"
