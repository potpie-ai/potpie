"""Live FalkorDBLite round-trip: write → read → reset against embedded Redis.

Uses the real embedded FalkorDBLite (via ``redislite``) — no server, no Docker
— with one shared graph handle injected into both writer + reader, exactly how
``build_container`` wires the Lite backend. Skips cleanly if FalkorDBLite isn't
installed.

This is the Phase-5 parity check: the same canonical Position-B operations the
Neo4j adapters expose, exercised end to end on FalkorDBLite through the real
adapters (no fakes).
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile

import pytest

pytestmark = pytest.mark.integration

# Import name is ``redislite`` (distribution: falkordblite); skip if absent.
falkordb_client = pytest.importorskip("redislite.falkordb_client")

from context_engine.adapters.outbound.graph.falkordb_reader import FalkorDBClaimQueryStore  # noqa: E402
from context_engine.adapters.outbound.graph.falkordb_writer import FalkorDBGraphWriter  # noqa: E402
from context_engine.domain.graph_mutations import (  # noqa: E402
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)
from context_engine.domain.ports.claim_query import ClaimQueryFilter  # noqa: E402


class _Settings:
    """Minimal enabled settings; graph is injected so paths aren't consulted."""

    def is_enabled(self) -> bool:
        return True

    def falkordb_graph_name(self) -> str:
        return "context_graph"


class _FakeEmbedder:
    name = "fake-integration-embedder"
    dimensions = 3

    def embed(self, text: str) -> tuple[float, ...]:
        t = text.lower()
        if "auth" in t or "login" in t:
            return (1.0, 0.0, 0.0)
        return (0.0, 1.0, 0.0)

    def embed_many(self, texts):
        return [self.embed(t) for t in texts]


@pytest.fixture()
def shared_graph():
    tmp = tempfile.mkdtemp(prefix="falkordblite_test_")
    db = falkordb_client.FalkorDB(f"{tmp}/context_graph.db")
    try:
        yield db.select_graph("context_graph")
    finally:
        db.close()  # stop the embedded redis-server
        shutil.rmtree(tmp, ignore_errors=True)


def test_write_read_reset_roundtrip(shared_graph) -> None:
    settings = _Settings()
    # One shared embedded handle into both adapters (as build_container does).
    writer = FalkorDBGraphWriter(settings, graph=shared_graph)
    reader = FalkorDBClaimQueryStore(settings, graph=shared_graph)
    pot = "potA"
    prov = ProvenanceRef(pot_id=pot, source_event_id="e1", source_system="agent")

    assert writer.enabled is True

    async def _seed() -> dict:
        assert await writer.ensure_indexes() is True
        n_ent = await writer.upsert_entities(
            pot,
            [
                EntityUpsert("service:web", ("Entity", "Service"), {"name": "web"}),
                EntityUpsert("service:auth", ("Entity", "Service"), {"name": "auth"}),
            ],
            prov,
        )
        n_edge = await writer.upsert_edges(
            pot,
            [
                EdgeUpsert(
                    "DEPENDS_ON",
                    "service:web",
                    "service:auth",
                    {"fact": "web depends on auth"},
                )
            ],
            prov,
        )
        # Duplicate identical edge upsert must not create a second live claim.
        await writer.upsert_edges(
            pot,
            [
                EdgeUpsert(
                    "DEPENDS_ON",
                    "service:web",
                    "service:auth",
                    {"fact": "web depends on auth"},
                )
            ],
            prov,
        )
        return {"n_ent": n_ent, "n_edge": n_edge}

    out = asyncio.run(_seed())
    assert out["n_ent"] == 2
    assert out["n_edge"] == 1

    rows = reader.find_claims(
        ClaimQueryFilter(pot_id=pot, predicate_in=("DEPENDS_ON",))
    )
    assert len(rows) == 1
    assert rows[0].predicate == "DEPENDS_ON"
    assert rows[0].subject_key == "service:web"
    assert rows[0].object_key == "service:auth"
    assert rows[0].fact == "web depends on auth"

    labels = reader.entity_labels(
        pot_id=pot, entity_keys=["service:web", "service:auth"]
    )
    assert "Service" in labels["service:web"]

    # Invalidate the edge → no longer in the live set.
    n_inv = asyncio.run(
        writer.invalidate(
            pot,
            [
                InvalidationOp(
                    target_entity_key=None,
                    target_edge=("DEPENDS_ON", "service:web", "service:auth"),
                    reason="test",
                )
            ],
            prov,
        )
    )
    assert n_inv == 1
    assert (
        reader.find_claims(ClaimQueryFilter(pot_id=pot, predicate_in=("DEPENDS_ON",)))
        == []
    )

    # reset_pot clears the partition with the Neo4j-parity result contract.
    final = asyncio.run(writer.reset_pot(pot))
    assert final["ok"] is True
    assert final["group_id_nodes_remaining"] == 0


def test_vector_search_orders_by_cosine_distance(shared_graph) -> None:
    settings = _Settings()
    embedder = _FakeEmbedder()
    writer = FalkorDBGraphWriter(settings, graph=shared_graph, embedder=embedder)
    reader = FalkorDBClaimQueryStore(settings, graph=shared_graph, embedder=embedder)
    pot = "potV"
    prov = ProvenanceRef(pot_id=pot, source_event_id="e1", source_system="agent")

    async def _seed() -> None:
        assert await writer.ensure_indexes() is True
        await writer.upsert_entities(
            pot,
            [
                EntityUpsert("service:web", ("Entity", "Service"), {}),
                EntityUpsert("service:auth", ("Entity", "Service"), {}),
                EntityUpsert("service:db", ("Entity", "Service"), {}),
            ],
            prov,
        )
        await writer.upsert_edges(
            pot,
            [
                EdgeUpsert(
                    "DEPENDS_ON",
                    "service:web",
                    "service:auth",
                    {"fact": "web depends on auth login"},
                ),
                EdgeUpsert(
                    "DEPENDS_ON",
                    "service:web",
                    "service:db",
                    {"fact": "web stores data in database"},
                ),
            ],
            prov,
        )

    asyncio.run(_seed())

    rows = reader.find_claims(
        ClaimQueryFilter(pot_id=pot, fact_query="login auth", limit=2)
    )

    assert [r.object_key for r in rows] == ["service:auth", "service:db"]
    assert rows[0].properties["semantic_similarity"] == pytest.approx(1.0)
    assert rows[1].properties["semantic_similarity"] == pytest.approx(0.0)
