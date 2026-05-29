"""Live FalkorDB round-trip: write → read → reset against the real client.

Skips cleanly unless a FalkorDB server is reachable. Point it at one with
``FALKORDB_TEST_URL`` (e.g. ``redis://localhost:6399`` for the local
``falkordb/falkordb`` container); otherwise the module is skipped so CI
without FalkorDB stays green.

This is the Phase-5 parity check: the same canonical Position-B operations the
Neo4j adapters expose, exercised end to end on FalkorDB through the real
adapters (no fakes).
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytestmark = pytest.mark.integration

falkordb = pytest.importorskip("falkordb")

from adapters.outbound.graph.falkordb_reader import FalkorDBClaimQueryStore  # noqa: E402
from adapters.outbound.graph.falkordb_writer import FalkorDBGraphWriter  # noqa: E402
from domain.graph_mutations import EdgeUpsert, EntityUpsert, ProvenanceRef  # noqa: E402
from domain.ports.claim_query import ClaimQueryFilter  # noqa: E402

_URL = os.getenv("FALKORDB_TEST_URL")


def _reachable(url: str) -> bool:
    try:
        falkordb.FalkorDB.from_url(url).select_graph("__ping__").query("RETURN 1")
        return True
    except Exception:  # noqa: BLE001
        return False


if not _URL or not _reachable(_URL):
    pytest.skip(
        "FALKORDB_TEST_URL not set or FalkorDB unreachable", allow_module_level=True
    )


class _Settings:
    """Minimal settings: enabled + a per-run isolated graph keyspace."""

    def __init__(self, graph_name: str) -> None:
        self._graph_name = graph_name

    def is_enabled(self) -> bool:
        return True

    def falkordb_url(self) -> str:
        return _URL

    def falkordb_graph_name(self) -> str:
        return self._graph_name


@pytest.fixture()
def settings():
    name = f"ce_test_{uuid.uuid4().hex[:8]}"
    s = _Settings(name)
    yield s
    # Best-effort cleanup of the throwaway keyspace.
    try:
        falkordb.FalkorDB.from_url(_URL).select_graph(name).delete()
    except Exception:  # noqa: BLE001
        pass


def test_write_read_reset_roundtrip(settings) -> None:
    writer = FalkorDBGraphWriter(settings)
    reader = FalkorDBClaimQueryStore(settings)
    pot = "potA"
    prov = ProvenanceRef(pot_id=pot, source_event_id="e1", source_system="agent")

    async def _run() -> dict:
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
            [EdgeUpsert("DEPENDS_ON", "service:web", "service:auth", {"fact": "web depends on auth"})],
            prov,
        )
        # Idempotency: a second identical edge upsert must not duplicate.
        await writer.upsert_edges(
            pot,
            [EdgeUpsert("DEPENDS_ON", "service:web", "service:auth", {"fact": "web depends on auth"})],
            prov,
        )
        reset = await writer.reset_pot(pot)
        return {"n_ent": n_ent, "n_edge": n_edge, "reset": reset}

    out = asyncio.run(_run())
    assert out["n_ent"] == 2
    assert out["n_edge"] == 1

    # Read back BEFORE reset would have wiped it — so re-seed + read synchronously.
    async def _seed_and_read():
        await writer.upsert_entities(
            pot,
            [
                EntityUpsert("service:web", ("Entity", "Service"), {"name": "web"}),
                EntityUpsert("service:auth", ("Entity", "Service"), {"name": "auth"}),
            ],
            prov,
        )
        await writer.upsert_edges(
            pot,
            [EdgeUpsert("DEPENDS_ON", "service:web", "service:auth", {"fact": "web depends on auth"})],
            prov,
        )

    asyncio.run(_seed_and_read())

    rows = reader.find_claims(ClaimQueryFilter(pot_id=pot, predicate_in=("DEPENDS_ON",)))
    assert len(rows) == 1
    assert rows[0].predicate == "DEPENDS_ON"
    assert rows[0].subject_key == "service:web"
    assert rows[0].object_key == "service:auth"
    assert rows[0].fact == "web depends on auth"

    labels = reader.entity_labels(pot_id=pot, entity_keys=["service:web", "service:auth"])
    assert "Service" in labels["service:web"]

    # Idempotency holds on the second seed too (still exactly one live claim).
    rows2 = reader.find_claims(ClaimQueryFilter(pot_id=pot, predicate_in=("DEPENDS_ON",)))
    assert len(rows2) == 1

    # Final reset clears the partition.
    final = asyncio.run(writer.reset_pot(pot))
    assert final["ok"] is True
    assert final["group_id_nodes_remaining"] == 0
    assert reader.find_claims(ClaimQueryFilter(pot_id=pot, predicate_in=("DEPENDS_ON",))) == []
