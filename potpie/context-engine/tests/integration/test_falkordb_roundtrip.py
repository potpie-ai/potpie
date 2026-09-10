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

from potpie_context_engine.adapters.outbound.graph.backends.falkordb_backend import (  # noqa: E402
    FalkorDBGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.falkordb_reader import (  # noqa: E402
    FalkorDBClaimQueryStore,
)
from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (  # noqa: E402
    FalkorDBGraphWriter,
)
from potpie_context_core.workbench_service import GraphWorkbenchService  # noqa: E402
from potpie_context_core.graph_mutations import (  # noqa: E402
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)
from potpie_context_core.ports.claim_query import ClaimQueryFilter  # noqa: E402


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


class _UnusedPlanStore:
    def save(self, _record) -> None:
        raise AssertionError("plan store should not be used")

    def get(self, *, pot_id: str, plan_id: str):
        raise AssertionError("plan store should not be used")

    def list(self, **_kwargs):
        raise AssertionError("plan store should not be used")


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


def test_entity_upsert_rejects_canonical_labels_conflicting_with_key(
    shared_graph,
) -> None:
    settings = _Settings()
    writer = FalkorDBGraphWriter(settings, graph=shared_graph)
    reader = FalkorDBClaimQueryStore(settings, graph=shared_graph)
    pot = "potRetype"
    prov = ProvenanceRef(pot_id=pot, source_event_id="e1", source_system="agent")
    key = "environment:production"

    async def _write() -> None:
        await writer.upsert_entities(
            pot,
            [
                EntityUpsert(
                    key,
                    (
                        "Entity",
                        "Activity",
                        "ConfigVariable",
                        "DeploymentTarget",
                        "Environment",
                    ),
                    {"name": "production"},
                )
            ],
            prov,
        )

    asyncio.run(_write())

    assert set(reader.entity_labels(pot_id=pot, entity_keys=[key])[key]) == {
        "Entity",
        "Environment",
    }


def test_entity_label_repair_cleans_existing_pollution(shared_graph) -> None:
    settings = _Settings()
    backend = FalkorDBGraphBackend(
        settings,
        graph_provider=lambda: shared_graph,
    )
    workbench = GraphWorkbenchService(
        backend=backend,
        plan_store=_UnusedPlanStore(),
    )
    reader = FalkorDBClaimQueryStore(settings, graph=shared_graph)
    pot = "potRepairLabels"
    key = "environment:production"
    shared_graph.query(
        "MERGE (e:Entity {group_id: $gid, entity_key: $key}) "
        "SET e:Activity:ConfigVariable:DeploymentTarget:Environment",
        params={"gid": pot, "key": key},
    )

    before = workbench.quality(pot_id=pot, report="entity-label-drift")
    report = backend.analytics.repair(pot, targets=["entity_labels"])
    after = workbench.quality(pot_id=pot, report="entity-label-drift")

    assert before.status == "degraded"
    assert before.findings[0].entity_keys == (key,)
    assert report.repaired == {"entity_labels": 1}
    assert after.status == "ok"
    assert after.findings == ()
    assert set(reader.entity_labels(pot_id=pot, entity_keys=[key])[key]) == {
        "Entity",
        "Environment",
    }


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


@pytest.mark.parametrize("fact_query", [None, "login auth"])
def test_source_ref_filters_accept_legacy_scalars(shared_graph, fact_query) -> None:
    """Exercise real Cypher with mixed stored types and exact OR matching."""
    settings = _Settings()
    writer = FalkorDBGraphWriter(settings, graph=shared_graph, embedder=_FakeEmbedder())
    assert asyncio.run(writer.ensure_indexes()) is True
    for key, refs, primary, gid in [
        ("scalar", "ref:a", "ref:primary", "p1"),
        ("array", ["ref:b", "ref:c"], "ref:other", "p1"),
        ("missing", None, "ref:d", "p1"),
        ("empty", [], "ref:empty", "p1"),
        ("substring", "ref:ab", "ref:none", "p1"),
        ("other-pot", "ref:a", "ref:primary", "p2"),
    ]:
        shared_graph.query(
            "CREATE (a:Entity {group_id: $gid, entity_key: $key})"
            "-[r:RELATES_TO {group_id: $gid, name: 'DEPENDS_ON',"
            " subject_key: $key, object_key: 'service:auth', claim_key: $key,"
            " fact: 'login auth', source_ref: $primary, source_refs: $refs}]->"
            "(b:Entity {group_id: $gid, entity_key: 'service:auth'}) "
            "SET r.fact_embedding = vecf32([1.0, 0.0, 0.0])",
            params={"gid": gid, "key": key, "refs": refs, "primary": primary},
        )
    reader = FalkorDBClaimQueryStore(
        settings, graph=shared_graph, embedder=_FakeEmbedder()
    )
    if fact_query:
        # Do not let the adapter's lexical fallback mask a broken vector filter.
        def no_fallback(_params):
            raise AssertionError("vector query unexpectedly fell back to lexical")

        reader._find_claims_lexical = no_fallback
    for refs, expected in [
        (("ref:a",), {"scalar"}),
        (("ref:c",), {"array"}),
        (("ref:a", "ref:c", "ref:d"), {"scalar", "array", "missing"}),
        (("ref:primary",), {"scalar"}),
        ((), {"scalar", "array", "missing", "empty", "substring"}),
    ]:
        rows = reader.find_claims(
            ClaimQueryFilter(
                pot_id="p1", source_ref_in=refs, fact_query=fact_query, limit=20
            )
        )
        assert {row.claim_key for row in rows} == expected
        if "scalar" in expected:
            assert next(
                row for row in rows if row.claim_key == "scalar"
            ).source_refs == ("ref:a",)
    if not fact_query:
        assert (
            reader.find_claims(
                ClaimQueryFilter(pot_id="p1", source_ref_in=("ref:unknown",))
            )
            == []
        )


@pytest.mark.parametrize("refs", ["ref:a", ["ref:a", "ref:b"]])
def test_writer_persists_source_refs_as_native_arrays(shared_graph, refs) -> None:
    writer = FalkorDBGraphWriter(_Settings(), graph=shared_graph)
    prov = ProvenanceRef(pot_id="p1", source_event_id="e1", source_system="agent")

    async def seed():
        await writer.upsert_entities(
            "p1",
            [
                EntityUpsert("service:web", ("Entity", "Service"), {}),
                EntityUpsert("service:auth", ("Entity", "Service"), {}),
            ],
            prov,
        )
        await writer.upsert_edges(
            "p1",
            [
                EdgeUpsert(
                    "DEPENDS_ON", "service:web", "service:auth", {"source_refs": refs}
                )
            ],
            prov,
        )

    asyncio.run(seed())
    result = shared_graph.query("MATCH ()-[r:RELATES_TO]->() RETURN r.source_refs")
    assert result.result_set == [[[refs] if isinstance(refs, str) else refs]]
    reader = FalkorDBClaimQueryStore(_Settings(), graph=shared_graph)
    assert (
        len(reader.find_claims(ClaimQueryFilter(pot_id="p1", source_ref_in=("ref:a",))))
        == 1
    )


@pytest.mark.parametrize("same_source", [True, False])
@pytest.mark.parametrize("separate_batches", [True, False])
def test_environment_claims_survive_normalization_storage_and_reassertion(
    shared_graph, same_source, separate_batches
):
    from potpie_context_core.entity_canonicalization import (
        canonicalize_reconciliation_plan,
    )
    from potpie_context_core.reconciliation import MutationBatch

    writer = FalkorDBGraphWriter(
        _Settings(), graph=shared_graph, embedder=_FakeEmbedder()
    )
    reader = FalkorDBClaimQueryStore(_Settings(), graph=shared_graph)
    prov = ProvenanceRef(pot_id="env-test", source_event_id="seed")
    edges = [
        EdgeUpsert(
            "CONFIGURES",
            "service:api",
            "config:mode",
            {
                "claim_key": f"claim:{env}",
                "environment": env,
                "source_ref": "fixture:config" if same_source else f"fixture:{env}",
                "fact": fact,
            },
        )
        for env, fact in [("prod", "auth production"), ("staging", "sandbox only")]
    ]

    async def write():
        await writer.upsert_entities(
            "env-test",
            [
                EntityUpsert("service:api", ("Entity", "Service")),
                EntityUpsert("config:mode", ("Entity", "ConfigVariable")),
            ],
            prov,
        )
        batches = [[edge] for edge in edges] if separate_batches else [edges]
        for group in batches:
            batch = MutationBatch(edge_upserts=group)
            canonicalize_reconciliation_plan(batch)
            await writer.upsert_edges("env-test", batch.edge_upserts, prov)
        await writer.upsert_edges("env-test", edges, prov)

    asyncio.run(write())
    claims = reader.find_claims(ClaimQueryFilter(pot_id="env-test"))
    assert len(claims) == 2
    assert {row.claim_key: row.fact for row in claims} == {
        "claim:prod": "auth production",
        "claim:staging": "sandbox only",
    }
    vectors = shared_graph.query(
        "MATCH ()-[r:RELATES_TO]->() RETURN r.claim_key, "
        "vec.cosineDistance(r.fact_embedding, vecf32([1, 0, 0]))"
    ).result_set
    assert dict(vectors) == {"claim:prod": 0.0, "claim:staging": 1.0}

    from potpie_context_engine.adapters.outbound.graph.falkordb_inspection import (
        FalkorDBInspection,
    )

    inspection = FalkorDBInspection(settings=_Settings(), graph=shared_graph)
    neighborhood = inspection.neighborhood(
        pot_id="env-test", entity_key="service:api", depth=2
    )
    assert len(neighborhood.edges) == 2
    assert {edge.properties["claim_key"] for edge in neighborhood.edges} == {
        "claim:prod",
        "claim:staging",
    }
