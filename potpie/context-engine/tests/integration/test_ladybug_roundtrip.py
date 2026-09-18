"""Live Ladybug round-trip: write → read → vector → BFS → reset.

Skips when the ``ladybug`` package is not installed. Uses a temp ``*.lbdb``
and a deterministic fake embedder (no network).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ladybug = pytest.importorskip("ladybug")

from potpie_context_engine.adapters.outbound.graph.backends.ladybug_backend import (  # noqa: E402
    LadybugGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.ladybug_reader import (  # noqa: E402
    LadybugClaimQueryStore,
)
from potpie_context_engine.adapters.outbound.graph.ladybug_writer import (  # noqa: E402
    LadybugGraphProvider,
    LadybugGraphWriter,
)
from potpie_context_engine.core.graph_mutations import (  # noqa: E402
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter  # noqa: E402
from potpie_context_engine.domain.retrieval_card import cosine_similarity  # noqa: E402


class _Settings:
    def __init__(self, path: str) -> None:
        self._path = path

    def is_enabled(self) -> bool:
        return True

    def ladybug_path(self) -> str:
        return self._path


class _FakeEmbedder:
    name = "fake-ladybug-embedder"
    dimensions = 8

    def embed(self, text: str) -> tuple[float, ...]:
        t = (text or "").lower()
        if "auth" in t or "login" in t:
            return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if "pizza" in t:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        return (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def embed_many(self, texts):
        return [self.embed(t) for t in texts]


@pytest.fixture()
def ladybug_home(tmp_path):
    path = str(tmp_path / "ladybug.lbdb")
    yield path
    # Ladybug may leave sidecar files; wipe the parent temp dir via pytest.


def test_write_read_vector_bfs_reset(ladybug_home) -> None:
    settings = _Settings(ladybug_home)
    embedder = _FakeEmbedder()
    provider = LadybugGraphProvider(settings)
    writer = LadybugGraphWriter(settings, conn_provider=provider, embedder=embedder)
    reader = LadybugClaimQueryStore(settings, conn_provider=provider, embedder=embedder)
    backend = LadybugGraphBackend(
        settings,
        writer=writer,
        graph_provider=provider,
        embedder=embedder,
    )
    pot = "potA"
    prov = ProvenanceRef(pot_id=pot, source_event_id="e1", source_system="agent")

    assert backend.provision().ok is True

    async def _seed() -> None:
        await writer.upsert_entities(
            pot,
            [
                EntityUpsert("service:web", ("Entity", "Service"), {"name": "web"}),
                EntityUpsert("service:auth", ("Entity", "Service"), {"name": "auth"}),
                EntityUpsert("service:db", ("Entity", "Service"), {"name": "db"}),
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
                    "service:auth",
                    "service:db",
                    {"fact": "auth depends on db"},
                ),
                EdgeUpsert(
                    "MENTIONS",
                    "service:web",
                    "service:db",
                    {"fact": "totally unrelated pizza recipe"},
                ),
            ],
            prov,
        )

    asyncio.run(_seed())

    rows = reader.find_claims(
        ClaimQueryFilter(pot_id=pot, predicate_in=("DEPENDS_ON",))
    )
    assert len(rows) == 2
    facts = {r.fact for r in rows}
    assert "web depends on auth login" in facts

    labels = reader.entity_labels(
        pot_id=pot, entity_keys=["service:web", "service:auth"]
    )
    assert "Service" in labels["service:web"]

    # Vector ordering: query "auth login" should rank auth claim above pizza.
    scored = reader.find_claims(
        ClaimQueryFilter(pot_id=pot, fact_query="auth login", limit=5)
    )
    assert scored
    assert scored[0].fact and "auth" in scored[0].fact.lower()
    assert scored[0].properties.get("semantic_similarity") is not None
    # Recall@5 vs cosine ground truth on stored embeddings.
    q = list(embedder.embed("auth login"))
    lexical = reader.find_claims(ClaimQueryFilter(pot_id=pot, include_invalidated=True))
    ranked = sorted(
        (
            cosine_similarity(q, list(r.fact_embedding or [])),
            r.fact,
        )
        for r in lexical
        if r.fact_embedding
    )
    ranked.reverse()
    top5 = {fact for _, fact in ranked[:5]}
    assert scored[0].fact in top5

    # Multi-hop BFS neighborhood includes 2-hop entity.
    neigh = backend.inspection.neighborhood(
        pot_id=pot, entity_key="service:web", depth=2
    )
    keys = {n.key for n in neigh.nodes}
    assert "service:db" in keys

    path = backend.inspection.path(
        pot_id=pot, from_key="service:web", to_key="service:db", max_depth=3
    )
    assert [n.key for n in path.nodes][0] == "service:web"
    assert [n.key for n in path.nodes][-1] == "service:db"

    # Semantic port wrapper.
    hits = backend.semantic.search(pot_id=pot, query="auth login", k=3)
    assert hits

    # Invalidate → hidden from live query.
    asyncio.run(
        writer.invalidate(
            pot,
            [
                InvalidationOp(
                    target_entity_key=None,
                    target_edge=(
                        "DEPENDS_ON",
                        "service:web",
                        "service:auth",
                    ),
                    reason="test",
                )
            ],
            prov,
        )
    )
    live = reader.find_claims(
        ClaimQueryFilter(pot_id=pot, predicate_in=("DEPENDS_ON",))
    )
    assert all(r.subject_key != "service:web" for r in live)

    # Claim-key invalidation must not interpret an entity key as a claim key.
    remaining_key = live[0].claim_key
    remaining_entity_key = live[0].subject_key
    assert (
        backend.mutation.invalidate(
            pot_id=pot, claim_keys=[remaining_entity_key], reason="test"
        )
        == 0
    )
    assert reader.find_claims(
        ClaimQueryFilter(pot_id=pot, claim_key_in=(remaining_key,))
    )

    # Reopen file with a fresh connection (data must survive).
    provider.close()
    provider2 = LadybugGraphProvider(settings)
    reader2 = LadybugClaimQueryStore(
        settings,
        conn_provider=provider2,
        embedder=embedder,
    )
    writer2 = LadybugGraphWriter(settings, conn_provider=provider2, embedder=embedder)
    assert len(reader2.find_claims(ClaimQueryFilter(pot_id=pot))) >= 1

    final = asyncio.run(writer2.reset_pot(pot))
    assert final["ok"] is True
    assert final["group_id_nodes_remaining"] == 0
    assert reader2.find_claims(ClaimQueryFilter(pot_id=pot)) == []


def test_windows_validation_note_documented() -> None:
    """Assert research note exists for Win64 wheel validation."""
    # tests/integration -> context-engine -> potpie -> potpie repo root
    repo_root = Path(__file__).resolve().parents[4]
    note = repo_root / "docs" / "research" / "ladybug-windows.md"
    assert note.is_file(), f"missing {note}"
