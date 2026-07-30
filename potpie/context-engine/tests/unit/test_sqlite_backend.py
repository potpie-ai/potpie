"""Strict SQLite + sqlite-vec graph backend tests."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("sqlite_vec")

from potpie_context_core.context_events import EventRef
from potpie_context_core.errors import CapabilityNotImplemented
from potpie_context_core.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceContext,
)
from potpie_context_core.lifecycle import SetupPlan
from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_core.reconciliation import MutationBatch
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
)
from potpie_context_engine.adapters.outbound.graph.backends import (
    KNOWN_PROFILES,
    build_backend,
)
from potpie_context_engine.adapters.outbound.graph.backends.sqlite_backend import (
    SQLiteGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.sqlite.connection import (
    SQLiteConnectionFactory,
    SQLiteVecUnavailableError,
)
from potpie_context_engine.adapters.outbound.graph.sqlite.mutation import (
    SQLiteMutation,
)
from potpie_context_engine.adapters.outbound.graph.sqlite.schema import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    METADATA_CONTRACT,
)
from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
    HashingEmbedder,
)
from potpie_context_engine.bootstrap.host_wiring import default_backend_profile
from potpie_context_engine.testing.conformance import run_graph_backend_conformance
from potpie_context_core.lifecycle import default_setup_backend

pytestmark = pytest.mark.unit

POT = "sqlite-pot"


class _MiniLMFixture:
    """Stable test vectors that identify with the strict production contract."""

    name = EMBEDDING_MODEL
    dimensions = EMBEDDING_DIM

    def embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
            vector[int.from_bytes(digest, "big") % self.dimensions] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return tuple(value / norm for value in vector)

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]


def _backend(tmp_path: Path) -> SQLiteGraphBackend:
    return SQLiteGraphBackend(
        path=tmp_path / "graph.sqlite3",
        embedder=_MiniLMFixture(),
    )


def _plan(*, fact: str = "web depends on payments api") -> MutationBatch:
    observed = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
    return MutationBatch(
        event_ref=EventRef(event_id="evt-1", source_system="repo", pot_id=POT),
        summary=fact,
        entity_upserts=[
            EntityUpsert(
                entity_key="service:web",
                labels=("Service",),
                properties={"description": "customer web frontend"},
            ),
            EntityUpsert(entity_key="service:payments", labels=("Service",)),
            EntityUpsert(entity_key="team:checkout", labels=("Team",)),
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type="DEPENDS_ON",
                from_entity_key="service:web",
                to_entity_key="service:payments",
                properties={
                    "claim_key": "claim:dependency",
                    "subgraph": "infra_topology",
                    "source_system": "github",
                    "source_ref": "Repo:Manifest",
                    "source_refs": ["doc:architecture"],
                    "evidence": [{"ref": "ticket:123"}],
                    "fact": fact,
                    "description": "checkout calls the payments backend",
                    "valid_at": observed.isoformat(),
                },
            ),
            EdgeUpsert(
                edge_type="OWNED_BY",
                from_entity_key="service:payments",
                to_entity_key="team:checkout",
                properties={
                    "claim_key": "claim:owner",
                    "subgraph": "code_topology",
                    "source_system": "catalog",
                    "source_ref": "catalog:payments",
                    "fact": "checkout team owns payments",
                    "valid_at": (observed + timedelta(hours=1)).isoformat(),
                },
            ),
        ],
    )


def test_sqlite_profile_provisions_exact_contract_and_fails_snapshot(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)

    result = backend.provision(SetupPlan(backend="sqlite"))

    assert result.state == "done"
    assert backend.profile == "sqlite"
    assert backend.match_mode == "vector"
    assert set(backend.capabilities().implemented()) == {
        "mutation",
        "claim_query",
        "semantic",
        "inspection",
        "analytics",
    }
    assert "sqlite" in KNOWN_PROFILES
    with sqlite3.connect(backend.path) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM graph_metadata"))
    assert metadata == dict(METADATA_CONTRACT)
    assert backend.provision(
        SetupPlan(backend="sqlite", embeddings="local")
    ).state == "failed"
    with pytest.raises(CapabilityNotImplemented):
        backend.snapshot.export(pot_id=POT, destination=str(tmp_path / "snapshot"))


def test_sqlite_satisfies_public_graph_backend_conformance(tmp_path: Path) -> None:
    run_graph_backend_conformance(lambda: _backend(tmp_path))


def test_mutation_filters_semantic_and_inspection_survive_restart(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    context = ProvenanceContext(mutation_id="mutation-1")
    plan = _plan()

    result = backend.mutation.apply(
        plan, expected_pot_id=POT, provenance_context=context
    )
    replay = backend.mutation.apply(
        plan, expected_pot_id=POT, provenance_context=context
    )

    assert result.ok is True
    assert replay == result
    assert len(backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))) == 2
    assert [
        row.claim_key
        for row in backend.claim_query.find_claims(
            ClaimQueryFilter(pot_id=POT, predicate_in=("DEPENDS_ON",))
        )
    ] == ["claim:dependency"]
    assert [
        row.claim_key
        for row in backend.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=POT,
                subject_label="Service",
                object_label="Service",
                subgraph_in=("infra_topology",),
                source_system_in=("github",),
            )
        )
    ] == ["claim:dependency"]
    for source_ref in ("repo:manifest", "DOC:ARCHITECTURE", "ticket:123"):
        rows = backend.claim_query.find_claims(
            ClaimQueryFilter(pot_id=POT, source_ref_in=(source_ref,))
        )
        assert [row.claim_key for row in rows] == ["claim:dependency"]

    knn = backend.semantic.search(
        pot_id=POT,
        query="web depends on payments api",
        k=2,
        filter_=ClaimQueryFilter(pot_id=POT, predicate_in=("DEPENDS_ON",)),
    )
    exact = backend.semantic.search(
        pot_id=POT,
        query="web depends on payments api",
        k=2,
        filter_=ClaimQueryFilter(pot_id=POT, subject_label="Service"),
    )
    assert knn[0].claim_key == "claim:dependency"
    assert exact[0].properties["semantic_similarity"] >= 0.0
    assert backend.inspection.neighborhood(
        pot_id=POT, entity_key="service:web", depth=2
    ).edges
    assert backend.inspection.path(
        pot_id=POT, from_key="service:web", to_key="team:checkout"
    ).edges
    assert backend.analytics.counts(POT) == {
        "claims": 2,
        "entities": 3,
        "predicates": 2,
        "invalidated": 0,
    }

    fresh = _backend(tmp_path)
    assert fresh.mutation.lookup_execution(
        plan, expected_pot_id=POT, mutation_id="mutation-1"
    ).result == result
    assert fresh.semantic.search(
        pot_id=POT, query="payments dependency", k=2
    )
    assert fresh.mutation.readiness(POT).ready is True


def test_temporal_filters_invalidation_deletion_and_pot_isolation(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    backend.mutation.apply(_plan(), expected_pot_id=POT)
    cutoff = datetime(2026, 7, 29, 12, 30, tzinfo=timezone.utc)

    assert [
        row.claim_key
        for row in backend.claim_query.find_claims(
            ClaimQueryFilter(pot_id=POT, valid_at_before=cutoff)
        )
    ] == ["claim:dependency"]
    assert [
        row.claim_key
        for row in backend.claim_query.find_claims(
            ClaimQueryFilter(pot_id=POT, valid_at_after=cutoff)
        )
    ] == ["claim:owner"]
    assert backend.claim_query.find_claims(ClaimQueryFilter(pot_id="other")) == []

    invalidated = backend.mutation.apply(
        MutationBatch(
            invalidations=[
                InvalidationOp(
                    target_entity_key=None,
                    target_edge=(
                        "DEPENDS_ON",
                        "service:web",
                        "service:payments",
                    ),
                    reason="superseded",
                )
            ]
        ),
        expected_pot_id=POT,
    )
    assert invalidated.mutation_summary.invalidations_applied == 1
    assert [
        row.claim_key
        for row in backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))
    ] == ["claim:owner"]
    historical = backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=POT,
            include_invalidated=True,
            fact_query="payments backend",
            limit=10,
        )
    )
    assert {row.claim_key for row in historical} == {
        "claim:dependency",
        "claim:owner",
    }

    deleted = backend.mutation.apply(
        MutationBatch(
            edge_deletes=[
                EdgeDelete("OWNED_BY", "service:payments", "team:checkout")
            ]
        ),
        expected_pot_id=POT,
    )
    assert deleted.mutation_summary.edge_deletes_applied == 1
    assert len(
        backend.claim_query.find_claims(
            ClaimQueryFilter(pot_id=POT, include_invalidated=True)
        )
    ) == 1
    assert backend.mutation.reset_pot(POT) == {"removed_claims": 1}
    assert backend.analytics.counts(POT)["claims"] == 0


def test_complete_filter_contract_applies_before_limit_and_semantic_ranking(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    backend.mutation.apply(
        _plan(),
        expected_pot_id=POT,
        provenance_context=ProvenanceContext(mutation_id="axis-mutation"),
    )
    before = datetime(2026, 7, 29, 11, 0, tzinfo=timezone.utc)
    cutoff = datetime(2026, 7, 29, 12, 30, tzinfo=timezone.utc)
    axes = {
        "predicate_in": ("MISSING", "DEPENDS_ON"),
        "subject_key_in": ("missing", "service:web"),
        "object_key_in": ("missing", "service:payments"),
        "claim_key_in": ("missing", "claim:dependency"),
        "subgraph_in": ("missing", "infra_topology"),
        "mutation_id_in": ("missing", "axis-mutation"),
        "source_ref_in": ("missing", "DOC:ARCHITECTURE"),
        "source_system_in": ("missing", "github"),
        "subject_label": "Service",
        "object_label": "Service",
        "valid_at_after": before,
        "valid_at_before": cutoff,
        "as_of": cutoff,
        "limit": 1,
    }

    ordinary = backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=POT, **axes)
    )
    semantic = backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=POT,
            fact_query="payments dependency",
            **axes,
        )
    )

    assert [row.claim_key for row in ordinary] == ["claim:dependency"]
    assert [row.claim_key for row in semantic] == ["claim:dependency"]
    assert "semantic_similarity" in semantic[0].properties
    assert backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=POT,
            claim_key_in=("claim:owner",),
            limit=1,
        )
    )[0].claim_key == "claim:owner"
    assert backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=POT,
            as_of=before,
        )
    ) == []


def test_temporal_filters_compare_normalized_instants(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    plan = _plan()
    plan.edge_upserts[0].properties["valid_at"] = "2026-07-29T12:00:00+05:30"
    backend.mutation.apply(plan, expected_pot_id=POT)
    cutoff = datetime(2026, 7, 29, 7, 0, tzinfo=timezone.utc)

    before = backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=POT,
            claim_key_in=("claim:dependency",),
            valid_at_before=cutoff,
        )
    )
    after = backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=POT,
            claim_key_in=("claim:dependency",),
            valid_at_after=cutoff,
        )
    )

    assert before[0].valid_at == datetime(
        2026, 7, 29, 6, 30, tzinfo=timezone.utc
    )
    assert after == []


def test_vector_failure_rolls_back_claim_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    backend.provision(SetupPlan(backend="sqlite"))
    original = SQLiteMutation._upsert_claim

    def fail_after_vector_write(connection, prepared):
        original(connection, prepared)
        raise sqlite3.OperationalError("injected vector DML failure")

    monkeypatch.setattr(
        SQLiteMutation, "_upsert_claim", staticmethod(fail_after_vector_write)
    )
    with pytest.raises(sqlite3.OperationalError, match="injected vector"):
        backend.mutation.apply(
            _plan(),
            expected_pot_id=POT,
            provenance_context=ProvenanceContext(mutation_id="rollback-mutation"),
        )

    with backend.connections.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM claim_vectors").fetchone()[0]
            == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM mutation_receipts").fetchone()[0]
            == 0
        )


def test_strict_embedder_receipt_and_schema_mismatches_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="requires sentence-transformers"):
        SQLiteGraphBackend(
            path=tmp_path / "hashing.sqlite3", embedder=HashingEmbedder()
        )

    wrong_dimension = _MiniLMFixture()
    wrong_dimension.dimensions = 3
    with pytest.raises(RuntimeError, match="384-dimensional"):
        SQLiteGraphBackend(
            path=tmp_path / "wrong-dimension.sqlite3",
            embedder=wrong_dimension,
        )

    backend = _backend(tmp_path)
    plan = _plan()
    backend.mutation.apply(
        plan,
        expected_pot_id=POT,
        provenance_context=ProvenanceContext(mutation_id="reserved"),
    )
    with pytest.raises(MutationExecutionReuseError):
        backend.mutation.apply(
            _plan(fact="different content"),
            expected_pot_id=POT,
            provenance_context=ProvenanceContext(mutation_id="reserved"),
        )

    with sqlite3.connect(backend.path) as connection:
        connection.execute(
            "UPDATE graph_metadata SET value = '768' WHERE key = 'embedding_dim'"
        )
    assert backend.mutation.readiness(POT).ready is False
    with pytest.raises(RuntimeError, match="metadata contract mismatch"):
        backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))


def test_extension_load_failure_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite_vec

    original_load = sqlite_vec.load

    def broken_load(connection):
        del connection
        raise OSError("blocked extension")

    monkeypatch.setattr(sqlite_vec, "load", broken_load)
    with pytest.raises(SQLiteVecUnavailableError, match="blocked extension"):
        SQLiteConnectionFactory(tmp_path / "broken.sqlite3").connect()

    def wrong_version(connection):
        original_load(connection)
        connection.create_function("vec_version", 0, lambda: "v0.2.0")

    monkeypatch.setattr(sqlite_vec, "load", wrong_version)
    with pytest.raises(SQLiteVecUnavailableError, match="found v0.2.0"):
        SQLiteConnectionFactory(tmp_path / "wrong-version.sqlite3").connect()


def test_build_backend_accepts_injected_strict_embedder(tmp_path: Path) -> None:
    backend = build_backend("sqlite", embedder=_MiniLMFixture())
    assert isinstance(backend, SQLiteGraphBackend)
    assert backend.profile == "sqlite"


def test_windows_defaults_to_sqlite_and_honors_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.delenv("CONTEXT_ENGINE_BACKEND", raising=False)
    monkeypatch.delenv("GRAPH_DB_BACKEND", raising=False)

    assert default_backend_profile() == "sqlite"
    assert default_setup_backend() == "sqlite"

    (tmp_path / "config.json").write_text(
        '{"backend": "embedded"}',
        encoding="utf-8",
    )
    assert default_backend_profile() == "embedded"

    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "falkordb_lite")
    assert default_backend_profile() == "falkordb_lite"
