"""Real MiniLM retrieval-quality gate for the strict SQLite profile."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

if importlib.util.find_spec("sqlite_vec") is None:
    pytest.skip("sqlite-vec is not installed on this platform", allow_module_level=True)
if importlib.util.find_spec("sentence_transformers") is None:
    pytest.skip(
        "sentence-transformers is not installed", allow_module_level=True
    )

from potpie_context_core.graph_mutations import EdgeUpsert, EntityUpsert
from potpie_context_core.reconciliation import MutationBatch
from potpie_context_engine.adapters.outbound.graph.backends.sqlite_backend import (
    SQLiteGraphBackend,
)
from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    SentenceTransformerEmbedder,
    default_sentence_transformer_cache,
)

pytestmark = pytest.mark.integration

POT = "sqlite-minilm-quality"

_CLAIMS = (
    (
        "claim:database-pool",
        "service:orders",
        "resource:postgres",
        "Postgres database connections are exhausted because the connection pool "
        "is too small, causing timeout errors under traffic.",
    ),
    (
        "claim:auth-keys",
        "service:identity",
        "resource:signing-keys",
        "Login tokens are rejected after signing key rotation because API "
        "instances still cache the old authentication key.",
    ),
    (
        "claim:payment-idempotency",
        "service:checkout",
        "service:payments",
        "Checkout retries can create duplicate card charges when the payment "
        "idempotency key is not reused.",
    ),
    (
        "claim:cdn-cache",
        "service:web",
        "resource:cdn",
        "Customers see stale website content after a deployment until the CDN "
        "cache is purged.",
    ),
    (
        "claim:queue-workers",
        "service:jobs",
        "resource:queue",
        "Background jobs remain stuck in the queue when worker processes stop "
        "consuming messages.",
    ),
    (
        "claim:memory",
        "service:reports",
        "resource:memory",
        "Large report exports consume excessive memory and the process is killed "
        "by the operating system.",
    ),
    (
        "claim:dns",
        "service:gateway",
        "resource:dns",
        "Requests fail during DNS resolution because the internal service record "
        "has expired.",
    ),
    (
        "claim:certificate",
        "service:edge",
        "resource:certificate",
        "TLS handshakes fail after the public certificate expires.",
    ),
    (
        "claim:disk",
        "service:audit",
        "resource:disk",
        "Audit writes stop when log files fill all available disk space.",
    ),
    (
        "claim:rate-limit",
        "service:client",
        "service:vendor",
        "The vendor API returns rate limit responses when request concurrency "
        "exceeds the contracted quota.",
    ),
)

_QUERIES = (
    ("database pool saturation and connection timeouts", "claim:database-pool"),
    ("users cannot authenticate after rotating token keys", "claim:auth-keys"),
    ("customers charged twice when checkout retries", "claim:payment-idempotency"),
    ("old frontend assets remain visible after release", "claim:cdn-cache"),
    ("async tasks are not being consumed by workers", "claim:queue-workers"),
)


def test_real_minilm_recall_and_mrr_survive_restart(tmp_path: Path) -> None:
    embedder = SentenceTransformerEmbedder(
        model_name=DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        cache_folder=default_sentence_transformer_cache(),
    )
    database = tmp_path / "quality.sqlite3"
    backend = SQLiteGraphBackend(path=database, embedder=embedder)
    entities = {
        key
        for _, subject_key, object_key, _ in _CLAIMS
        for key in (subject_key, object_key)
    }
    backend.mutation.apply(
        MutationBatch(
            summary="MiniLM SQLite quality corpus",
            entity_upserts=[
                EntityUpsert(entity_key=key, labels=("Entity",))
                for key in sorted(entities)
            ],
            edge_upserts=[
                EdgeUpsert(
                    edge_type="DEPENDS_ON",
                    from_entity_key=subject_key,
                    to_entity_key=object_key,
                    properties={
                        "claim_key": claim_key,
                        "source_ref": f"quality:{claim_key}",
                        "description": description,
                        "fact": description,
                        "subgraph": "debugging",
                    },
                )
                for claim_key, subject_key, object_key, description in _CLAIMS
            ],
        ),
        expected_pot_id=POT,
    )

    _assert_quality(backend)
    _assert_quality(
        SQLiteGraphBackend(path=database, embedder=embedder)
    )


def _assert_quality(backend: SQLiteGraphBackend) -> None:
    reciprocal_ranks: list[float] = []
    recalled = 0
    for query, expected_claim_key in _QUERIES:
        rows = backend.semantic.search(pot_id=POT, query=query, k=5)
        ranked = [row.claim_key for row in rows]
        if expected_claim_key in ranked:
            rank = ranked.index(expected_claim_key) + 1
            recalled += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    recall_at_5 = recalled / len(_QUERIES)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    assert recall_at_5 == 1.0
    assert mrr >= 0.9
