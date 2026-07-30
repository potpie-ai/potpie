"""Versioned schema and persisted embedding contract for the SQLite backend."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping

from potpie_context_engine.adapters.outbound.graph.sqlite.connection import (
    SQLITE_VEC_VERSION,
    SQLiteGraphStoreError,
)

SCHEMA_VERSION = "1"
EMBEDDING_PROVIDER = "sentence-transformers"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BYTES = EMBEDDING_DIM * 4
RETRIEVAL_CARD_VERSION = "1"

METADATA_CONTRACT: Mapping[str, str] = {
    "schema_version": SCHEMA_VERSION,
    "embedding_provider": EMBEDDING_PROVIDER,
    "embedding_model": EMBEDDING_MODEL,
    "embedding_dim": str(EMBEDDING_DIM),
    "retrieval_card_version": RETRIEVAL_CARD_VERSION,
    "sqlite_vec_version": SQLITE_VEC_VERSION,
}

_CORE_SCHEMA = f"""
CREATE TABLE graph_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE entities (
    pot_id TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    properties_json TEXT NOT NULL DEFAULT '{{}}',
    PRIMARY KEY (pot_id, entity_key)
);

CREATE TABLE entity_labels (
    pot_id TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    label TEXT NOT NULL,
    PRIMARY KEY (pot_id, entity_key, label),
    FOREIGN KEY (pot_id, entity_key)
        REFERENCES entities (pot_id, entity_key) ON DELETE CASCADE
);

CREATE TABLE claims (
    claim_id INTEGER PRIMARY KEY,
    pot_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    subject_key TEXT NOT NULL,
    object_key TEXT NOT NULL,
    claim_key TEXT,
    subgraph TEXT,
    source_system TEXT,
    source_ref TEXT,
    fact TEXT,
    description TEXT,
    retrieval_card TEXT NOT NULL,
    properties_json TEXT NOT NULL DEFAULT '{{}}',
    source_refs_json TEXT NOT NULL DEFAULT '[]',
    evidence_json TEXT NOT NULL DEFAULT '[]',
    truth TEXT,
    confidence REAL,
    environment TEXT,
    valid_at TEXT,
    valid_until TEXT,
    invalid_at TEXT,
    observed_at TEXT,
    mutation_id TEXT NOT NULL,
    graph_contract_version TEXT,
    ontology_version TEXT,
    embedding BLOB NOT NULL
        CHECK(typeof(embedding) = 'blob' AND length(embedding) = {EMBEDDING_BYTES}),
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL CHECK(embedding_dim = {EMBEDDING_DIM})
);

CREATE TABLE mutation_receipts (
    mutation_id TEXT PRIMARY KEY,
    pot_id TEXT NOT NULL,
    batch_fingerprint TEXT NOT NULL,
    result_json TEXT NOT NULL
);

CREATE INDEX claims_pot_live
    ON claims (pot_id, invalid_at, claim_id);
CREATE INDEX claims_pot_predicate
    ON claims (pot_id, predicate, invalid_at);
CREATE INDEX claims_pot_subject
    ON claims (pot_id, subject_key, invalid_at);
CREATE INDEX claims_pot_object
    ON claims (pot_id, object_key, invalid_at);
CREATE INDEX claims_pot_valid_at
    ON claims (pot_id, valid_at);
CREATE INDEX claims_pot_mutation
    ON claims (pot_id, mutation_id);
CREATE UNIQUE INDEX claims_live_claim_key
    ON claims (pot_id, claim_key)
    WHERE invalid_at IS NULL AND claim_key IS NOT NULL;
CREATE UNIQUE INDEX claims_live_source_edge
    ON claims (pot_id, source_ref, predicate, subject_key, object_key)
    WHERE invalid_at IS NULL AND source_ref IS NOT NULL;
"""

_VECTOR_SCHEMA = f"""
CREATE VIRTUAL TABLE claim_vectors USING vec0(
    claim_id INTEGER PRIMARY KEY,
    embedding float[{EMBEDDING_DIM}] distance_metric=cosine,
    pot_id TEXT PARTITION KEY,
    predicate TEXT,
    subgraph TEXT,
    subject_key TEXT,
    object_key TEXT,
    source_system TEXT,
    mutation_id TEXT
);
"""

_REQUIRED_TABLES = frozenset(
    {
        "graph_metadata",
        "entities",
        "entity_labels",
        "claims",
        "mutation_receipts",
        "claim_vectors",
    }
)


def ensure_schema(connection: sqlite3.Connection) -> None:
    """Provision an empty database or verify the exact persisted contract."""

    objects = {
        str(row[0])
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_schema
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            """
        )
    }
    if "graph_metadata" not in objects:
        if objects:
            raise SQLiteGraphStoreError(
                "refusing to provision the sqlite graph schema into a non-empty "
                f"database (found: {', '.join(sorted(objects))})"
            )
        try:
            # ``executescript`` commits an existing transaction before running.
            # Put BEGIN in the script itself so core tables, vec0 shadow tables,
            # and metadata still form one rollback-safe provisioning unit.
            connection.executescript(
                f"BEGIN IMMEDIATE;\n{_CORE_SCHEMA}\n{_VECTOR_SCHEMA}"
            )
            connection.executemany(
                "INSERT INTO graph_metadata(key, value) VALUES (?, ?)",
                tuple(METADATA_CONTRACT.items()),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        objects = _schema_objects(connection)

    missing_tables = sorted(_REQUIRED_TABLES - objects)
    if missing_tables:
        raise SQLiteGraphStoreError(
            "sqlite graph schema is incomplete; missing "
            + ", ".join(missing_tables)
        )

    actual = {
        str(row["key"]): str(row["value"])
        for row in connection.execute("SELECT key, value FROM graph_metadata")
    }
    mismatches = {
        key: (expected, actual.get(key))
        for key, expected in METADATA_CONTRACT.items()
        if actual.get(key) != expected
    }
    if mismatches or set(actual) != set(METADATA_CONTRACT):
        detail = ", ".join(
            f"{key}={found!r} (expected {expected!r})"
            for key, (expected, found) in sorted(mismatches.items())
        )
        extras = sorted(set(actual) - set(METADATA_CONTRACT))
        if extras:
            detail = f"{detail}; unexpected keys: {', '.join(extras)}".strip("; ")
        raise SQLiteGraphStoreError(
            f"sqlite graph metadata contract mismatch: {detail or 'unknown mismatch'}"
        )


def verify_vector_projection(connection: sqlite3.Connection) -> None:
    """Reject live canonical/vector drift and incompatible claim embeddings."""

    invalid = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM claims
        WHERE embedding_model <> ?
           OR embedding_dim <> ?
           OR typeof(embedding) <> 'blob'
           OR length(embedding) <> {EMBEDDING_BYTES}
        """,
        (EMBEDDING_MODEL, EMBEDDING_DIM),
    ).fetchone()[0]
    if int(invalid):
        raise SQLiteGraphStoreError(
            f"sqlite graph contains {int(invalid)} incompatible claim embeddings"
        )

    missing = connection.execute(
        """
        SELECT COUNT(*)
        FROM claims AS c
        LEFT JOIN claim_vectors AS v ON v.claim_id = c.claim_id
        WHERE c.invalid_at IS NULL AND v.claim_id IS NULL
        """
    ).fetchone()[0]
    stale = connection.execute(
        """
        SELECT COUNT(*)
        FROM claim_vectors AS v
        LEFT JOIN claims AS c ON c.claim_id = v.claim_id
        WHERE c.claim_id IS NULL OR c.invalid_at IS NOT NULL
        """
    ).fetchone()[0]
    if int(missing) or int(stale):
        raise SQLiteGraphStoreError(
            "sqlite vector projection drift detected "
            f"(missing={int(missing)}, stale={int(stale)})"
        )


def _schema_objects(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_schema
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            """
        )
    }


__all__ = [
    "EMBEDDING_BYTES",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "EMBEDDING_PROVIDER",
    "METADATA_CONTRACT",
    "RETRIEVAL_CARD_VERSION",
    "SCHEMA_VERSION",
    "ensure_schema",
    "verify_vector_projection",
]
