"""Neo4j snapshot transaction contract without requiring a server."""

import pytest

from potpie_context_engine.adapters.outbound.graph.neo4j_snapshot import (
    _ensure_revision_constraint,
    _merge,
)


class _Result:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def __iter__(self):
        return iter(self.rows)

    def single(self):
        return self.rows[0] if self.rows else None

    def consume(self):
        return None


class _Transaction:
    def __init__(self, *, entities=(), claims=()):
        self.entities = list(entities)
        self.claims = list(claims)
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))
        if "RETURN v.version" in query:
            return _Result([{"version": 2}])
        if "RETURN entities,collect" in query:
            return _Result([{"entities": self.entities, "claims": self.claims}])
        return _Result()


def _payload(name="web"):
    return {
        "format_version": "2", "pot_id": "target",
        "entities": [{"key": "service:web", "labels": ["Entity", "Service"], "properties": {"name": name}}],
        "claims": [],
    }


def test_merge_locks_validates_writes_and_advances_revision() -> None:
    tx = _Transaction()
    _merge(tx, "target", _payload())
    assert "lock_sequence" in tx.queries[0][0]
    assert any("MERGE (e:Entity" in query for query, _ in tx.queries)
    assert "v.version=v.version+1" in tx.queries[-1][0]
    assert sum("RETURN entities,collect" in query for query, _ in tx.queries) == 1


def test_revision_constraint_is_installed_before_using_merge_as_lock() -> None:
    tx = _Transaction()
    _ensure_revision_constraint(tx)
    assert "REQUIRE r.pot_id IS UNIQUE" in tx.queries[0][0]


def test_identical_retry_does_not_advance_revision() -> None:
    tx = _Transaction(
        entities=[
            {
                "key": "service:web",
                "labels": ["Entity", "Service"],
                "props": {
                    "group_id": "target",
                    "entity_key": "service:web",
                    "name": "web",
                },
            }
        ]
    )
    _merge(tx, "target", _payload())
    assert not any("MERGE (e:Entity" in query for query, _ in tx.queries)
    assert not any("v.version=v.version+1" in query for query, _ in tx.queries)


def test_conflict_is_detected_before_any_canonical_write() -> None:
    tx = _Transaction(
        entities=[{"key": "service:web", "labels": ["Entity", "Service"], "props": {"group_id": "target", "entity_key": "service:web", "name": "old"}}]
    )
    with pytest.raises(ValueError, match="conflicts with target"):
        _merge(tx, "target", _payload("new"))
    assert not any("MERGE (e:Entity" in query for query, _ in tx.queries)
    assert not any("v.version=v.version+1" in query for query, _ in tx.queries)
