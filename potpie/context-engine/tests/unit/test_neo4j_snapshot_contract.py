"""Exact snapshot effects borrow the canonical mutation transaction."""

import pytest

from potpie_context_engine.adapters.outbound.graph.neo4j_snapshot_write import (
    merge_snapshot,
)
from potpie_context_engine.core.graph_snapshot_mutation import normalize_snapshot_import


class _Result:
    def __init__(self, rows=()):
        self.rows = list(rows)

    async def fetch(self, count):
        return self.rows[:count]

    async def consume(self):
        return None


class _Transaction:
    def __init__(self, *, entities=(), claims=()):
        self.entities = list(entities)
        self.claims = list(claims)
        self.queries = []

    async def run(self, query, **params):
        self.queries.append((query, params))
        if "RETURN entities,collect" in query:
            return _Result([{"entities": self.entities, "claims": self.claims}])
        return _Result()


def _payload(name="web"):
    return normalize_snapshot_import(
        {
            "format_version": "2",
            "pot_id": "target",
            "entities": [
                {
                    "key": "service:web",
                    "labels": ["Entity", "Service"],
                    "properties": {"name": name},
                }
            ],
            "claims": [],
        },
        pot_id="target",
    )


async def test_merge_validates_and_writes_without_owning_revision_or_receipt():
    tx = _Transaction()
    result, affected = await merge_snapshot(
        tx, pot_id="target", mutation_id="import", incoming=_payload()
    )
    assert result.administrative_summary == {
        "imported_entities": 1,
        "imported_claims": 0,
    }
    assert affected == ()
    assert any("MERGE (e:Entity" in query for query, _ in tx.queries)
    assert sum("RETURN entities,collect" in query for query, _ in tx.queries) == 1
    assert all(
        "PotpieRevision" not in query and "PotpieMutationReceipt" not in query
        for query, _ in tx.queries
    )


async def test_unnormalized_import_is_rejected_before_any_read_or_write():
    tx = _Transaction()
    payload = _payload()
    payload["entities"][0]["properties"].pop("record_id")
    with pytest.raises(ValueError, match="normalized stable record"):
        await merge_snapshot(
            tx, pot_id="target", mutation_id="import", incoming=payload
        )
    assert tx.queries == []


async def test_identical_merge_has_no_effects():
    payload = _payload()
    tx = _Transaction(
        entities=[
            {
                "key": "service:web",
                "labels": ["Entity", "Service"],
                "props": {
                    "group_id": "target",
                    "entity_key": "service:web",
                    **payload["entities"][0]["properties"],
                },
            }
        ]
    )
    result, affected = await merge_snapshot(
        tx, pot_id="target", mutation_id="import", incoming=payload
    )
    assert result.administrative_summary == {
        "imported_entities": 0,
        "imported_claims": 0,
    }
    assert affected == ()
    assert not any("MERGE (e:Entity" in query for query, _ in tx.queries)


async def test_conflict_is_detected_before_any_canonical_write():
    tx = _Transaction(
        entities=[
            {
                "key": "service:web",
                "labels": ["Entity", "Service"],
                "props": {
                    "group_id": "target",
                    "entity_key": "service:web",
                    "name": "old",
                },
            }
        ]
    )
    with pytest.raises(ValueError, match="conflicts with target"):
        await merge_snapshot(
            tx, pot_id="target", mutation_id="import", incoming=_payload("new")
        )
    assert not any("MERGE (e:Entity" in query for query, _ in tx.queries)
