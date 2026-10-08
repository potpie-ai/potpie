"""Unit tests for Ladybug ensure_schema / claim Cypher shapes (fake connection)."""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.graph.ladybug_vector import (
    LadybugVectorConfig,
)
from potpie_context_engine.adapters.outbound.graph.ladybug_writer import (
    _VECTOR_INDEX,
    _records_from_result,
    ensure_schema,
)

pytestmark = pytest.mark.unit


class _FakeConn:
    def __init__(self, *, schema_exists: bool = False) -> None:
        self.schema_exists = schema_exists
        self.calls: list[tuple[str, dict | None]] = []

    def execute(self, cypher: str, params: dict | None = None):
        self.calls.append((cypher, params))
        if "MATCH (e:Entity)" in cypher and not self.schema_exists:
            raise RuntimeError("Binder exception: Table Entity does not exist.")
        return []


class _ClosableResult:
    def __init__(self) -> None:
        self.closed = False

    def get_column_names(self) -> list[str]:
        return ["value"]

    def __iter__(self):
        return iter([["row"]])

    def close(self) -> None:
        self.closed = True


def test_ensure_schema_creates_tables_and_vector_index() -> None:
    conn = _FakeConn(schema_exists=False)
    cfg = LadybugVectorConfig(mu=32, ml=64, efc=400, cache_embeddings=True)
    assert ensure_schema(conn, 8, vector_config=cfg) is True
    joined = "\n".join(c[0] for c in conn.calls)
    assert "CREATE NODE TABLE Entity" in joined
    assert "CREATE NODE TABLE Claim" in joined
    assert "FLOAT[8]" in joined
    assert "CREATE REL TABLE CLAIM_SUBJECT" in joined
    assert "CREATE REL TABLE CLAIM_OBJECT" in joined
    assert "CREATE_VECTOR_INDEX" in joined
    assert f"'{_VECTOR_INDEX}'" in joined
    assert "'Claim'" in joined
    assert "mu := 32" in joined
    assert "cache_embeddings := true" in joined


def test_records_from_result_closes_native_result() -> None:
    result = _ClosableResult()

    assert _records_from_result(result) == [{"value": "row"}]
    assert result.closed is True


def test_ensure_schema_skips_ddl_when_entity_table_exists() -> None:
    conn = _FakeConn(schema_exists=True)
    assert ensure_schema(conn, 256) is True
    joined = "\n".join(c[0] for c in conn.calls)
    assert "CREATE NODE TABLE Entity" not in joined
    assert "LOAD VECTOR" in joined
    assert "CREATE_VECTOR_INDEX" in joined
