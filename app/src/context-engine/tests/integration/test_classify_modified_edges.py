"""classify_modified_edges maintenance job (mocked Neo4j driver)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphProvider


@pytest.mark.asyncio
async def test_classify_modified_edges_dry_run_empty() -> None:
    from adapters.outbound.graphiti.classify_modified_edges import (
        classify_modified_edges_for_group,
    )

    driver = MagicMock()
    driver.provider = GraphProvider.NEO4J
    driver.execute_query = AsyncMock(return_value=([], None, None))

    out = await classify_modified_edges_for_group(driver, "pot-1", dry_run=True)
    assert out["ok"] is True
    assert out["examined"] == 0
    assert out["would_update"] == 0
    driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_classify_modified_edges_dry_run_migration_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adapters.outbound.graphiti.classify_modified_edges import (
        classify_modified_edges_for_group,
    )

    driver = MagicMock()
    driver.provider = GraphProvider.NEO4J
    driver.execute_query = AsyncMock(
        return_value=(
            [
                {
                    "uuid": "e1",
                    "name": "MODIFIED",
                    "fact": "Ledger was migrated from Mongo to Postgres.",
                    "lifecycle_status": None,
                    "source_labels": ["Entity", "Service"],
                    "target_labels": ["Entity", "DataStore"],
                }
            ],
            None,
            None,
        )
    )

    out = await classify_modified_edges_for_group(driver, "pot-1", dry_run=True)
    assert out["ok"] is True
    assert out["examined"] == 1
    assert out["would_update"] == 1
    assert out["samples"][0]["new_name"] == "MIGRATED_TO"


@pytest.mark.asyncio
async def test_classify_write_applies_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_ALLOW_EDGE_CLASSIFY_WRITE", "1")
    monkeypatch.setenv("CONTEXT_ENGINE_CLASSIFY_MODIFIED_EDGES", "1")

    from adapters.outbound.graphiti.classify_modified_edges import (
        classify_modified_edges_for_group,
    )

    driver = MagicMock()
    driver.provider = GraphProvider.NEO4J
    driver.execute_query = AsyncMock(
        side_effect=[
            (
                [
                    {
                        "uuid": "e1",
                        "name": "MODIFIED",
                        "fact": "Service was migrated to Postgres.",
                        "lifecycle_status": None,
                        "source_labels": ["Entity", "Service"],
                        "target_labels": ["Entity", "DataStore"],
                    }
                ],
                None,
                None,
            ),
            ([{"uuid": "e1"}], None, None),
        ]
    )

    out = await classify_modified_edges_for_group(driver, "pot-1", dry_run=False)
    assert out.get("ok") is True
    assert out.get("updated") == 1
    assert driver.execute_query.await_count == 2


@pytest.mark.asyncio
async def test_classify_write_blocked_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_ALLOW_EDGE_CLASSIFY_WRITE", "0")

    from adapters.outbound.graphiti.classify_modified_edges import (
        classify_modified_edges_for_group,
    )

    driver = MagicMock()
    driver.provider = GraphProvider.NEO4J

    out = await classify_modified_edges_for_group(driver, "pot-1", dry_run=False)
    assert out.get("ok") is False
    assert out.get("error") == "write_blocked"
