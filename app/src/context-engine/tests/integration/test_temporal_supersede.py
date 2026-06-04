"""Integration-style tests for predicate-family auto-supersede (mocked driver)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_auto_supersede_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_AUTO_SUPERSEDE", "0")
    from adapters.outbound.graphiti.temporal_supersede import (
        apply_predicate_family_auto_supersede,
    )

    driver = MagicMock()
    out = await apply_predicate_family_auto_supersede(driver, "pot-1")
    assert out.get("skipped") == "disabled"
    driver.execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_auto_supersede_invalidates_older_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strictly older ``valid_at`` is superseded; the newer binding stays live."""
    monkeypatch.setenv("CONTEXT_ENGINE_AUTO_SUPERSEDE", "1")

    from graphiti_core.driver.driver import GraphProvider

    from adapters.outbound.graphiti.temporal_supersede import (
        apply_predicate_family_auto_supersede,
    )

    driver = MagicMock()
    driver.provider = GraphProvider.NEO4J
    driver.execute_query = AsyncMock(
        side_effect=[
            (
                [
                    {
                        "uuid": "e-old",
                        "name": "USES_DATA_STORE",
                        "source": "svc-1",
                        "target": "mongo-1",
                        "target_labels": ["Entity"],
                        "valid_at": "2025-01-01T00:00:00+00:00",
                        "invalid_at": None,
                        "created_at": None,
                    },
                    {
                        "uuid": "e-new",
                        "name": "USES_DATA_STORE",
                        "source": "svc-1",
                        "target": "pg-1",
                        "target_labels": ["Entity"],
                        "valid_at": "2025-06-01T00:00:00+00:00",
                        "invalid_at": None,
                        "created_at": None,
                    },
                ],
                None,
                None,
            ),
            ([{"uuid": "e-old"}], None, None),
        ]
    )

    out = await apply_predicate_family_auto_supersede(driver, "pot-1")
    assert out["invalidated_count"] == 1
    assert driver.execute_query.await_count == 2


@pytest.mark.asyncio
async def test_auto_supersede_same_timestamp_two_objects_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same ``valid_at`` + different objects is a contradiction — leave for conflict detection."""
    monkeypatch.setenv("CONTEXT_ENGINE_AUTO_SUPERSEDE", "1")

    from graphiti_core.driver.driver import GraphProvider

    from adapters.outbound.graphiti.temporal_supersede import (
        apply_predicate_family_auto_supersede,
    )

    driver = MagicMock()
    driver.provider = GraphProvider.NEO4J
    driver.execute_query = AsyncMock(
        side_effect=[
            (
                [
                    {
                        "uuid": "e-a",
                        "name": "STORED_IN",
                        "source": "bar-1",
                        "target": "pg-1",
                        "target_labels": ["Entity", "DataStore"],
                        "valid_at": "2025-03-01T12:00:00+00:00",
                        "invalid_at": None,
                        "created_at": None,
                    },
                    {
                        "uuid": "e-b",
                        "name": "STORED_IN",
                        "source": "bar-1",
                        "target": "mysql-1",
                        "target_labels": ["Entity", "DataStore"],
                        "valid_at": "2025-03-01T12:00:00+00:00",
                        "invalid_at": None,
                        "created_at": None,
                    },
                ],
                None,
                None,
            ),
        ]
    )

    out = await apply_predicate_family_auto_supersede(driver, "pot-1")
    assert out["invalidated_count"] == 0
    driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_supersede_invalidates_cho_with_later_migrated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-type supersede: ``CHOSE`` (datastore hint) vs ``MIGRATED_TO`` (fix 02)."""
    monkeypatch.setenv("CONTEXT_ENGINE_AUTO_SUPERSEDE", "1")

    from graphiti_core.driver.driver import GraphProvider

    from adapters.outbound.graphiti.temporal_supersede import (
        apply_predicate_family_auto_supersede,
    )

    driver = MagicMock()
    driver.provider = GraphProvider.NEO4J
    driver.execute_query = AsyncMock(
        side_effect=[
            (
                [
                    {
                        "uuid": "e-chose",
                        "name": "CHOSE",
                        "source": "ledger-1",
                        "target": "mongo-1",
                        "target_labels": ["Entity", "DataStore"],
                        "valid_at": "2025-01-15T00:00:00+00:00",
                        "invalid_at": None,
                        "created_at": None,
                    },
                    {
                        "uuid": "e-mig",
                        "name": "MIGRATED_TO",
                        "source": "ledger-1",
                        "target": "pg-1",
                        "target_labels": ["Entity", "DataStore"],
                        "valid_at": "2025-08-12T00:00:00+00:00",
                        "invalid_at": None,
                        "created_at": None,
                    },
                ],
                None,
                None,
            ),
            ([{"uuid": "e-chose"}], None, None),
        ]
    )

    out = await apply_predicate_family_auto_supersede(driver, "pot-1")
    assert out["invalidated_count"] == 1
    assert driver.execute_query.await_count == 2
