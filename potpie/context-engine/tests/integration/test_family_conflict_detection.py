"""Integration-style conflict detection (mocked Neo4j driver)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_apply_family_conflict_detection_skipped_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_CONFLICT_DETECT", "0")
    from adapters.outbound.graphiti.family_conflict_detection import (
        apply_family_conflict_detection,
    )

    driver = MagicMock()
    out = await apply_family_conflict_detection(driver, "pot-1")
    assert out.get("skipped") == "disabled"
    driver.execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_apply_family_conflict_detection_creates_issue_mocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_CONFLICT_DETECT", "1")

    from graphiti_core.driver.driver import GraphProvider

    from adapters.outbound.graphiti.family_conflict_detection import (
        apply_family_conflict_detection,
    )

    t_old = "2025-01-15T00:00:00+00:00"
    t_new = "2025-08-15T00:00:00+00:00"

    driver = MagicMock()
    driver.provider = GraphProvider.NEO4J
    driver.execute_query = AsyncMock(
        side_effect=[
            (
                [
                    {
                        "uuid": "e-mongo",
                        "name": "USES_DATA_STORE",
                        "source": "svc-1",
                        "target": "mongo-1",
                        "valid_at": t_old,
                        "created_at": None,
                    },
                    {
                        "uuid": "e-pg",
                        "name": "USES_DATA_STORE",
                        "source": "svc-1",
                        "target": "pg-1",
                        "valid_at": t_new,
                        "created_at": None,
                    },
                ],
                None,
                None,
            ),
            ([], None, None),
            ([{"uuid": "qi-1"}], None, None),
            ([{"uuid": "qi-1"}], None, None),
        ]
    )

    out = await apply_family_conflict_detection(driver, "pot-x")
    assert out.get("ok") is True
    assert out.get("candidates", 0) >= 1
    assert driver.execute_query.await_count >= 2


@pytest.mark.asyncio
async def test_apply_family_conflict_detection_same_valid_at_contradiction_mocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same timestamp + different datastore targets → contradiction candidate."""
    monkeypatch.setenv("CONTEXT_ENGINE_CONFLICT_DETECT", "1")

    from graphiti_core.driver.driver import GraphProvider

    from adapters.outbound.graphiti.family_conflict_detection import (
        apply_family_conflict_detection,
    )

    t = "2025-03-01T12:00:00+00:00"

    driver = MagicMock()
    driver.provider = GraphProvider.NEO4J
    driver.execute_query = AsyncMock(
        side_effect=[
            (
                [
                    {
                        "uuid": "e-pg",
                        "name": "STORED_IN",
                        "source": "bar-1",
                        "target": "pg-1",
                        "target_labels": ["Entity", "DataStore"],
                        "valid_at": t,
                        "created_at": None,
                    },
                    {
                        "uuid": "e-mysql",
                        "name": "STORED_IN",
                        "source": "bar-1",
                        "target": "mysql-1",
                        "target_labels": ["Entity", "DataStore"],
                        "valid_at": t,
                        "created_at": None,
                    },
                ],
                None,
                None,
            ),
            ([], None, None),
            ([{"uuid": "qi-1"}], None, None),
            ([{"uuid": "qi-1"}], None, None),
        ]
    )

    out = await apply_family_conflict_detection(driver, "pot-x")
    assert out.get("ok") is True
    assert out.get("candidates") == 1
    assert out.get("issues_created") == 1
