"""Graphiti post-write pass stamps prov_* on the Episodic node and extracted edges."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapters.outbound.graphiti.apply_episode_provenance import (
    apply_episode_provenance,
)
from domain.graph_mutations import ProvenanceRef


def _driver_with_neo4j_provider() -> MagicMock:
    from graphiti_core.driver.driver import GraphProvider

    drv = MagicMock()
    drv.provider = GraphProvider.NEO4J
    # execute_query returns (records, keys, summary)
    drv.execute_query = AsyncMock(return_value=([{"cnt": 3}], None, None))
    return drv


@pytest.mark.asyncio
async def test_apply_episode_provenance_stamps_node_and_edges() -> None:
    drv = _driver_with_neo4j_provider()
    prov = ProvenanceRef(
        pot_id="pot-1",
        source_event_id="evt-1",
        source_system="github",
        source_kind="pull_request",
        source_ref="github/pr/42",
        event_occurred_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        graph_updated_at=datetime(2026, 4, 1, 0, 5, tzinfo=timezone.utc),
        confidence=0.9,
        created_by_agent="ingestion-agent",
        reconciliation_run_id="run-1",
    )

    out = await apply_episode_provenance(drv, "pot-1", "ep-uuid", prov)

    assert out["ok"] is True
    assert out["edges_stamped"] == 3
    # Two Cypher calls: node stamp + edge stamp.
    assert drv.execute_query.await_count == 2

    node_call = drv.execute_query.await_args_list[0]
    node_props = node_call.kwargs["props"]
    assert node_props["prov_source_system"] == "github"
    assert node_props["prov_source_ref"] == "github/pr/42"
    assert node_props["prov_reconciliation_run_id"] == "run-1"

    edge_call = drv.execute_query.await_args_list[1]
    assert edge_call.kwargs["episode_uuid"] == "ep-uuid"
    assert edge_call.kwargs["gid"] == "pot-1"
    assert "IN coalesce(r.episodes" in edge_call.args[0]


@pytest.mark.asyncio
async def test_apply_episode_provenance_noop_when_no_episode() -> None:
    drv = _driver_with_neo4j_provider()
    out = await apply_episode_provenance(
        drv, "pot-1", "", ProvenanceRef(pot_id="p", source_event_id="e")
    )
    assert out["ok"] is True
    assert out["skipped"] == "no_episode_or_provenance"
    drv.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_episode_provenance_noop_when_no_provenance() -> None:
    drv = _driver_with_neo4j_provider()
    out = await apply_episode_provenance(drv, "pot-1", "ep-1", None)
    assert out["skipped"] == "no_episode_or_provenance"
    drv.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_episode_provenance_skips_non_neo4j_driver() -> None:
    drv = MagicMock()
    drv.provider = "SQLITE"
    drv.execute_query = AsyncMock()
    out = await apply_episode_provenance(
        drv, "pot-1", "ep-1", ProvenanceRef(pot_id="p", source_event_id="e")
    )
    assert out["skipped"] == "unsupported_provider"
    drv.execute_query.assert_not_awaited()
