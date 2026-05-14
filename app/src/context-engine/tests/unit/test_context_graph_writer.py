"""Tests for write methods on the unified Graphiti context graph adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from adapters.outbound.graphiti.context_graph import GraphitiContextGraphAdapter
from application.services.context_reader_registry import ContextReaderRegistry
from domain.reconciliation import MutationSummary, ReconciliationPlan, ReconciliationResult
from domain.context_events import EventRef


def test_context_graph_apply_plan_delegates_to_use_case() -> None:
    episodic = MagicMock()
    structural = MagicMock()
    graph = GraphitiContextGraphAdapter(
        episodic=episodic,
        structural=structural,
        readers=ContextReaderRegistry(),
    )

    plan = ReconciliationPlan(
        event_ref=EventRef(event_id="e1", source_system="t", pot_id="p1"),
        summary="s",
        episodes=[],
        entity_upserts=[],
        edge_upserts=[],
        edge_deletes=[],
        invalidations=[],
    )
    from unittest.mock import patch

    with patch(
        "adapters.outbound.graphiti.context_graph.apply_reconciliation_plan",
        return_value=ReconciliationResult(
            ok=True,
            episode_uuids=["u1"],
            mutation_summary=MutationSummary(episodes_written=1),
            error=None,
        ),
    ) as mock_apply:
        out = graph.apply_plan(plan, expected_pot_id="p1")
    mock_apply.assert_called_once()
    assert out.ok is True
    assert out.episode_uuids == ["u1"]


def test_context_graph_write_raw_episode_delegates() -> None:
    episodic = MagicMock()
    episodic.add_episode.return_value = "uuid-1"
    structural = MagicMock()
    graph = GraphitiContextGraphAdapter(
        episodic=episodic,
        structural=structural,
        readers=ContextReaderRegistry(),
    )
    now = datetime.now(timezone.utc)
    out = graph.write_raw_episode(
        "pot1",
        "n",
        "body",
        "src",
        now,
    )
    assert out.get("episode_uuid") == "uuid-1"
    episodic.add_episode.assert_called_once()


def test_context_graph_reset_pot_delegates_to_episodic() -> None:
    """Phase 1: episodic.reset_pot's full sweep is the only reset call."""
    episodic = MagicMock()
    episodic.reset_pot.return_value = {"ok": True}
    structural = MagicMock()
    graph = GraphitiContextGraphAdapter(
        episodic=episodic,
        structural=structural,
        readers=ContextReaderRegistry(),
    )

    out = graph.reset_pot("pot1")

    assert out == {
        "pot_id": "pot1",
        "ok": True,
        "episodic": {"ok": True},
    }
    episodic.reset_pot.assert_called_once_with("pot1")
    structural.reset_pot.assert_not_called()


def test_context_graph_reset_pot_stops_on_episodic_failure() -> None:
    episodic = MagicMock()
    episodic.reset_pot.return_value = {"ok": False, "error": "bad"}
    structural = MagicMock()
    graph = GraphitiContextGraphAdapter(
        episodic=episodic,
        structural=structural,
        readers=ContextReaderRegistry(),
    )

    out = graph.reset_pot("pot1")

    assert out == {
        "pot_id": "pot1",
        "ok": False,
        "episodic": {"ok": False, "error": "bad"},
        "error": "bad",
    }
    structural.reset_pot.assert_not_called()
