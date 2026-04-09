"""Tests for :class:`DefaultContextGraphWriter` (Phase 6 execution boundary)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from adapters.outbound.context_graph_writer_adapter import DefaultContextGraphWriter
from domain.reconciliation import MutationSummary, ReconciliationPlan, ReconciliationResult
from domain.context_events import EventRef


def test_writer_apply_plan_delegates_to_use_case() -> None:
    episodic = MagicMock()
    structural = MagicMock()
    writer = DefaultContextGraphWriter(episodic, structural, mutation_applier=None)

    plan = ReconciliationPlan(
        event_ref=EventRef(event_id="e1", source_system="t", pot_id="p1"),
        summary="s",
        episodes=[],
        entity_upserts=[],
        edge_upserts=[],
        edge_deletes=[],
        invalidations=[],
        compat_github_pr_merged=None,
    )
    from unittest.mock import patch

    with patch(
        "adapters.outbound.context_graph_writer_adapter.apply_reconciliation_plan",
        return_value=ReconciliationResult(
            ok=True,
            episode_uuids=["u1"],
            mutation_summary=MutationSummary(episodes_written=1),
            error=None,
        ),
    ) as mock_apply:
        out = writer.apply_plan(plan, expected_pot_id="p1")
    mock_apply.assert_called_once()
    assert out.ok is True
    assert out.episode_uuids == ["u1"]


def test_writer_write_raw_episode_delegates() -> None:
    episodic = MagicMock()
    episodic.add_episode.return_value = "uuid-1"
    structural = MagicMock()
    writer = DefaultContextGraphWriter(episodic, structural)
    now = datetime.now(timezone.utc)
    out = writer.write_raw_episode(
        "pot1",
        "n",
        "body",
        "src",
        now,
    )
    assert out.get("episode_uuid") == "uuid-1"
    episodic.add_episode.assert_called_once()
