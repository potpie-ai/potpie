"""Sole outbound port for mutating Graphiti and the structural graph (execution-owned)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from domain.reconciliation import ReconciliationPlan, ReconciliationResult


class ContextGraphWriter(Protocol):
    """
    Bounded API for deterministic graph writes used by step execution and sync reconcile.

    Planners and inbound adapters should not implement or call this directly; build a
    :class:`DefaultContextGraphWriter` in execution paths only.
    """

    def apply_plan(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
    ) -> ReconciliationResult:
        """Apply one validated reconciliation plan slice (episodes + structural effects)."""
        ...

    def write_raw_episode(
        self,
        pot_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
    ) -> dict[str, Any]:
        """Write a single raw episodic episode; returns ``{"episode_uuid": ...}``."""
        ...
