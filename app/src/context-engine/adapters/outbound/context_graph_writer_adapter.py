"""Default :class:`ContextGraphWriter` — sole adapter from execution use cases to graph ports."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from application.use_cases.apply_reconciliation_plan import apply_reconciliation_plan
from application.use_cases.ingest_episode import ingest_episode as ingest_episode_uc
from domain.ports.context_graph_writer import ContextGraphWriter
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.graph_mutation_applier import GraphMutationApplierPort
from domain.ports.structural_graph import StructuralGraphPort
from domain.reconciliation import ReconciliationPlan, ReconciliationResult


class DefaultContextGraphWriter(ContextGraphWriter):
    """Bounded writes for episodic + structural graphs (used by step apply and sync reconcile)."""

    __slots__ = ("_episodic", "_structural", "_mutation_applier")

    def __init__(
        self,
        episodic: EpisodicGraphPort,
        structural: StructuralGraphPort,
        mutation_applier: GraphMutationApplierPort | None = None,
    ) -> None:
        self._episodic = episodic
        self._structural = structural
        self._mutation_applier = mutation_applier

    def apply_plan(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
    ) -> ReconciliationResult:
        return apply_reconciliation_plan(
            self._episodic,
            self._structural,
            plan,
            expected_pot_id=expected_pot_id,
            mutation_applier=self._mutation_applier,
        )

    def write_raw_episode(
        self,
        pot_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
    ) -> dict[str, Any]:
        return ingest_episode_uc(
            self._episodic,
            pot_id,
            name,
            episode_body,
            source_description,
            reference_time,
        )
