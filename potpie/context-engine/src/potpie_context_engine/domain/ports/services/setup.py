"""``SetupOrchestrator`` — the one first-run flow.

A single owner owns this sequence + its UX; each step it runs is a bespoke
per-component method owned independently (config, installer, backend, pots,
state store, migrator, daemon, auth, skills, ingest). The orchestrator is an
*application* concern so the managed API server reuses the identical sequence —
only the composition root swaps adapters.

    potpie setup           -> SetupOrchestrator.run(plan)      (ensure each, in order)
    potpie setup --dry-run -> SetupOrchestrator.preview(plan)  (describe; run nothing)
"""

from __future__ import annotations

from typing import Protocol

from potpie_context_core.lifecycle import SetupPlan, SetupPreview, SetupReport, StepResult


class SetupObserver(Protocol):
    def step_started(self, *, step: str, hard: bool) -> None: ...

    def step_completed(self, *, result: StepResult, duration_ms: int) -> None: ...


class NoOpSetupObserver:
    def step_started(self, *, step: str, hard: bool) -> None:
        del step, hard

    def step_completed(self, *, result: StepResult, duration_ms: int) -> None:
        del result, duration_ms


class SetupOrchestrator(Protocol):
    """Sequences the bespoke per-component lifecycle methods."""

    def set_observer(self, observer: SetupObserver) -> None: ...

    def preview(self, plan: SetupPlan) -> SetupPreview:
        """Dry-run: the ordered steps ``run`` would execute (owner, hard/soft,
        skip reason), unexecuted. Carries no ``StepResult``s."""
        ...

    def plan(self, plan: SetupPlan) -> list[StepResult]:
        """Deprecated alias for the ordered steps as ``PLANNED`` ``StepResult``s.

        Retained during the ``preview`` migration; prefer :meth:`preview`.
        """
        ...

    def run(self, plan: SetupPlan) -> SetupReport:
        """Execute each step in dependency order; never crash on a soft gap."""
        ...


__all__ = ["NoOpSetupObserver", "SetupObserver", "SetupOrchestrator"]
