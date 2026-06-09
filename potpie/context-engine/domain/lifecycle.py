"""Shared value objects for the component lifecycle / setup flow.

``potpie setup`` is one orchestration (``SetupOrchestrator``) over bespoke
per-component methods. These value objects are the currency between them: a
``SetupPlan`` flows in, each step yields a ``StepResult``, and the run produces a
``SetupReport``. See docs/context-graph/architecture.md "Component Lifecycle &
Setup".

The orchestrator splits **hard deps** (must succeed) from **soft steps**
(best-effort): an unbuilt step raises ``CapabilityNotImplemented`` and is recorded
as ``not_implemented`` rather than crashing setup, so component owners can land
their step independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

# StepResult.state values.
DONE = "done"
SKIPPED = "skipped"
NOT_IMPLEMENTED = "not_implemented"
FAILED = "failed"
PLANNED = "planned"

_OK_STATES = frozenset({DONE, SKIPPED, PLANNED})


@dataclass(frozen=True, slots=True)
class SetupPlan:
    """The first-run intent the orchestrator executes (built from CLI flags)."""

    mode: str = "local"  # local setup; managed auth uses LoginPlan
    host_mode: str = "daemon"  # daemon | in_process — flips daemon/installer hardness
    backend: str = "embedded"  # embedded | postgres | neo4j | in_memory
    repo: str | None = "."
    pot: str = "foo-pot"
    agent: str = "claude"
    scan: bool = False
    assume_yes: bool = False
    defer_default_pot: bool = False
    defer_skills: bool = False


@dataclass(frozen=True, slots=True)
class StepResult:
    """Outcome of one setup step. ``hard`` steps gate ``SetupReport.ok``."""

    step: str
    state: str  # done | skipped | not_implemented | failed | planned
    detail: str | None = None
    hard: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.state in _OK_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "state": self.state,
            "detail": self.detail,
            "hard": self.hard,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SetupReport:
    """The full result of a ``setup`` run: the plan + every step's result."""

    plan: SetupPlan
    steps: tuple[StepResult, ...]

    @property
    def ok(self) -> bool:
        """True when every *hard* step succeeded (soft steps never gate ok)."""
        return all(step.ok for step in self.steps if step.hard)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "plan": _plan_dict(self.plan),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class PlannedSetupStep:
    """One step as it appears in a dry-run ``SetupPreview`` (unexecuted).

    Carries the owner + hard/soft classification + skip reason so an operator
    can read the plan — and a component owner can find their seam — before
    anything runs. ``hard`` mirrors ``StepResult.hard``: hard steps gate the run.
    """

    step: str
    hard: bool
    owner: str
    action: str
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "hard": self.hard,
            "owner": self.owner,
            "action": self.action,
            "skip_reason": self.skip_reason,
        }


@dataclass(frozen=True, slots=True)
class SetupPreview:
    """The result of ``setup --dry-run``: the plan + the steps it *would* run.

    Distinct from ``SetupReport`` — a preview never executes a step and never
    carries ``StepResult``s. ``ok_to_run`` is False when a precondition would
    block the run outright (today always True; reserved for the daemon owner).
    """

    plan: SetupPlan
    steps: tuple[PlannedSetupStep, ...]
    ok_to_run: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": True,
            "ok_to_run": self.ok_to_run,
            "plan": _plan_dict(self.plan),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class LoginPlan:
    """The managed-backend login intent (separate lifecycle from local setup).

    ``potpie login`` authenticates against ``backend_url`` (defaults to the
    configured ``cloud.backend_url``) and binds managed pots into the same pot
    surface. The managed-auth owner consumes this (see HU3).
    """

    backend_url: str | None = None
    org: str | None = None


def _plan_dict(plan: SetupPlan) -> dict[str, Any]:
    return {
        "mode": plan.mode,
        "host_mode": plan.host_mode,
        "backend": plan.backend,
        "repo": plan.repo,
        "pot": plan.pot,
        "agent": plan.agent,
        "scan": plan.scan,
        "defer_default_pot": plan.defer_default_pot,
        "defer_skills": plan.defer_skills,
    }


__all__ = [
    "DONE",
    "FAILED",
    "NOT_IMPLEMENTED",
    "PLANNED",
    "SKIPPED",
    "LoginPlan",
    "PlannedSetupStep",
    "SetupPlan",
    "SetupPreview",
    "SetupReport",
    "StepResult",
]
