"""``DefaultSetupOrchestrator`` — the one first-run sequence.

Calls the bespoke per-component lifecycle methods in dependency order and folds
each outcome into a ``SetupReport``. Each step is wrapped so an unbuilt body
(``CapabilityNotImplemented``) becomes a ``not_implemented`` ``StepResult`` and
any other error becomes ``failed`` — setup runs what is ready and reports the
rest, never crashing. **Hard** steps gate ``SetupReport.ok``; **soft** steps do
not. See docs/context-graph/architecture.md "Component Lifecycle & Setup".

One ``_SEAM_PLAN`` table is the single source of truth for the ordered steps,
their owners, and their actions; ``preview`` (dry-run), ``plan`` (deprecated
alias), and ``run`` all derive from it, and hardness is derived from
``plan.host_mode`` so the daemon/installer steps are hard for a detached daemon
and skipped for an in-process host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from domain.errors import CapabilityNotImplemented
from domain.lifecycle import (
    DONE,
    FAILED,
    NOT_IMPLEMENTED,
    PLANNED,
    SKIPPED,
    PlannedSetupStep,
    SetupPlan,
    SetupPreview,
    SetupReport,
    StepResult,
)
from domain.ports.graph.backend import GraphBackend
from domain.ports.install import Installer
from domain.ports.services.auth import AuthService
from domain.ports.services.config import ConfigService
from domain.ports.services.pot_management import PotManagementService
from domain.ports.services.skill_manager import SkillManager
from domain.ports.services.state_store import MigrationPort, StateStorePort
from host.daemon import Daemon

# Dependency-ordered seam plan: (step, owner, action template). Mirrors the
# architecture.md "Seam → owner map"; the orchestrator depends only on each
# owner's method signature.
_SEAM_PLAN: tuple[tuple[str, str, str], ...] = (
    (
        "config",
        "config / workspace",
        "ensure home + write config.json (profile={mode})",
    ),
    (
        "installer",
        "installer / packaging",
        "ensure CLI on PATH + register service unit",
    ),
    ("backend.provision", "graph backend", "provision '{backend}' store"),
    ("pot.init", "pot management", "init control-plane state store (mode={mode})"),
    (
        "state_store.provision",
        "relational state store",
        "provision relational state store",
    ),
    ("migrator.migrate", "migrations", "run schema migrations"),
    ("pot.default", "pot management", "create + activate pot '{pot}'"),
    ("daemon", "daemon / host", "ensure host running"),
    ("auth", "auth", "init local auth"),
    ("source", "pot management", "register repo '{repo}'"),
    ("skills", "skill manager", "install skills for '{agent}'"),
    ("scan", "ingestion / scan", "scan working tree"),
)

# Soft steps never gate the run. Host-gated steps are hard only for a detached
# daemon host; an in-process host skips them.
_SOFT_STEPS = frozenset({"auth", "source", "skills", "scan"})
_HOST_GATED = frozenset({"installer", "daemon"})


def _is_hard(step: str, host_mode: str) -> bool:
    if step in _SOFT_STEPS:
        return False
    if step in _HOST_GATED:
        return host_mode == "daemon"
    return True


def _skip_reason(step: str, plan: SetupPlan) -> str | None:
    if step in _HOST_GATED and plan.host_mode != "daemon":
        return "in-process host — no detached daemon / service unit to manage"
    if step == "source" and not plan.repo:
        return "no --repo provided"
    if step == "pot.default" and plan.defer_default_pot:
        return "named in post-setup wizard"
    if step == "skills" and plan.defer_skills:
        return "chosen in post-setup wizard"
    return None


def _planned_steps(plan: SetupPlan) -> list[PlannedSetupStep]:
    """The ordered dry-run steps for ``plan`` (omitting scan unless requested)."""
    steps: list[PlannedSetupStep] = []
    for step, owner, action_tmpl in _SEAM_PLAN:
        if step == "scan" and not plan.scan:
            continue
        if step == "pot.default" and plan.defer_default_pot:
            continue
        if step == "skills" and plan.defer_skills:
            continue
        steps.append(
            PlannedSetupStep(
                step=step,
                hard=_is_hard(step, plan.host_mode),
                owner=owner,
                action=action_tmpl.format(
                    mode=plan.mode,
                    backend=plan.backend,
                    pot=plan.pot,
                    agent=plan.agent,
                    repo=plan.repo,
                ),
                skip_reason=_skip_reason(step, plan),
            )
        )
    return steps


@dataclass(slots=True)
class DefaultSetupOrchestrator:
    """Sequences config → installer → backend → pot.init → state store → migrate
    → default pot → daemon → auth → source → skills → scan over the bespoke
    per-component methods."""

    config: ConfigService
    installer: Installer
    backend: GraphBackend
    pots: PotManagementService
    state_store: StateStorePort
    migrator: MigrationPort
    daemon: Daemon
    auth: AuthService
    skills: SkillManager

    def preview(self, plan: SetupPlan) -> SetupPreview:
        """Dry-run: the ordered steps ``run`` would execute, unexecuted."""
        return SetupPreview(
            plan=plan, steps=tuple(_planned_steps(plan)), ok_to_run=True
        )

    def plan(self, plan: SetupPlan) -> list[StepResult]:
        """Deprecated alias: the planned steps as ``PLANNED`` ``StepResult``s."""
        return [
            StepResult(p.step, PLANNED, p.action, hard=p.hard)
            for p in _planned_steps(plan)
        ]

    def run(self, plan: SetupPlan) -> SetupReport:
        def hard(step: str) -> bool:
            return _is_hard(step, plan.host_mode)

        steps: list[StepResult] = [
            self._step("config", hard("config"), lambda: self._config(plan)),
            self._step("installer", hard("installer"), self._installer),
            self._step(
                "backend.provision",
                hard("backend.provision"),
                lambda: self.backend.provision(plan),
            ),
            self._step(
                "pot.init",
                hard("pot.init"),
                lambda: self.pots.init(mode=plan.mode, backend=plan.backend),
            ),
            self._step(
                "state_store.provision",
                hard("state_store.provision"),
                self.state_store.provision,
            ),
            self._step(
                "migrator.migrate", hard("migrator.migrate"), self.migrator.migrate
            ),
            *(
                [
                    self._step(
                        "pot.default",
                        hard("pot.default"),
                        lambda: self._default_pot(plan),
                    )
                ]
                if not plan.defer_default_pot
                else []
            ),
            self._step("daemon", hard("daemon"), lambda: self.daemon.ensure(plan)),
            self._step("auth", hard("auth"), self.auth.init_local),
            self._step("source", hard("source"), lambda: self._source(plan)),
            *(
                [
                    self._step(
                        "skills", hard("skills"), lambda: self._skills(plan)
                    )
                ]
                if not plan.defer_skills
                else []
            ),
        ]
        if plan.scan:
            steps.append(self._step("scan", hard("scan"), lambda: self._scan(plan)))
        return SetupReport(plan=plan, steps=tuple(steps))

    # --- step runner --------------------------------------------------------
    def _step(self, name: str, hard: bool, fn: Callable[[], object]) -> StepResult:
        try:
            result = fn()
        except CapabilityNotImplemented as exc:
            return StepResult(name, NOT_IMPLEMENTED, exc.detail or str(exc), hard=hard)
        except Exception as exc:  # never let one component crash the whole run
            return StepResult(name, FAILED, str(exc), hard=hard)
        if isinstance(result, StepResult):
            # Preserve the component's own state/detail; enforce this step's name + hard flag.
            return StepResult(
                name, result.state, result.detail, hard=hard, metadata=result.metadata
            )
        return StepResult(name, DONE, _describe(result), hard=hard)

    # --- step bodies --------------------------------------------------------
    def _config(self, plan: SetupPlan) -> str:
        self.config.ensure_home()
        path = self.config.write_defaults(plan)
        return f"config at {path}"

    def _installer(self) -> StepResult:
        if self.installer.is_installed():
            return StepResult("installer", SKIPPED, "CLI already available")
        self.installer.install_cli()
        self.installer.register_service()
        return StepResult("installer", DONE, "CLI installed + service registered")

    def _default_pot(self, plan: SetupPlan) -> str:
        pot = self.pots.create_pot(name=plan.pot, use=True)
        return f"active pot '{pot.name}' ({pot.pot_id})"

    def _source(self, plan: SetupPlan) -> StepResult:
        if not plan.repo:
            return StepResult("source", SKIPPED, "no --repo")
        active = self.pots.active_pot()
        if active is None:
            return StepResult("source", SKIPPED, "no active pot")
        existing = self.pots.list_sources(pot_id=active.pot_id)
        if any(s.kind == "repo" and s.name == plan.repo for s in existing):
            return StepResult(
                "source", SKIPPED, f"repo '{plan.repo}' already registered"
            )
        self.pots.add_source(pot_id=active.pot_id, kind="repo", location=plan.repo)
        return StepResult("source", DONE, f"registered repo '{plan.repo}'")

    def _skills(self, plan: SetupPlan) -> str | StepResult:
        if plan.agent.strip().lower() == "default":
            return StepResult("skills", SKIPPED, "no global skill target for default agent")
        result = self.skills.install(agent=plan.agent)
        return f"installed {list(result.changed)} for {plan.agent}"

    def _scan(self, plan: SetupPlan) -> StepResult:
        raise CapabilityNotImplemented(
            "ingest.scan",
            detail="scanner ingestion is not wired into setup yet",
            recommended_next_action="run 'potpie ingest scan' once scanners land",
        )


def _describe(result: object) -> str | None:
    return None if result is None else str(result)


__all__ = ["DefaultSetupOrchestrator"]
