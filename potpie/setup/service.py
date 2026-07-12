"""Root-owned product setup sequencing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from potpie_context_engine.contracts import (
    PotCreateRequest,
    PotInfoRequest,
    ProvisionApplyRequest,
    ProvisionInspectRequest,
    SourceAddRequest,
    SourceListRequest,
)

from potpie.runtime.sync_view import await_engine
from potpie.setup.contracts import (
    DONE,
    FAILED,
    PLANNED,
    SKIPPED,
    PlannedSetupStep,
    SetupPlan,
    SetupPreview,
    SetupReport,
    StepResult,
)


class NoOpSetupObserver:
    def step_started(self, *, step: str, hard: bool) -> None:
        del step, hard

    def step_completed(self, *, result: StepResult, duration_ms: int) -> None:
        del result, duration_ms


@dataclass(slots=True)
class ProductSetupService:
    runtime: Any
    observer: Any = field(default_factory=NoOpSetupObserver)

    def set_observer(self, observer: Any) -> None:
        self.observer = observer

    def preview(self, plan: SetupPlan) -> SetupPreview:
        steps = (
            PlannedSetupStep("config", True, "potpie", "persist product settings"),
            PlannedSetupStep("installer", False, "potpie", "verify product install"),
            PlannedSetupStep("auth", False, "potpie", "inspect account state"),
            PlannedSetupStep(
                "daemon",
                plan.host_mode == "daemon",
                "potpie",
                "start or reconcile the product daemon",
                "in-process runtime" if plan.host_mode != "daemon" else None,
            ),
            PlannedSetupStep(
                "backend.provision", True, "potpie-context-engine", "provision engine"
            ),
            *(
                ()
                if plan.defer_default_pot
                else (
                    PlannedSetupStep(
                        "pot.default", True, "potpie-context-engine", "ensure pot"
                    ),
                )
            ),
            PlannedSetupStep(
                "source",
                False,
                "potpie-context-engine",
                "register repository source",
                "no repository" if not plan.repo else None,
            ),
            *(
                ()
                if plan.defer_skills
                else (
                    PlannedSetupStep("skills", False, "potpie", "install agent skills"),
                )
            ),
        )
        return SetupPreview(plan=plan, steps=steps)

    def plan(self, plan: SetupPlan) -> list[StepResult]:
        return [
            StepResult(step.step, PLANNED, step.action, hard=step.hard)
            for step in self.preview(plan).steps
        ]

    def run(self, plan: SetupPlan) -> SetupReport:
        steps: list[StepResult] = [
            self._step("config", True, lambda: self._configure(plan)),
            self._step("installer", False, self._installer),
            self._step("auth", False, self._auth),
            self._step(
                "daemon",
                plan.host_mode == "daemon",
                lambda: self._daemon(plan),
            ),
            self._step("backend.provision", True, lambda: self._provision(plan)),
        ]
        if not plan.defer_default_pot:
            steps.append(self._step("pot.default", True, lambda: self._pot(plan)))
        steps.append(self._step("source", False, lambda: self._source(plan)))
        if not plan.defer_skills:
            steps.append(self._step("skills", False, lambda: self._skills(plan)))
        return SetupReport(plan=plan, steps=tuple(steps))

    def _step(
        self, name: str, hard: bool, operation: Callable[[], StepResult | str]
    ) -> StepResult:
        try:
            self.observer.step_started(step=name, hard=hard)
        except Exception:  # noqa: BLE001, S110
            pass
        started = time.perf_counter()
        try:
            result = operation()
            step = (
                StepResult(
                    name,
                    result.state,
                    result.detail,
                    hard=hard,
                    metadata=result.metadata,
                )
                if isinstance(result, StepResult)
                else StepResult(name, DONE, str(result), hard=hard)
            )
        except Exception as exc:  # noqa: BLE001 - setup reports every failure.
            step = StepResult(name, FAILED, str(exc), hard=hard)
        duration_ms = int((time.perf_counter() - started) * 1000)
        try:
            self.observer.step_completed(result=step, duration_ms=duration_ms)
        except Exception:  # noqa: BLE001, S110
            pass
        return step

    def _configure(self, plan: SetupPlan) -> str:
        self.runtime.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.runtime.config.set("runtime_mode", plan.host_mode.replace("_", "-"))
        self.runtime.config.set("backend", plan.backend)
        return f"product config at {self.runtime.settings.data_dir / 'config.json'}"

    def _installer(self) -> StepResult:
        if self.runtime.installer.is_installed():
            return StepResult("installer", SKIPPED, "Potpie is already installed")
        self.runtime.installer.install_cli()
        self.runtime.installer.register_service()
        return StepResult("installer", DONE, "Potpie installed")

    def _auth(self) -> StepResult:
        identity = self.runtime.auth.whoami()
        return StepResult(
            "auth",
            DONE if identity.authenticated else SKIPPED,
            identity.subject if identity.authenticated else "not logged in",
        )

    def _daemon(self, plan: SetupPlan) -> StepResult:
        if plan.host_mode != "daemon":
            return StepResult("daemon", SKIPPED, "in-process runtime selected")
        status = self.runtime.daemon.status()
        if status.get("up") and status.get("backend") in {None, plan.backend}:
            return StepResult("daemon", SKIPPED, "daemon already running")
        if status.get("up"):
            self.runtime.daemon.stop()
        result = self.runtime.daemon.start(backend=plan.backend)
        return StepResult("daemon", DONE, f"daemon started (pid={result.get('pid')})")

    def _provision(self, plan: SetupPlan) -> StepResult:
        inspection = await_engine(
            self.runtime.engine.provision.inspect(
                ProvisionInspectRequest(pot_id=plan.pot)
            )
        )
        report = await_engine(
            self.runtime.engine.provision.apply(ProvisionApplyRequest(pot_id=plan.pot))
        )
        state = DONE if report.ok else FAILED
        details = ", ".join(f"{step.name}={step.state}" for step in report.steps)
        return StepResult(
            "backend.provision",
            state,
            details or f"{inspection.backend} inspected",
            metadata={"backend": report.backend},
        )

    def _pot(self, plan: SetupPlan) -> StepResult:
        active = await_engine(self.runtime.engine.pots.info(PotInfoRequest()))
        if active is not None:
            return StepResult("pot.default", SKIPPED, f"active pot '{active.name}'")
        created = await_engine(
            self.runtime.engine.pots.create(
                PotCreateRequest(name=plan.pot, repo=plan.repo, use=True)
            )
        )
        return StepResult("pot.default", DONE, f"active pot '{created.name}'")

    def _source(self, plan: SetupPlan) -> StepResult:
        if not plan.repo or plan.defer_default_pot:
            return StepResult("source", SKIPPED, "repository registration deferred")
        active = await_engine(self.runtime.engine.pots.info(PotInfoRequest()))
        if active is None:
            return StepResult("source", SKIPPED, "no active pot")
        sources = await_engine(
            self.runtime.engine.sources.list(SourceListRequest(pot_id=active.pot_id))
        )
        location = str(Path(plan.repo).expanduser().resolve())
        if any(
            item.kind == "repo" and item.location == location for item in sources.items
        ):
            return StepResult("source", SKIPPED, "repository already registered")
        await_engine(
            self.runtime.engine.sources.add(
                SourceAddRequest(
                    pot_id=active.pot_id,
                    kind="repo",
                    location=location,
                )
            )
        )
        return StepResult("source", DONE, f"registered repo '{location}'")

    def _skills(self, plan: SetupPlan) -> StepResult:
        result = self.runtime.skills.install(agent=plan.agent, scope="global")
        return StepResult(
            "skills",
            DONE if result.changed else SKIPPED,
            ", ".join(result.changed) or "skills already current",
        )


__all__ = ["NoOpSetupObserver", "ProductSetupService"]
