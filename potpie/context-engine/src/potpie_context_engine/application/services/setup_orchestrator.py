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

import time
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Callable
from urllib.parse import urlparse

from potpie_context_engine.domain.errors import CapabilityNotImplemented
from potpie_context_engine.domain.embedding_modes import (
    EMBEDDING_MODEL_PREP_SKIPPED_ALIASES,
    SEMANTIC_EMBEDDER_ALIASES,
    normalize_embedding_mode,
)
from potpie_context_engine.domain.lifecycle import (
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
from potpie_context_engine.domain.ports.graph.backend import GraphBackend
from potpie_context_engine.domain.ports.install import Installer
from potpie_context_engine.domain.ports.services.auth import AuthService
from potpie_context_engine.domain.ports.services.config import ConfigService
from potpie_context_engine.domain.ports.services.pot_management import PotManagementService
from potpie_context_engine.domain.ports.services.skill_manager import SkillManager
from potpie_context_engine.domain.ports.services.state_store import MigrationPort, StateStorePort
from potpie_context_engine.domain.ports.services.setup import NoOpSetupObserver, SetupObserver
from potpie_context_engine.host.daemon import Daemon

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
    (
        "embeddings.model",
        "intelligence / embeddings",
        "prepare {embeddings} model '{embedding_model}'",
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
)

# Soft steps never gate the run. Host-gated steps are hard only for a detached
# daemon host; an in-process host skips them.
_SOFT_STEPS = frozenset({"auth", "source", "skills", "embeddings.model"})
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
    if step == "source" and plan.defer_default_pot:
        return "deferred until post-setup first pot"
    if step == "pot.default" and plan.defer_default_pot:
        return "named in post-setup wizard"
    if step == "skills" and plan.defer_skills:
        return "chosen in post-setup wizard"
    if (
        step == "embeddings.model"
        and normalize_embedding_mode(plan.embeddings)
        in EMBEDDING_MODEL_PREP_SKIPPED_ALIASES
    ):
        return f"embedding mode is {plan.embeddings}"
    return None


def _planned_steps(plan: SetupPlan) -> list[PlannedSetupStep]:
    """The ordered dry-run steps for ``plan``."""
    steps: list[PlannedSetupStep] = []
    for step, owner, action_tmpl in _SEAM_PLAN:
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
                    embeddings=plan.embeddings,
                    embedding_model=plan.embedding_model,
                ),
                skip_reason=_skip_reason(step, plan),
            )
        )
    return steps


@dataclass(slots=True)
class DefaultSetupOrchestrator:
    """Sequences config → installer → backend → pot.init → state store → migrate
    → default pot → daemon → auth → source → skills over the bespoke
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
    observer: SetupObserver = field(default_factory=NoOpSetupObserver)

    def set_observer(self, observer: SetupObserver) -> None:
        self.observer = observer

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
                "embeddings.model",
                hard("embeddings.model"),
                lambda: self._embedding_model(plan),
            ),
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
                [self._step("skills", hard("skills"), lambda: self._skills(plan))]
                if not plan.defer_skills
                else []
            ),
        ]
        return SetupReport(plan=plan, steps=tuple(steps))

    # --- step runner --------------------------------------------------------
    def _step(self, name: str, hard: bool, fn: Callable[[], object]) -> StepResult:
        self._notify_step_started(step=name, hard=hard)
        started = time.perf_counter()
        try:
            result = fn()
            if isinstance(result, StepResult):
                # Preserve the component's own state/detail; enforce this step's name + hard flag.
                step_result = StepResult(
                    name,
                    result.state,
                    result.detail,
                    hard=hard,
                    metadata=result.metadata,
                )
            else:
                step_result = StepResult(name, DONE, _describe(result), hard=hard)
        except CapabilityNotImplemented as exc:
            step_result = StepResult(
                name, NOT_IMPLEMENTED, exc.detail or str(exc), hard=hard
            )
        except Exception as exc:  # never let one component crash the whole run
            step_result = StepResult(name, FAILED, str(exc), hard=hard)
        duration_ms = int((time.perf_counter() - started) * 1000)
        self._notify_step_completed(result=step_result, duration_ms=duration_ms)
        return step_result

    def _notify_step_started(self, *, step: str, hard: bool) -> None:
        try:
            self.observer.step_started(step=step, hard=hard)
        except Exception:  # noqa: BLE001 - setup analytics must not affect setup.
            return

    def _notify_step_completed(self, *, result: StepResult, duration_ms: int) -> None:
        try:
            self.observer.step_completed(result=result, duration_ms=duration_ms)
        except Exception:  # noqa: BLE001 - setup analytics must not affect setup.
            return

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

    def _embedding_model(self, plan: SetupPlan) -> StepResult:
        mode = normalize_embedding_mode(plan.embeddings)
        if mode in EMBEDDING_MODEL_PREP_SKIPPED_ALIASES:
            return StepResult(
                "embeddings.model",
                SKIPPED,
                f"embedding mode is {plan.embeddings}",
                metadata={"mode": plan.embeddings},
            )
        embedder = getattr(self.backend, "embedder", None)
        prepare = getattr(embedder, "prepare", None)
        if embedder is None or not callable(prepare):
            if mode in SEMANTIC_EMBEDDER_ALIASES and (
                getattr(embedder, "name", None) == "local-hashing-v1"
            ):
                return StepResult(
                    "embeddings.model",
                    FAILED,
                    "sentence-transformers is unavailable; using local-hashing-v1",
                    metadata={
                        "mode": plan.embeddings,
                        "model": plan.embedding_model,
                        "fallback": "local-hashing-v1",
                    },
                )
            return StepResult(
                "embeddings.model",
                SKIPPED,
                "embedding model is not configurable for this backend",
                metadata={"mode": plan.embeddings},
            )
        try:
            metadata = prepare()
        except Exception as exc:  # noqa: BLE001 - setup should keep moving offline.
            return StepResult(
                "embeddings.model",
                FAILED,
                f"{plan.embedding_model} was not prepared: {exc}",
                metadata={"mode": plan.embeddings, "model": plan.embedding_model},
            )
        if not isinstance(metadata, dict):
            metadata = {}
        model = metadata.get("model") or plan.embedding_model
        cache = metadata.get("cache_folder")
        detail = f"{model} ready"
        if cache:
            detail = f"{detail} in {cache}"
        return StepResult(
            "embeddings.model",
            DONE,
            detail,
            metadata={**metadata, "mode": plan.embeddings, "model": model},
        )

    def _source(self, plan: SetupPlan) -> StepResult:
        if not plan.repo:
            return StepResult("source", SKIPPED, "no --repo provided")
        if plan.defer_default_pot:
            return StepResult("source", SKIPPED, "deferred until post-setup first pot")
        active = self.pots.active_pot()
        if active is None:
            return StepResult("source", SKIPPED, "no active pot")
        existing = self.pots.list_sources(pot_id=active.pot_id)
        if any(s.kind == "repo" and s.name == plan.repo for s in existing):
            return StepResult(
                "source", SKIPPED, f"repo '{plan.repo}' already registered"
            )
        location = _resolve_setup_repo_location(plan.repo)
        self.pots.add_source(pot_id=active.pot_id, kind="repo", location=location)
        self.pots.set_repo_default(repo=location, pot_id=active.pot_id)
        return StepResult("source", DONE, f"registered repo '{location}'")

    def _skills(self, plan: SetupPlan) -> str | StepResult:
        if plan.agent.strip().lower() == "default":
            return StepResult(
                "skills", SKIPPED, "no global skill target for default agent"
            )
        result = self.skills.install(agent=plan.agent)
        return f"installed {list(result.changed)} for {plan.agent}"


def _describe(result: object) -> str | None:
    return None if result is None else str(result)


__all__ = ["DefaultSetupOrchestrator"]


def _resolve_setup_repo_location(location: str) -> str:
    raw = (location or "").strip()
    if raw.lower() in (".", "current"):
        cwd = Path.cwd().resolve()
        remote = _current_git_remote(cwd)
        return remote or str(cwd)
    if raw.startswith((".", "~")):
        return str(Path(raw).expanduser().resolve(strict=False))
    return raw


def _current_git_remote(cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), "remote", "get-url", "origin"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return _normalize_repo_ref(proc.stdout.strip())


def _normalize_repo_ref(value: str) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    if raw.endswith(".git"):
        raw = raw[:-4]
    if raw.startswith("git@") and ":" in raw:
        host, path = raw[4:].split(":", 1)
        return f"{host}/{path}".strip("/")
    if "://" in raw:
        parsed = urlparse(raw)
        if parsed.netloc and parsed.path:
            return f"{parsed.netloc}/{parsed.path.strip('/')}"
    return raw
