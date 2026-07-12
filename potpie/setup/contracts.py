"""Root-owned setup and public status value objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol

DONE = "done"
SKIPPED = "skipped"
FAILED = "failed"
PLANNED = "planned"
NOT_IMPLEMENTED = "not_implemented"
_OK_STATES = frozenset({DONE, SKIPPED, PLANNED})


@dataclass(frozen=True, slots=True)
class SetupPlan:
    mode: str = "local"
    host_mode: str = "daemon"
    backend: str = "embedded"
    repo: str | None = "."
    pot: str = "default"
    agent: str = "claude"
    scan: bool = False
    assume_yes: bool = False
    defer_default_pot: bool = False
    defer_skills: bool = False
    embeddings: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass(frozen=True, slots=True)
class StepResult:
    step: str
    state: str
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
class PlannedSetupStep:
    step: str
    hard: bool
    owner: str
    action: str
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SetupPreview:
    plan: SetupPlan
    steps: tuple[PlannedSetupStep, ...]
    ok_to_run: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok_to_run": self.ok_to_run,
            "plan": asdict(self.plan),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class SetupReport:
    plan: SetupPlan
    steps: tuple[StepResult, ...]

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps if step.hard)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "plan": asdict(self.plan),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class ProductStatusResult:
    schema_version: str
    ready: bool
    runtime_mode: str
    daemon_state: str
    pot_id: str | None
    pot_name: str | None
    backend: str
    backend_ready: bool
    storage_ready: bool
    ingestion_ready: bool
    source_count: int
    last_ingestion_at: Any = None
    skills_state: str = "unknown"
    setup_state: str = "unknown"
    issues: tuple[str, ...] = ()
    recommended_next_action: Mapping[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["issues"] = list(self.issues)
        timestamp = data.get("last_ingestion_at")
        if timestamp is not None and hasattr(timestamp, "isoformat"):
            data["last_ingestion_at"] = timestamp.isoformat()
        return data


class SetupObserver(Protocol):
    def step_started(self, *, step: str, hard: bool) -> None: ...

    def step_completed(self, *, result: StepResult, duration_ms: int) -> None: ...


class SetupOrchestrator(Protocol):
    def set_observer(self, observer: SetupObserver) -> None: ...

    def preview(self, plan: SetupPlan) -> SetupPreview: ...

    def run(self, plan: SetupPlan) -> SetupReport: ...


__all__ = [
    "DONE",
    "FAILED",
    "NOT_IMPLEMENTED",
    "PLANNED",
    "SKIPPED",
    "PlannedSetupStep",
    "ProductStatusResult",
    "SetupPlan",
    "SetupObserver",
    "SetupOrchestrator",
    "SetupPreview",
    "SetupReport",
    "StepResult",
]
