"""Canonical ingestion event-store domain models (architecture vocabulary).

See docs/implementation-plans/ingestion-event-store-architecture.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

# --- Public lifecycle (stable API / dashboard) ---

IngestionEventStatus = Literal["queued", "processing", "done", "error"]
IngestionStepStatus = Literal["queued", "processing", "done", "error"]

# --- Internal observability (operators; may evolve independently of public status) ---

IngestionEventStage = Literal[
    "accepted",
    "planning",
    "planned",
    "executing",
    "completed",
    "failed",
]


@dataclass(frozen=True, slots=True)
class IngestionEvent:
    """Durable unit of requested ingestion work (parent object for APIs / dashboard)."""

    event_id: str
    pot_id: str
    ingestion_kind: str
    """Family of ingestion (e.g. ``agent_reconciliation``, ``raw_episode``)."""
    source_channel: str
    """Surface that submitted the event: ``cli``, ``http``, ``webhook``, etc."""
    source_system: str
    event_type: str
    action: str
    source_id: str
    """Stable id within provider scope (legacy ``context_events`` dedupe axis)."""
    dedup_key: str | None
    """Submission equivalence key; distinct from ``event_id`` (identity)."""
    status: IngestionEventStatus
    """Canonical lifecycle (``queued`` / ``processing`` / ``done`` / ``error``)."""
    stage: IngestionEventStage | None
    """Optional finer-grained stage; public ``status`` stays stable."""
    submitted_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error: str | None
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    step_total: int = 0
    step_done: int = 0
    step_error: int = 0
    provider: str = "github"
    provider_host: str = "github.com"
    repo_name: str = ""
    source_event_id: str | None = None
    job_id: str | None = None
    correlation_id: str | None = None
    idempotency_key: str | None = None
    occurred_at: datetime | None = None
    """When the source says the event occurred (if known)."""
    raw_status: str | None = None
    """Stored ``context_events.status`` before canonical mapping (legacy pipeline strings)."""


@dataclass(frozen=True, slots=True)
class IngestionPlan:
    """In-memory planner bundle (durable JSON is stored on ``context_reconciliation_runs.plan_json``)."""

    plan_id: str
    event_id: str
    planner_type: str
    version: int
    summary: str | None


@dataclass(frozen=True, slots=True)
class EpisodeStep:
    """One ordered execution unit derived from a plan."""

    step_id: str
    event_id: str
    pot_id: str
    sequence: int
    kind: str
    status: IngestionStepStatus
    input: dict[str, Any]
    attempt_count: int
    result: dict[str, Any] | None
    error: str | None
    queued_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Deterministic output of a single step executor."""

    step_id: str
    success: bool
    episode_ref: str | None
    """Graphiti / episodic reference when applicable."""
    structural_effects: dict[str, Any]
    """Summary counts or ids for structural graph writes."""
    error: str | None


@dataclass(frozen=True, slots=True)
class EventReceipt:
    """Return value after submit (async or sync waiter)."""

    event_id: str
    status: IngestionEventStatus
    terminal_event: IngestionEvent | None = None
    """When ``wait=True`` and terminal state reached, full event snapshot."""
    error: str | None = None
    duplicate: bool = False
    """True when an equivalent submission already exists (dedupe / idempotency)."""
    job_id: str | None = None
    """Async queue / correlation id when applicable."""
    episode_uuid: str | None = None
    """Raw episodic ingest: Graphiti episode id after inline apply when available."""
    reconciliation: Any | None = None
    """Agent reconciliation: populated on synchronous inline reconcile."""
    extras: dict[str, Any] | None = None
    """Opaque per-kind details (e.g. merged PR ``bridges`` / ``repo_name``)."""


@dataclass(frozen=True, slots=True)
class IngestionSubmissionRequest:
    """Canonical input to :class:`IngestionSubmissionService` after adapter normalization."""

    pot_id: str
    ingestion_kind: str
    source_channel: str
    source_system: str
    event_type: str
    action: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    """Optional client token; control plane maps to dedupe where applicable."""
    dedup_key: str | None = None
    """When set, used for duplicate detection; else control plane may derive."""
    event_id: str | None = None
    """Optional stable id (e.g. agent reconciliation); generated if omitted."""
    source_id: str | None = None
    """Required for ``agent_reconciliation`` (dedupe axis with repo scope)."""
    provider: str | None = None
    provider_host: str | None = None
    repo_name: str | None = None
    """When omitted, resolved from pot configuration."""
    source_event_id: str | None = None
    artifact_refs: tuple[str, ...] = ()
    occurred_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class EventTransition:
    """Parameters for atomic lifecycle updates."""

    to_status: IngestionEventStatus | None = None
    to_stage: IngestionEventStage | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class EventListFilters:
    """Filters for listing ingestion events."""

    statuses: tuple[IngestionEventStatus, ...] | None = None
    ingestion_kinds: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class EventListPage:
    """Cursor page of events (latest first)."""

    items: tuple[IngestionEvent, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class StepClaimResult:
    """Result of claiming the next executable step for a pot."""

    step: EpisodeStep | None
    """None if no queued work or lost race to another worker."""


@dataclass(frozen=True, slots=True)
class PlanWithSteps:
    """Planner output with ordered step definitions (steps persisted as ``context_episode_steps``)."""

    plan: IngestionPlan
    steps: tuple[EpisodeStep, ...]


@dataclass(frozen=True, slots=True)
class CreateIngestionEventParams:
    """Input to persist a new ingestion event (control plane assigns ``event_id`` upstream)."""

    event_id: str
    pot_id: str
    ingestion_kind: str
    source_channel: str
    source_system: str
    event_type: str
    action: str
    source_id: str
    dedup_key: str | None
    status: IngestionEventStatus
    stage: IngestionEventStage | None
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    submitted_at: datetime | None = None
    """Default: store uses transaction time if omitted."""
    provider: str = "github"
    provider_host: str = "github.com"
    repo_name: str = ""
