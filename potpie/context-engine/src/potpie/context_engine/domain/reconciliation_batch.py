"""Reconciliation batches: domain types for the single-agent ingestion flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from potpie.context_engine.domain.context_events import ContextEvent

BATCH_STATUS_PENDING = "pending"
BATCH_STATUS_CLAIMED = "claimed"
BATCH_STATUS_RUNNING = "running"
BATCH_STATUS_DONE = "done"
BATCH_STATUS_FAILED = "failed"


@dataclass(slots=True, frozen=True)
class BatchEventRef:
    """Reference to one event inside a batch (lightweight membership)."""

    event_id: str
    added_at: datetime
    processed_at: datetime | None = None


@dataclass(slots=True, frozen=True)
class ReconciliationBatch:
    """A pot-scoped batch of events awaiting agent processing."""

    id: str
    pot_id: str
    status: str
    attempt_count: int
    created_at: datetime
    claimed_at: datetime | None
    completed_at: datetime | None
    last_error: str | None


@dataclass(slots=True)
class BatchAgentContext:
    """Runtime input to ``ReconciliationAgentPort.run_batch``."""

    batch_id: str
    pot_id: str
    repo_name: str | None
    events: list[ContextEvent]
    prior_messages_json: list[dict[str, Any]] | None = None
    """Resumption checkpoint (pydantic-deep message history) when continuing a crashed run."""
    attempt_number: int = 1
    chunk_index: int = 0
    """0-based chunk this run covers (chunked batches). Namespaces streamed
    model-part ids and is persisted into the resume checkpoint."""
    chunks_total: int = 1
    start_seq: int = 0
    """Execution-log seq to continue from. Non-zero when resuming a crashed
    run or running a later chunk so the append-only log stays monotonic."""
    resume_completed_event_ids: list[str] = field(default_factory=list)
    """Events a prior (crashed) attempt already finished — the agent must
    not redo their side effects."""


@dataclass(slots=True)
class BatchAgentOutcome:
    """Result of running the agent over a batch."""

    ok: bool
    completed_event_ids: list[str] = field(default_factory=list)
    tool_call_count: int = 0
    error: str | None = None
    prompt: str | None = None
    """The user prompt passed to the agent (one record per batch run)."""
    agent_messages_json: list[dict[str, Any]] | None = None
    """Serialized ``result.all_messages()`` from pydantic-ai; the agent trace."""
    final_response: str | None = None
    """Last text response emitted by the agent (the wrap-up message), if any."""
    agent_name: str | None = None
    agent_version: str | None = None
    toolset_version: str | None = None
    last_seq: int = 0
    """Highest execution-log seq this run emitted. Threads into the next
    chunk's ``start_seq`` so the batch's append-only log stays monotonic."""


@dataclass(slots=True, frozen=True)
class AgentCheckpoint:
    """Persisted agent state for crash resumption + observability."""

    batch_id: str
    messages_json: list[dict[str, Any]]
    tool_call_count: int
    updated_at: datetime
