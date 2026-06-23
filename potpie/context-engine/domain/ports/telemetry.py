"""Telemetry port: cost emission and drift snapshots.

Adapters persist or forward cost events and drift snapshots; the domain
emits them at well-defined call sites (LLM calls, post-resolve quality
assessment) without caring about the sink. The default :class:`NoOpTelemetry`
makes this safe to call unconditionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class CostEvent:
    """One LLM call attributable to the engine.

    ``kind`` is a stable label distinguishing where the call originated:

    - ``"agent"`` — the reconciliation agent batch run
    - ``"synthesis"`` — answer-summary synthesizer on the resolve path
    - ``"llm_extract"`` — the planner agent's LLM extraction passes
    - ``"connector"`` — connector-side LLM (e.g. Linear normalization)

    ``input_tokens`` / ``output_tokens`` may be None when the SDK did not
    surface usage. ``latency_ms`` is wall time of the call.
    """

    pot_id: str
    kind: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int | None = None
    batch_id: str | None = None
    event_id: str | None = None
    occurred_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DriftSnapshot:
    """Aggregated drift signals for one pot at one moment.

    All counts are non-negative integers; ``status`` is one of
    ``"good" | "watch" | "degraded" | "unknown"`` (matches the existing
    ``GraphQualityReport.status`` taxonomy so consumers don't need a second
    enum). ``metadata`` is free-form for adapter-specific context (e.g. the
    triggering query id).
    """

    pot_id: str
    status: str = "unknown"
    source_ref_count: int = 0
    stale_ref_count: int = 0
    needs_verification_ref_count: int = 0
    verification_failed_ref_count: int = 0
    source_access_gap_count: int = 0
    missing_coverage_count: int = 0
    fallback_count: int = 0
    open_conflicts_count: int = 0
    captured_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TelemetryPort(Protocol):
    """Cost + drift sink. Calls must be cheap and side-effect-only."""

    def record_cost(self, event: CostEvent) -> None: ...

    def record_drift(self, snapshot: DriftSnapshot) -> None: ...


class NoOpTelemetry:
    """Default implementation: discard everything. Test- and standalone-safe."""

    def record_cost(self, event: CostEvent) -> None:
        del event

    def record_drift(self, snapshot: DriftSnapshot) -> None:
        del snapshot
