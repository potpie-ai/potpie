"""Cost+drift telemetry — NoOp default and capture semantics."""

from __future__ import annotations

from datetime import datetime, timezone

from domain.ports.telemetry import (
    CostEvent,
    DriftSnapshot,
    NoOpTelemetry,
)


class _CaptureTelemetry:
    def __init__(self) -> None:
        self.cost: list[CostEvent] = []
        self.drift: list[DriftSnapshot] = []

    def record_cost(self, event: CostEvent) -> None:
        self.cost.append(event)

    def record_drift(self, snapshot: DriftSnapshot) -> None:
        self.drift.append(snapshot)


def test_noop_telemetry_swallows_calls():
    sink = NoOpTelemetry()
    sink.record_cost(CostEvent(pot_id="p1", kind="agent"))
    sink.record_drift(DriftSnapshot(pot_id="p1"))


def test_capture_telemetry_records_payloads():
    sink = _CaptureTelemetry()
    now = datetime.now(timezone.utc)
    sink.record_cost(
        CostEvent(
            pot_id="p1",
            kind="agent",
            model="openai:gpt-5.4-mini",
            input_tokens=120,
            output_tokens=60,
            total_tokens=180,
            latency_ms=2400,
            batch_id="b1",
            occurred_at=now,
            metadata={"completed_events": 3},
        )
    )
    sink.record_drift(
        DriftSnapshot(
            pot_id="p1",
            status="watch",
            source_ref_count=10,
            stale_ref_count=2,
            captured_at=now,
        )
    )
    assert sink.cost[0].kind == "agent"
    assert sink.cost[0].input_tokens == 120
    assert sink.drift[0].status == "watch"
    assert sink.drift[0].stale_ref_count == 2


def test_envelope_includes_meta_cost_and_quality_drift():
    """Envelope wires per-call latencies into ``meta.cost`` and signals into ``quality.drift``."""
    from domain.agent_context_port import bundle_to_agent_envelope
    from domain.intelligence_models import (
        ContextResolutionRequest,
        IntelligenceBundle,
        ResolutionMeta,
    )
    from domain.graph_quality import GraphQualityReport

    req = ContextResolutionRequest(pot_id="p1", query="ping")
    bundle = IntelligenceBundle(
        request=req,
        meta=ResolutionMeta(
            provider="MockIntelligenceProvider",
            total_latency_ms=120,
            per_call_latency_ms={"semantic_search": 90, "decision_context": 30},
        ),
        quality=GraphQualityReport(
            status="watch",
            metrics={
                "stale_ref_count": 2,
                "needs_verification_ref_count": 1,
                "verification_failed_ref_count": 0,
                "source_access_gap_count": 0,
                "missing_coverage_count": 1,
                "fallback_count": 0,
            },
        ),
    )
    env = bundle_to_agent_envelope(
        bundle,
        synthesis_usage={"model": "openai:gpt-5.4-mini", "input_tokens": 10, "output_tokens": 4},
    )
    assert env["meta"]["cost"]["resolve_ms"] == 120
    assert env["meta"]["cost"]["per_call_latency_ms"] == {
        "semantic_search": 90,
        "decision_context": 30,
    }
    assert env["meta"]["cost"]["synthesis"]["model"] == "openai:gpt-5.4-mini"
    drift = env["quality"]["drift"]
    assert drift["status"] == "watch"
    assert drift["signals"]["stale_refs"] == 2
    assert drift["signals"]["needs_verification_refs"] == 1
