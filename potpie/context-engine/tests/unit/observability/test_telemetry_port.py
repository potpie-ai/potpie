"""Cost+drift telemetry — NoOp default and capture semantics."""

from __future__ import annotations

from potpie_context_engine.domain.ports.telemetry import (
    CostEvent,
    DriftSnapshot,
    NoOpTelemetry,
)


def test_noop_telemetry_swallows_calls():
    sink = NoOpTelemetry()
    sink.record_cost(CostEvent(pot_id="p1", kind="agent"))
    sink.record_drift(DriftSnapshot(pot_id="p1"))
