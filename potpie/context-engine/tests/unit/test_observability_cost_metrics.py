"""Phase C — cost/drift→OTel bridge + composition-root infra proxy."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from potpie_context_engine.adapters.outbound.observability.telemetry_bridge import (
    ObservabilityTelemetryBridge,
)
from potpie_context_engine.bootstrap.observability_proxy import instrument_adapter
from potpie_context_engine.domain.ports.telemetry import CostEvent, DriftSnapshot


class Rec:
    def __init__(self) -> None:
        self.counters: list[tuple[str, float]] = []
        self.hist: list[tuple[str, float]] = []
        self.gauges: list[tuple[str, float]] = []

    @contextmanager
    def span(self, name, *, kind="internal", attributes=None, links=None):
        class _S:
            def set_attribute(self, *a):
                pass

            def set_error(self, *a):
                pass

            def record_exception(self, *a):
                pass

        yield _S()

    def counter(self, name, value=1, *, attributes=None):
        self.counters.append((name, value))

    def histogram(self, name, value, *, attributes=None):
        self.hist.append((name, value))

    def gauge(self, name, value, *, attributes=None):
        self.gauges.append((name, value))


class SpyTelemetry:
    def __init__(self) -> None:
        self.costs: list[CostEvent] = []
        self.drifts: list[DriftSnapshot] = []

    def record_cost(self, e: CostEvent) -> None:
        self.costs.append(e)

    def record_drift(self, s: DriftSnapshot) -> None:
        self.drifts.append(s)


@pytest.mark.unit
def test_bridge_preserves_inner_and_mirrors_cost() -> None:
    inner, obs = SpyTelemetry(), Rec()
    bridge = ObservabilityTelemetryBridge(inner, obs)
    bridge.record_cost(
        CostEvent(
            pot_id="p",
            kind="agent",
            model="m",
            input_tokens=10,
            output_tokens=4,
            total_tokens=14,
            latency_ms=33,
        )
    )
    # inner sink behavior preserved exactly ("bridge, not merge")
    assert len(inner.costs) == 1 and inner.costs[0].total_tokens == 14
    # mirrored into OTel metrics
    assert ("ce.llm.calls_total", 1) in obs.counters
    assert ("ce.llm.tokens_total", 14) in obs.counters
    assert ("ce.llm.latency_ms", 33.0) in obs.hist


@pytest.mark.unit
def test_bridge_mirrors_drift_as_gauges() -> None:
    inner, obs = SpyTelemetry(), Rec()
    ObservabilityTelemetryBridge(inner, obs).record_drift(
        DriftSnapshot(
            pot_id="p", status="degraded", stale_ref_count=7, open_conflicts_count=2
        )
    )
    assert len(inner.drifts) == 1
    assert ("ce.drift.stale_refs", 7) in obs.gauges
    assert ("ce.drift.open_conflicts", 2) in obs.gauges


@pytest.mark.unit
def test_bridge_forwards_unknown_attrs() -> None:
    class Extra(SpyTelemetry):
        def flush(self) -> str:
            return "flushed"

    bridge = ObservabilityTelemetryBridge(Extra(), Rec())
    assert bridge.flush() == "flushed"


@pytest.mark.unit
def test_bridge_never_raises_on_obs_failure() -> None:
    class Boom:
        def counter(self, *a, **k):
            raise RuntimeError("metrics down")

        def histogram(self, *a, **k):
            raise RuntimeError("metrics down")

        def gauge(self, *a, **k):
            raise RuntimeError("metrics down")

    inner = SpyTelemetry()
    bridge = ObservabilityTelemetryBridge(inner, Boom())
    bridge.record_cost(CostEvent(pot_id="p", kind="agent"))
    bridge.record_drift(DriftSnapshot(pot_id="p"))
    # inner still ran despite the mirror blowing up
    assert len(inner.costs) == 1 and len(inner.drifts) == 1


@pytest.mark.unit
def test_proxy_instruments_methods_and_forwards_attributes() -> None:
    calls: list[str] = []

    class FakeNeo4j:
        enabled = True

        def get_decisions(self, q: str) -> list[str]:
            calls.append(q)
            return [f"decision:{q}"]

        def _internal(self) -> str:
            return "private"

    obs = Rec()
    px = instrument_adapter(FakeNeo4j(), "neo4j", obs)

    # attribute forwarding (the duck-typed `.enabled` the policy reads)
    assert px.enabled is True
    # private attrs pass straight through, not wrapped
    assert px._internal() == "private"
    # public method runs, returns correctly, and is instrumented
    assert px.get_decisions("why") == ["decision:why"]
    assert calls == ["why"]
    assert any(n == "ce.neo4j.query_ms" for n, _ in obs.hist)


@pytest.mark.unit
def test_proxy_records_errors_and_reraises() -> None:
    class Boom:
        def get_thing(self):
            raise ValueError("neo4j exploded")

    obs = Rec()
    px = instrument_adapter(Boom(), "neo4j", obs)
    with pytest.raises(ValueError):
        px.get_thing()
    assert ("ce.neo4j.errors_total", 1) in obs.counters
    assert any(n == "ce.neo4j.query_ms" for n, _ in obs.hist)
