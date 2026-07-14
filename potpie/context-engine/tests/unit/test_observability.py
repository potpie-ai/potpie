"""Phase A — observability spine: port contract, NoOp, console, wiring."""

from __future__ import annotations

import logging

import pytest

from potpie_context_engine.adapters.outbound.observability.console import (
    ConsoleObservability,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.bootstrap.ingestion_server import _default_observability
from potpie_context_engine.composition import build_engine_components
from potpie_context_engine.bootstrap.observability_context import (
    bind_correlation,
    correlation_scope,
    get_correlation,
    reset_correlation,
)
from potpie_context_engine.bootstrap.observability_runtime import (
    get_observability,
    set_observability,
)
from potpie_context_engine.domain.ports.observability import (
    NoOpObservability,
    ObservabilityPort,
)


@pytest.mark.unit
@pytest.mark.parametrize("obs", [NoOpObservability(), ConsoleObservability()])
def test_observability_port_contract(obs: ObservabilityPort) -> None:
    # span() must always yield a usable span, even when disabled.
    with obs.span("unit.work", attributes={"a": 1}, links=["bad-traceparent"]) as s:
        s.set_attribute("k", "v")
        s.set_attributes({"x": 1, "y": None})
        s.add_event("checkpoint", {"n": 2})
        s.set_error("explained")
    obs.counter("c_total", 3, attributes={"pot_id": "p"})
    obs.histogram("h_ms", 12.5)
    obs.gauge("g", 1)
    assert obs.current_traceparent() is None


@pytest.mark.unit
def test_span_records_and_reraises_exceptions() -> None:
    obs = ConsoleObservability()
    with pytest.raises(ValueError):
        with obs.span("boom"):
            raise ValueError("kaboom")


@pytest.mark.unit
def test_correlation_scope_merges_and_restores() -> None:
    assert get_correlation() == {}
    with correlation_scope(pot_id="p1", event_id="e1", not_a_key="x"):
        c = get_correlation()
        assert c == {"pot_id": "p1", "event_id": "e1"}
        with correlation_scope(batch_id="b1"):
            inner = get_correlation()
            assert inner == {"pot_id": "p1", "event_id": "e1", "batch_id": "b1"}
        assert get_correlation() == {"pot_id": "p1", "event_id": "e1"}
    assert get_correlation() == {}


@pytest.mark.unit
def test_bind_and_reset_token() -> None:
    tok = bind_correlation(run_id="r1")
    assert get_correlation()["run_id"] == "r1"
    reset_correlation(tok)
    assert "run_id" not in get_correlation()


@pytest.mark.unit
def test_default_observability_ships_dark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_OBSERVABILITY", raising=False)
    assert isinstance(_default_observability(), NoOpObservability)


@pytest.mark.unit
def test_default_observability_console_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_OBSERVABILITY", "console")
    assert isinstance(_default_observability(), ConsoleObservability)


@pytest.mark.unit
def test_default_observability_otel_requires_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Enabled but no OTLP endpoint configured → stays dark (NoOp).
    monkeypatch.setenv("CONTEXT_ENGINE_OBSERVABILITY", "1")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    assert isinstance(_default_observability(), NoOpObservability)


@pytest.mark.unit
def test_logging_setup_injects_correlation(caplog: pytest.LogCaptureFixture) -> None:
    from potpie_context_engine.bootstrap.logging_setup import CorrelationFilter

    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "hi", None, None)
    with correlation_scope(pot_id="pZ", event_id="eZ"):
        assert CorrelationFilter().filter(rec) is True
        assert rec.pot_id == "pZ"
        assert rec.event_id == "eZ"


@pytest.mark.unit
def test_engine_components_wire_process_observability() -> None:
    obs = NoOpObservability()
    original = get_observability()

    try:
        build_engine_components(backend=InMemoryGraphBackend(), observability=obs)
        assert get_observability() is obs
    finally:
        set_observability(original)
