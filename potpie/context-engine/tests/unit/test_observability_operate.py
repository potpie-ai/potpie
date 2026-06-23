"""Phase D — readiness probes + JSON logging with correlation/audit."""

from __future__ import annotations

import json
import logging

import pytest

from potpie.context_engine.adapters.inbound.http.api.router import (
    _check_postgres,
    _check_redis,
    health,
)
from potpie.context_engine.bootstrap.logging_setup import CorrelationFilter, JsonFormatter
from potpie.context_engine.bootstrap.observability_context import correlation_scope


@pytest.mark.unit
def test_health_contract_unchanged() -> None:
    assert health() == {"status": "ok"}


@pytest.mark.unit
def test_check_postgres_states() -> None:
    class DB:
        def execute(self, _q):
            return 1

    class BadDB:
        def execute(self, _q):
            raise RuntimeError("pg down")

    assert _check_postgres(DB()) == (True, None)
    assert _check_postgres(None)[0] is False
    ok, detail = _check_postgres(BadDB())
    assert ok is False and "pg down" in detail


@pytest.mark.unit
def test_check_redis_optional_when_noop() -> None:
    class Container:
        class _Pub:  # no _client attribute → NoOp publisher
            pass

        event_stream_publisher = _Pub()

    ok, detail = _check_redis(Container())
    assert ok is True and "NoOp" in detail


@pytest.mark.unit
def test_check_redis_pings_real_client() -> None:
    class GoodClient:
        def ping(self):
            return True

    class BadClient:
        def ping(self):
            raise RuntimeError("redis gone")

    class C:
        def __init__(self, client):
            self.event_stream_publisher = type("P", (), {"_client": client})()

    assert _check_redis(C(GoodClient())) == (True, None)
    assert _check_redis(C(BadClient()))[0] is False


@pytest.mark.unit
def test_json_formatter_renders_correlation_and_audit() -> None:
    rec = logging.LogRecord(
        "context_engine.operator_audit",
        logging.WARNING,
        "router.py",
        1,
        "operator_action pot.reset",
        None,
        None,
    )
    rec.audit = {"action": "pot.reset", "pot_id": "p1"}  # the dormant channel
    with correlation_scope(trace_id="abc123", event_id="e9", pot_id="p1"):
        CorrelationFilter().filter(rec)
        line = JsonFormatter().format(rec)
    doc = json.loads(line)
    assert doc["level"] == "WARNING"
    assert doc["logger"] == "context_engine.operator_audit"
    assert doc["trace_id"] == "abc123"
    assert doc["event_id"] == "e9"
    # the previously-dropped audit payload now renders
    assert doc["audit"] == {"action": "pot.reset", "pot_id": "p1"}


@pytest.mark.unit
def test_json_formatter_handles_nonserializable_extra() -> None:
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "msg", None, None)
    rec.weird = object()  # not JSON-serializable
    CorrelationFilter().filter(rec)
    doc = json.loads(JsonFormatter().format(rec))
    assert "weird" in doc  # repr fallback, did not raise
