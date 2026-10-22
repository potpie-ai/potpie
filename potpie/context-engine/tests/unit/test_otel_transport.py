"""Exporter transport selection must match the installed platform packages."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from potpie_context_engine.adapters.outbound.observability import otel


@pytest.fixture(autouse=True)
def clean_otel_configuration(monkeypatch):
    for name in os.environ:
        if name.startswith("OTEL_EXPORTER_OTLP"):
            monkeypatch.delenv(name)


@pytest.fixture(params=("TRACES", "METRICS"))
def signal(request):
    return request.param


def exporter(signal):
    return (
        otel._otlp_span_exporter()
        if signal == "TRACES"
        else otel._otlp_metric_exporter()
    )


def windows_arm(monkeypatch):
    # Scope platform simulation to the adapter, so native SDK dependencies
    # still import against the real host that executes these unit tests.
    monkeypatch.setattr(otel, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(otel, "platform", SimpleNamespace(machine=lambda: "ARM64"))


def missing_grpc(monkeypatch, signal):
    kind = "trace" if signal == "TRACES" else "metric"
    monkeypatch.setitem(
        sys.modules, f"opentelemetry.exporter.otlp.proto.grpc.{kind}_exporter", None
    )


def test_windows_arm_defaults_to_http_port_4318_without_grpc(monkeypatch, signal):
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    windows_arm(monkeypatch)
    missing_grpc(monkeypatch, signal)
    instance = exporter(signal)
    try:
        assert ".http." in type(instance).__module__
        assert instance._endpoint == f"http://localhost:4318/v1/{signal.lower()}"
    finally:
        instance.shutdown()


def test_other_platforms_keep_grpc_port_4317(monkeypatch, signal):
    pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc")
    monkeypatch.setattr(otel, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(otel, "platform", SimpleNamespace(machine=lambda: "aarch64"))
    instance = exporter(signal)
    try:
        assert ".grpc." in type(instance).__module__
        assert instance._endpoint == "localhost:4317"
    finally:
        instance.shutdown()


def test_http_protocol_honors_configured_base_endpoint(monkeypatch, signal):
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/collector/"
    )
    instance = exporter(signal)
    try:
        assert (
            instance._endpoint == f"http://localhost:4318/collector/v1/{signal.lower()}"
        )
    finally:
        instance.shutdown()


def test_signal_endpoint_and_protocol_take_precedence_without_url_rewriting(
    monkeypatch, signal
):
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    windows_arm(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    monkeypatch.setenv(f"OTEL_EXPORTER_OTLP_{signal}_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/ignored")
    # A collector may deliberately expose HTTP on any port, including 4317.
    endpoint = "http://localhost:4317/custom-signal-path"
    monkeypatch.setenv(f"OTEL_EXPORTER_OTLP_{signal}_ENDPOINT", endpoint)
    instance = exporter(signal)
    try:
        assert ".http." in type(instance).__module__
        assert instance._endpoint == endpoint
    finally:
        instance.shutdown()


def test_explicit_grpc_failure_is_not_silently_changed_to_http(
    monkeypatch, signal, caplog
):
    windows_arm(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    missing_grpc(monkeypatch, signal)
    with pytest.raises(RuntimeError, match="HTTP endpoint.*4318"):
        exporter(signal)
    assert "http/protobuf" in caplog.text


def test_existing_implicit_grpc_failure_still_falls_back(monkeypatch, signal):
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    monkeypatch.setattr(otel, "sys", SimpleNamespace(platform="linux"))
    missing_grpc(monkeypatch, signal)
    instance = exporter(signal)
    try:
        assert ".http." in type(instance).__module__
    finally:
        instance.shutdown()


def test_unsupported_protocol_is_diagnosed(monkeypatch, signal, caplog):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json")
    with pytest.raises(ValueError, match="unsupported OTLP protocol"):
        exporter(signal)
    assert "http/json" in caplog.text
