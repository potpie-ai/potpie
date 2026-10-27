"""Recording telemetry boundaries for graph CLI tests."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from potpie.cli.telemetry import product_analytics
from potpie.cli.telemetry.context import TelemetryContext


class _RecordedSpan:
    def __init__(self, attrs: dict) -> None:
        self.attrs = attrs
        self.error = None

    def set_attribute(self, key, value):
        self.attrs[key] = value

    def set_attributes(self, attributes):
        self.attrs.update(dict(attributes))

    def add_event(self, *_args, **_kwargs):
        pass

    def record_exception(self, exc):
        self.attrs["exception"] = repr(exc)

    def set_error(self, message=None):
        self.error = message or True


class _RecordingObservability:
    def __init__(self) -> None:
        self.spans: list[tuple[str, dict, _RecordedSpan]] = []
        self.counters: list[tuple[str, int, dict]] = []
        self.histograms: list[tuple[str, float, dict]] = []

    @contextmanager
    def span(self, name, *, kind="internal", attributes=None, links=None):
        del links
        attrs = {"kind": kind, **dict(attributes or {})}
        span = _RecordedSpan(attrs)
        self.spans.append((name, attrs, span))
        yield span

    def current_traceparent(self):
        return None

    @contextmanager
    def baggage(self, **_items):
        yield

    def counter(self, name, value=1, *, attributes=None):
        self.counters.append((name, value, dict(attributes or {})))

    def histogram(self, name, value, *, attributes=None):
        self.histograms.append((name, value, dict(attributes or {})))

    def gauge(self, name, value, *, attributes=None):
        del name, value, attributes


class _ProductAnalyticsSink:
    def __init__(self) -> None:
        self.events = []

    def capture(self, event) -> None:
        self.events.append(event)


def _bind_graph_product_analytics(
    monkeypatch: pytest.MonkeyPatch,
) -> _ProductAnalyticsSink:
    sink = _ProductAnalyticsSink()
    monkeypatch.setattr(product_analytics, "_sink", sink)
    monkeypatch.setattr(
        product_analytics,
        "current_telemetry_context",
        lambda: TelemetryContext(
            anonymous_install_id="install_graph",
            invocation_id="invoke_graph",
            daemon_session_id="daemon_graph",
            environment="test",
            command="graph",
            subcommand=None,
            output_mode="json",
            cli_version="0.1.0",
            python_version="3.13.0",
            os="darwin",
            arch="arm64",
        ),
    )
    return sink
