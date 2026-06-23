"""Bridge business telemetry (cost + drift) into OTel metrics.

``TelemetryPort`` stays its own port with its own Postgres tables and
product surface (``meta.cost`` / ``quality.drift``). This decorator wraps
the configured sink and *also* mirrors every cost event / drift snapshot
into OTel metrics — so the same numbers land in Prometheus/Grafana without
touching any of the ~3 ``record_cost`` / ``record_drift`` call sites.
"Bridge, not merge": the inner sink's behavior is preserved exactly;
mirroring is best-effort and never raises.
"""

from __future__ import annotations

from potpie.context_engine.domain.ports.observability import ObservabilityPort
from potpie.context_engine.domain.ports.telemetry import CostEvent, DriftSnapshot, TelemetryPort


class ObservabilityTelemetryBridge:
    """Wrap a :class:`TelemetryPort`, mirroring into an OTel sink."""

    def __init__(self, inner: TelemetryPort, obs: ObservabilityPort) -> None:
        self._inner = inner
        self._obs = obs

    def record_cost(self, event: CostEvent) -> None:
        self._inner.record_cost(event)
        try:
            attrs = {
                "kind": event.kind,
                "model": event.model or "unknown",
                "pot_id": event.pot_id,
            }
            self._obs.counter("ce.llm.calls_total", 1, attributes=attrs)
            if event.total_tokens is not None:
                self._obs.counter(
                    "ce.llm.tokens_total", event.total_tokens, attributes=attrs
                )
            if event.input_tokens is not None:
                self._obs.counter(
                    "ce.llm.input_tokens_total",
                    event.input_tokens,
                    attributes=attrs,
                )
            if event.output_tokens is not None:
                self._obs.counter(
                    "ce.llm.output_tokens_total",
                    event.output_tokens,
                    attributes=attrs,
                )
            if event.latency_ms is not None:
                self._obs.histogram(
                    "ce.llm.latency_ms",
                    float(event.latency_ms),
                    attributes=attrs,
                )
        except Exception:  # noqa: BLE001 — telemetry never fails a request
            pass

    def record_drift(self, snapshot: DriftSnapshot) -> None:
        self._inner.record_drift(snapshot)
        try:
            attrs = {"pot_id": snapshot.pot_id, "status": snapshot.status}
            self._obs.gauge(
                "ce.drift.stale_refs", snapshot.stale_ref_count, attributes=attrs
            )
            self._obs.gauge(
                "ce.drift.verification_failed_refs",
                snapshot.verification_failed_ref_count,
                attributes=attrs,
            )
            self._obs.gauge(
                "ce.drift.source_access_gaps",
                snapshot.source_access_gap_count,
                attributes=attrs,
            )
            self._obs.gauge(
                "ce.drift.open_conflicts",
                snapshot.open_conflicts_count,
                attributes=attrs,
            )
            self._obs.gauge(
                "ce.drift.missing_coverage",
                snapshot.missing_coverage_count,
                attributes=attrs,
            )
        except Exception:  # noqa: BLE001
            pass

    # Forward anything else (future TelemetryPort methods) untouched.
    def __getattr__(self, name: str):
        return getattr(self._inner, name)
