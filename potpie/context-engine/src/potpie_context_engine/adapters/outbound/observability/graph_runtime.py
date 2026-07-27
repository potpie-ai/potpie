"""Bridge graph-runtime lifecycle events into the engine observability port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from potpie_context_engine.domain.ports.observability import ObservabilityPort


@dataclass(frozen=True, slots=True)
class GraphRuntimeObserver:
    """Emit bounded-cardinality operation counts and latency histograms."""

    observability: ObservabilityPort

    def observe(self, event: str, fields: Mapping[str, Any]) -> None:
        try:
            phase = str(fields.get("phase", "completed"))
            attributes = {
                "operation": event.removeprefix("graph."),
                "phase": phase,
            }
            for name in ("ok", "status", "error_type"):
                value = fields.get(name)
                if value is not None:
                    attributes[name] = value
            if phase in {"completed", "failed"}:
                self.observability.counter(
                    "ce.graph.runtime.operations_total",
                    attributes=attributes,
                )
            duration_ms = fields.get("duration_ms")
            if duration_ms is not None:
                self.observability.histogram(
                    "ce.graph.runtime.duration_ms",
                    float(duration_ms),
                    attributes=attributes,
                )
        except Exception:
            # Observability is an auxiliary path and must never affect graph state.
            return


__all__ = ["GraphRuntimeObserver"]
