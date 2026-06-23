"""Composition-root helpers for the process observability sink."""

from __future__ import annotations

from context_engine.domain.ports.observability import NoOpObservability, ObservabilityPort


def observability_enabled() -> bool:
    import os

    raw = os.getenv("CONTEXT_ENGINE_OBSERVABILITY", "").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def default_observability() -> ObservabilityPort:
    """Build the env-selected observability sink.

    Defaults to ``NoOpObservability`` so local CLI/daemon usage stays dark
    unless operators explicitly opt in.
    """
    import os

    if not observability_enabled():
        return NoOpObservability()
    mode = os.getenv("CONTEXT_ENGINE_OBSERVABILITY", "").strip().lower()
    if mode == "console":
        try:
            from context_engine.adapters.outbound.observability.console import ConsoleObservability

            return ConsoleObservability()
        except Exception:  # noqa: BLE001
            return NoOpObservability()
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    )
    if not endpoint:
        return NoOpObservability()
    try:
        from context_engine.adapters.outbound.observability.otel import OtelObservability

        return OtelObservability()
    except Exception:  # noqa: BLE001 - missing extra / setup failure -> dark
        return NoOpObservability()


__all__ = ["default_observability", "observability_enabled"]
