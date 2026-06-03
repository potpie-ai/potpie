"""Process-wide accessor for the wired :class:`ObservabilityPort`.

The hexagonal DI container is the source of truth: ``build_ingestion_server``
installs the selected sink here. Application/domain call sites that already
receive a container keep using ``container.observability``. But three
cross-cutting concerns are *composition-root* concerns and cannot reach a
container instance:

* the inbound ASGI middleware (installed in ``create_app`` before any
  container exists),
* the Celery batch worker (a fresh process; the container is rebuilt there),
* the infra-adapter instrumentation wrapper.

For those, this module exposes the same instance the container holds —
exactly the pattern OpenTelemetry itself uses (``get_tracer_provider`` is
process-global). The default stays :class:`NoOpObservability`, so reading
before bootstrap is safe and the feature still ships dark.
"""

from __future__ import annotations

from domain.ports.observability import NoOpObservability, ObservabilityPort

_OBSERVABILITY: ObservabilityPort = NoOpObservability()


def set_observability(obs: ObservabilityPort) -> None:
    """Called by ``build_ingestion_server`` with the env-selected sink."""
    global _OBSERVABILITY
    _OBSERVABILITY = obs


def get_observability() -> ObservabilityPort:
    """Return the process observability sink (NoOp until bootstrap runs)."""
    return _OBSERVABILITY
