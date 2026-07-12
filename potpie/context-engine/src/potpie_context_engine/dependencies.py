"""Injectable capabilities used by the engine composition root."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from potpie_context_engine.domain.ports.graph.backend import GraphBackend
from potpie_context_engine.domain.ports.ledger.client import EventLedgerClientPort
from potpie_context_engine.domain.ports.observability import ObservabilityPort

HttpApplicationFactory = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class EngineDependencies:
    """Optional adapters supplied by an embedding application.

    Product settings, credentials, UI, skills, process lifecycle, and product
    telemetry are intentionally absent from this type.
    """

    backend: GraphBackend | None = None
    ledger_client: EventLedgerClientPort | None = None
    observability: ObservabilityPort | None = None
    http_application_factory: HttpApplicationFactory | None = None


__all__ = ["EngineDependencies", "HttpApplicationFactory"]
