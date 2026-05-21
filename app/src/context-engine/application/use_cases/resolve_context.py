"""Application entry point for context intelligence resolution."""

from __future__ import annotations

import time

from application.services.context_resolution import ContextResolutionService
from bootstrap.observability_context import correlation_scope
from bootstrap.observability_runtime import get_observability
from domain.intelligence_models import ContextResolutionRequest, IntelligenceBundle
from domain.ports.observability import SPAN_KIND_SERVER


async def resolve_context(
    service: ContextResolutionService,
    request: ContextResolutionRequest,
) -> IntelligenceBundle:
    """Resolve contextual evidence for a query within a pot.

    The read-side trace root: one ``context.resolve`` span per request,
    with the pot bound to the correlation context so per-reader spans and
    log lines underneath carry it.
    """
    obs = get_observability()
    attrs = {
        "pot_id": request.pot_id,
        "resolve.intent": request.intent,
        "resolve.mode": request.mode,
        "resolve.source_policy": request.source_policy,
        "resolve.include_count": len(request.include),
    }
    with correlation_scope(pot_id=request.pot_id):
        with obs.span(
            "context.resolve", kind=SPAN_KIND_SERVER, attributes=attrs
        ) as span:
            t0 = time.perf_counter()
            try:
                bundle = await service.resolve(request)
            except Exception as exc:  # noqa: BLE001 — annotate + re-raise
                span.record_exception(exc)
                span.set_error(repr(exc))
                obs.counter(
                    "ce.resolve.total",
                    1,
                    attributes={"pot_id": request.pot_id, "result": "error"},
                )
                raise
            dur_ms = (time.perf_counter() - t0) * 1000.0
            obs.histogram(
                "ce.resolve.latency_ms",
                dur_ms,
                attributes={"pot_id": request.pot_id},
            )
            obs.counter(
                "ce.resolve.total",
                1,
                attributes={"pot_id": request.pot_id, "result": "ok"},
            )
            span.set_attribute("resolve.latency_ms", round(dur_ms, 1))
            return bundle
