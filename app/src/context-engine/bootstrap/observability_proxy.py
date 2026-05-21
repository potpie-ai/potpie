"""Composition-root instrumentation proxy for infra adapters.

Hexagonal-correct placement: infra observability is a cross-cutting
concern wired where dependencies are wired (the container), never
sprinkled into the adapters themselves. The proxy forwards every
attribute untouched and wraps only public *callables* in a span +
latency histogram + error counter.

Scope note: applied to the Neo4j structural read adapter (all-sync
``get_*`` surface, zero ``isinstance`` coupling — safe to wrap). The
Graphiti episodic adapter is intentionally *not* proxied: a route does
``isinstance(container.episodic, GraphitiEpisodicAdapter)``, and its
hottest call (``add_episode``) already has a dedicated span. Only applied
when observability is live, so there is zero overhead in the NoOp default.
"""

from __future__ import annotations

import time
from typing import Any

from domain.ports.observability import SPAN_KIND_CLIENT, ObservabilityPort


class _InstrumentedAdapter:
    """Transparent proxy: forwards all attrs, instruments public methods."""

    __slots__ = ("_t", "_p", "_o", "_cache")

    def __init__(self, target: Any, prefix: str, obs: ObservabilityPort) -> None:
        object.__setattr__(self, "_t", target)
        object.__setattr__(self, "_p", prefix)
        object.__setattr__(self, "_o", obs)
        object.__setattr__(self, "_cache", {})

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, "_t")
        attr = getattr(target, name)
        if name.startswith("_") or not callable(attr):
            return attr
        cache = object.__getattribute__(self, "_cache")
        if name in cache:
            return cache[name]
        prefix = object.__getattribute__(self, "_p")
        obs = object.__getattribute__(self, "_o")

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            attrs = {"db.system": prefix, "db.op": name}
            try:
                with obs.span(
                    f"{prefix}.{name}",
                    kind=SPAN_KIND_CLIENT,
                    attributes=attrs,
                ) as span:
                    try:
                        return attr(*args, **kwargs)
                    except Exception as exc:  # noqa: BLE001 — annotate+reraise
                        span.set_error(repr(exc))
                        obs.counter(
                            f"ce.{prefix}.errors_total",
                            1,
                            attributes={"op": name},
                        )
                        raise
            finally:
                obs.histogram(
                    f"ce.{prefix}.query_ms",
                    (time.perf_counter() - t0) * 1000.0,
                    attributes={"op": name},
                )

        cache[name] = _wrapped
        return _wrapped


def instrument_adapter(
    target: Any, prefix: str, obs: ObservabilityPort
) -> Any:
    """Wrap ``target`` so its public methods emit spans/metrics."""
    return _InstrumentedAdapter(target, prefix, obs)
