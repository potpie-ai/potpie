"""OpenTelemetry observability adapter — OTLP → collector (Tempo/Prometheus).

This is the *only* module in the codebase that imports ``opentelemetry``.
Everything else depends on :class:`domain.ports.observability.ObservabilityPort`,
which is what keeps the backend swappable.

Provider-aware (the critical integration detail):

* **Parent provider present** (e.g. running inside the Potpie Celery worker,
  where Logfire already installed an SDK ``TracerProvider`` that instruments
  pydantic-ai): we *attach* an extra OTLP ``BatchSpanProcessor`` to the
  existing provider and never call ``set_tracer_provider``. Our spans ride
  the same provider as the agent's, so model/tool spans nest under our
  event spans automatically.
* **No parent provider** (standalone HTTP / CLI / MCP): we own a fresh
  ``TracerProvider`` and register it.

Configuration is entirely through standard ``OTEL_*`` env (the SDK reads
``OTEL_EXPORTER_OTLP_ENDPOINT`` etc. itself), so the backend can change
without touching code. If the ``observability`` extra is not installed, or
any setup step fails, the adapter degrades to a silent no-op — observability
never fails the process.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

from bootstrap.observability_context import get_correlation
from domain.ports.observability import (
    SPAN_KIND_CLIENT,
    SPAN_KIND_CONSUMER,
    SPAN_KIND_INTERNAL,
    SPAN_KIND_PRODUCER,
    SPAN_KIND_SERVER,
    _NOOP_SPAN,
)

logger = logging.getLogger(__name__)

# Guard so repeated OtelObservability() never double-attaches a processor to
# the same provider (build_ingestion_server can run more than once per process).
_ATTACHED_PROVIDERS: set[int] = set()
_SETUP_LOCK = threading.Lock()
_OPENAI_INSTRUMENTED = False


def _instrument_openai_sdk() -> None:
    """Auto-instrument the OpenAI SDK once per process.

    Captures OpenAI SDK calls made by downstream libraries (LLM-backed
    directly for entity/edge extraction + embeddings. Auto-instrumentation
    turns those into spans that nest under our ``graph.add_episode`` span
    (and inherit its pot/event baggage), so the most expensive LLM step is
    finally visible with token usage.
    """
    global _OPENAI_INSTRUMENTED
    if _OPENAI_INSTRUMENTED:
        return
    try:
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument()
        _OPENAI_INSTRUMENTED = True
        logger.info("observability: OpenAI SDK instrumented")
    except Exception as exc:  # noqa: BLE001 — extra may be absent
        logger.debug("observability: OpenAI instrumentation skipped: %r", exc)


def _otlp_span_exporter():
    """Best-effort OTLP span exporter: gRPC first, HTTP fallback."""
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter()
    except Exception:  # noqa: BLE001 — grpc variant may be absent
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HttpSpanExporter,
        )

        return HttpSpanExporter()


def _otlp_metric_exporter():
    try:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )

        return OTLPMetricExporter()
    except Exception:  # noqa: BLE001
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter as HttpMetricExporter,
        )

        return HttpMetricExporter()


class OtelObservability:
    """OTel-backed observability. Construct only when an OTLP endpoint is set.

    Raises on import failure (no SDK) so ``_default_observability`` can fall
    back to NoOp; once constructed, no method ever raises into the caller.
    """

    def __init__(self) -> None:
        # Imported here so the module only hard-requires opentelemetry when
        # this adapter is actually selected.
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        self._trace = trace
        self._propagator = TraceContextTextMapPropagator()
        self._kind_map = {
            SPAN_KIND_INTERNAL: trace.SpanKind.INTERNAL,
            SPAN_KIND_SERVER: trace.SpanKind.SERVER,
            SPAN_KIND_CLIENT: trace.SpanKind.CLIENT,
            SPAN_KIND_PRODUCER: trace.SpanKind.PRODUCER,
            SPAN_KIND_CONSUMER: trace.SpanKind.CONSUMER,
        }
        service_name = os.getenv("OTEL_SERVICE_NAME", "context-engine")

        with _SETUP_LOCK:
            provider = trace.get_tracer_provider()
            if isinstance(provider, SdkTracerProvider):
                # Parent provider (Logfire in the Celery worker). Attach an
                # extra OTLP processor pointing at our collector; ride along.
                pid = id(provider)
                if pid not in _ATTACHED_PROVIDERS:
                    try:
                        provider.add_span_processor(
                            BatchSpanProcessor(_otlp_span_exporter())
                        )
                        _ATTACHED_PROVIDERS.add(pid)
                        logger.info(
                            "observability: attached OTLP exporter to existing "
                            "TracerProvider (parent-provider mode)"
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "observability: could not attach to parent "
                            "provider: %r",
                            exc,
                        )
            else:
                # Standalone — we own the provider.
                own = SdkTracerProvider(
                    resource=Resource.create({"service.name": service_name})
                )
                own.add_span_processor(BatchSpanProcessor(_otlp_span_exporter()))
                trace.set_tracer_provider(own)
                _ATTACHED_PROVIDERS.add(id(own))
                logger.info(
                    "observability: installed own TracerProvider "
                    "(standalone mode, service=%s)",
                    service_name,
                )

            self._meter = self._setup_meter(
                metrics, Resource, service_name
            )

        self._tracer = trace.get_tracer("context-engine")
        self._instruments: dict[tuple[str, str], Any] = {}
        self._inst_lock = threading.Lock()
        _instrument_openai_sdk()

    # ---- setup helpers -------------------------------------------------

    def _setup_meter(self, metrics, Resource, service_name: str):
        try:
            from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider
            from opentelemetry.sdk.metrics.export import (
                PeriodicExportingMetricReader,
            )

            mp = metrics.get_meter_provider()
            if not isinstance(mp, SdkMeterProvider):
                reader = PeriodicExportingMetricReader(_otlp_metric_exporter())
                mp = SdkMeterProvider(
                    resource=Resource.create({"service.name": service_name}),
                    metric_readers=[reader],
                )
                metrics.set_meter_provider(mp)
            return metrics.get_meter("context-engine")
        except Exception as exc:  # noqa: BLE001
            logger.warning("observability: meter setup failed: %r", exc)
            return None

    # ---- ObservabilityPort --------------------------------------------

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = SPAN_KIND_INTERNAL,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[str] | None = None,
    ) -> Iterator[Any]:
        try:
            otel_links = self._build_links(links)
            attrs = self._merge_attrs(attributes)
            cm = self._tracer.start_as_current_span(
                name,
                kind=self._kind_map.get(kind, self._trace.SpanKind.INTERNAL),
                attributes=attrs,
                links=otel_links,
                record_exception=True,
                set_status_on_exception=True,
            )
        except Exception as exc:  # noqa: BLE001 — never break the caller
            logger.debug("observability: span open failed for %s: %r", name, exc)
            yield _NOOP_SPAN
            return
        with cm as raw:
            yield _SpanWrapper(raw, self._trace)

    def current_traceparent(self) -> str | None:
        try:
            carrier: dict[str, str] = {}
            self._propagator.inject(carrier)
            return carrier.get("traceparent")
        except Exception:  # noqa: BLE001
            return None

    @contextmanager
    def baggage(self, **items: Any) -> Iterator[None]:
        from opentelemetry import baggage as _bag
        from opentelemetry import context as _ctx

        ctx = None
        try:
            cur = _ctx.get_current()
            for key, value in items.items():
                if value is not None:
                    cur = _bag.set_baggage(f"ce.{key}", str(value), context=cur)
            ctx = _ctx.attach(cur)
        except Exception:  # noqa: BLE001 — never break the agent run
            ctx = None
        try:
            yield
        finally:
            if ctx is not None:
                try:
                    _ctx.detach(ctx)
                except Exception:  # noqa: BLE001
                    pass

    def counter(
        self, name: str, value: int = 1, *, attributes: Mapping[str, Any] | None = None
    ) -> None:
        inst = self._instrument("counter", name)
        if inst is not None:
            try:
                inst.add(value, attributes=_metric_attrs(attributes))
            except Exception:  # noqa: BLE001
                pass

    def histogram(
        self,
        name: str,
        value: float,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        inst = self._instrument("histogram", name)
        if inst is not None:
            try:
                inst.record(value, attributes=_metric_attrs(attributes))
            except Exception:  # noqa: BLE001
                pass

    def gauge(
        self,
        name: str,
        value: float,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        inst = self._instrument("gauge", name)
        if inst is not None:
            try:
                inst.set(value, attributes=_metric_attrs(attributes))
            except Exception:  # noqa: BLE001
                pass

    # ---- internals -----------------------------------------------------

    def _instrument(self, kind: str, name: str):
        if self._meter is None:
            return None
        key = (kind, name)
        inst = self._instruments.get(key)
        if inst is not None:
            return inst
        with self._inst_lock:
            inst = self._instruments.get(key)
            if inst is not None:
                return inst
            try:
                if kind == "counter":
                    inst = self._meter.create_counter(name)
                elif kind == "histogram":
                    inst = self._meter.create_histogram(name)
                else:
                    inst = self._meter.create_gauge(name)
            except Exception as exc:  # noqa: BLE001 — older SDKs lack gauge
                logger.debug("observability: instrument %s failed: %r", name, exc)
                inst = None
            self._instruments[key] = inst
            return inst

    def _merge_attrs(
        self, attributes: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        # Stamp the active correlation ids onto every span so a span is
        # always tied back to its event/pot/batch even without baggage.
        attrs: dict[str, Any] = {}
        for k, v in get_correlation().items():
            attrs[f"ce.{k}"] = v
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    attrs[k] = v
        return attrs

    def _build_links(self, links: Sequence[str] | None):
        if not links:
            return None
        out = []
        for tp in links:
            try:
                ctx = self._propagator.extract({"traceparent": tp})
                sc = self._trace.get_current_span(ctx).get_span_context()
                if sc and sc.is_valid:
                    out.append(self._trace.Link(sc))
            except Exception:  # noqa: BLE001
                continue
        return out or None


class _SpanWrapper:
    """Adapts an OTel span to the domain :class:`Span` protocol."""

    __slots__ = ("_s", "_trace")

    def __init__(self, span: Any, trace_mod: Any) -> None:
        self._s = span
        self._trace = trace_mod

    def set_attribute(self, key: str, value: Any) -> None:
        try:
            if value is not None:
                self._s.set_attribute(key, value)
        except Exception:  # noqa: BLE001
            pass

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        for k, v in attributes.items():
            self.set_attribute(k, v)

    def add_event(
        self, name: str, attributes: Mapping[str, Any] | None = None
    ) -> None:
        try:
            self._s.add_event(name, attributes=dict(attributes or {}))
        except Exception:  # noqa: BLE001
            pass

    def record_exception(self, exc: BaseException) -> None:
        try:
            self._s.record_exception(exc)
        except Exception:  # noqa: BLE001
            pass

    def set_error(self, message: str | None = None) -> None:
        try:
            self._s.set_status(
                self._trace.Status(self._trace.StatusCode.ERROR, message)
            )
        except Exception:  # noqa: BLE001
            pass


def _metric_attrs(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    corr = get_correlation()
    # Only low-cardinality correlation keys belong on metric labels.
    for k in ("pot_id", "source"):
        if corr.get(k):
            out[k] = corr[k]
    if attributes:
        for k, v in attributes.items():
            if v is not None:
                out[k] = v
    return out
