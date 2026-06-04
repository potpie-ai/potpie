"""Observability port: tracing spans + metric primitives.

A thin, backend-neutral seam. The domain and application layers emit spans
and metrics through this port and never import ``opentelemetry`` directly —
that import lives only inside the OTel adapter, which is what makes the
backend genuinely swappable (OTLP → Tempo/Prometheus today, anything else
tomorrow).

The default :class:`NoOpObservability` makes every call site safe to invoke
unconditionally, exactly like :class:`domain.ports.telemetry.NoOpTelemetry`.
All calls must be cheap and must never raise into the caller — observability
never fails a request.

Span ``kind`` is a small string set (mapped to the OTel enum inside the
adapter) so the domain stays dependency-free:

- ``"internal"`` — in-process unit of work (default)
- ``"server"`` — handling an inbound request
- ``"client"`` — outbound call to a dependency
- ``"producer"`` / ``"consumer"`` — async queue enqueue / dequeue
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

SPAN_KIND_INTERNAL = "internal"
SPAN_KIND_SERVER = "server"
SPAN_KIND_CLIENT = "client"
SPAN_KIND_PRODUCER = "producer"
SPAN_KIND_CONSUMER = "consumer"


@runtime_checkable
class Span(Protocol):
    """The handle yielded by :meth:`ObservabilityPort.span`.

    Every method is side-effect-only and must swallow its own failures.
    """

    def set_attribute(self, key: str, value: Any) -> None: ...

    def set_attributes(self, attributes: Mapping[str, Any]) -> None: ...

    def add_event(
        self, name: str, attributes: Mapping[str, Any] | None = None
    ) -> None: ...

    def record_exception(self, exc: BaseException) -> None: ...

    def set_error(self, message: str | None = None) -> None: ...


class ObservabilityPort(Protocol):
    """Tracing + metrics sink. Calls must be cheap and never raise."""

    def span(
        self,
        name: str,
        *,
        kind: str = SPAN_KIND_INTERNAL,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[str] | None = None,
    ) -> Any:
        """Context manager that opens a span and yields a :class:`Span`.

        ``links`` is a sequence of W3C ``traceparent`` strings — used by the
        batch trace to link back to each event's (long-gone) ingress trace.
        The contextmanager must yield even when tracing is disabled so call
        sites are unconditional.
        """
        ...

    def current_traceparent(self) -> str | None:
        """W3C ``traceparent`` of the active span, or ``None``.

        Persisted into ``context_events.correlation_id`` at admission so the
        async batch run can link back across the windowed delay.
        """
        ...

    def baggage(self, **items: Any) -> Any:
        """Context manager attaching OTel baggage for the block.

        Used around the agent run so the *child* spans created by
        pydantic-ai's own instrumentation inherit the pot / batch / run /
        event ids and can be traced back to the agent run. Must yield even
        when tracing is disabled.
        """
        ...

    def counter(
        self, name: str, value: int = 1, *, attributes: Mapping[str, Any] | None = None
    ) -> None: ...

    def histogram(
        self,
        name: str,
        value: float,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None: ...

    def gauge(
        self,
        name: str,
        value: float,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None: ...


class _NoopSpan:
    """A span handle that does nothing. Shared singleton-friendly."""

    __slots__ = ()

    def set_attribute(self, key: str, value: Any) -> None:
        del key, value

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        del attributes

    def add_event(
        self, name: str, attributes: Mapping[str, Any] | None = None
    ) -> None:
        del name, attributes

    def record_exception(self, exc: BaseException) -> None:
        del exc

    def set_error(self, message: str | None = None) -> None:
        del message


_NOOP_SPAN = _NoopSpan()


class NoOpObservability:
    """Default implementation: discard everything. Test- and standalone-safe.

    Crucially, :meth:`span` still yields a usable (no-op) span so call sites
    never branch on whether observability is wired.
    """

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = SPAN_KIND_INTERNAL,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[str] | None = None,
    ) -> Iterator[_NoopSpan]:
        del name, kind, attributes, links
        yield _NOOP_SPAN

    def current_traceparent(self) -> str | None:
        return None

    @contextmanager
    def baggage(self, **items: Any) -> Iterator[None]:
        del items
        yield

    def counter(
        self, name: str, value: int = 1, *, attributes: Mapping[str, Any] | None = None
    ) -> None:
        del name, value, attributes

    def histogram(
        self,
        name: str,
        value: float,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        del name, value, attributes

    def gauge(
        self,
        name: str,
        value: float,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        del name, value, attributes
