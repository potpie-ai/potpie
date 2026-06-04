"""Console observability adapter — spans + metrics to a logger.

Zero third-party dependencies. Intended for local development and tests:
set ``CONTEXT_ENGINE_OBSERVABILITY=console`` to see the trace skeleton and
metric emissions without standing up a collector. Honors the correlation
context so each line carries the active ids.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

from bootstrap.observability_context import get_correlation
from domain.ports.observability import SPAN_KIND_INTERNAL

logger = logging.getLogger("context_engine.observability")


class _ConsoleSpan:
    __slots__ = ("_name", "_attrs", "_id")

    def __init__(self, name: str, attrs: dict[str, Any]) -> None:
        self._name = name
        self._attrs = attrs
        self._id = uuid.uuid4().hex[:8]

    def set_attribute(self, key: str, value: Any) -> None:
        self._attrs[key] = value

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        self._attrs.update(attributes)

    def add_event(
        self, name: str, attributes: Mapping[str, Any] | None = None
    ) -> None:
        logger.info("span.event %s span=%s %s", name, self._id, dict(attributes or {}))

    def record_exception(self, exc: BaseException) -> None:
        logger.warning("span.exception span=%s %r", self._id, exc)

    def set_error(self, message: str | None = None) -> None:
        self._attrs["error"] = message or True


class ConsoleObservability:
    """Print-style observability. Never raises into the caller."""

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = SPAN_KIND_INTERNAL,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[str] | None = None,
    ) -> Iterator[_ConsoleSpan]:
        attrs = dict(attributes or {})
        sp = _ConsoleSpan(name, attrs)
        start = time.perf_counter()
        logger.info(
            "span.start %s kind=%s id=%s links=%d corr=%s",
            name,
            kind,
            sp._id,
            len(links or ()),
            get_correlation(),
        )
        try:
            yield sp
        except BaseException as exc:  # noqa: BLE001 — annotate then re-raise
            sp.set_error(repr(exc))
            raise
        finally:
            dur_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "span.end %s id=%s dur_ms=%.1f attrs=%s",
                name,
                sp._id,
                dur_ms,
                sp._attrs,
            )

    def current_traceparent(self) -> str | None:
        return None

    @contextmanager
    def baggage(self, **items: Any) -> Iterator[None]:
        logger.info("baggage.set %s", {k: v for k, v in items.items() if v is not None})
        yield

    def counter(
        self, name: str, value: int = 1, *, attributes: Mapping[str, Any] | None = None
    ) -> None:
        logger.info("metric.counter %s += %s %s", name, value, dict(attributes or {}))

    def histogram(
        self,
        name: str,
        value: float,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        logger.info("metric.histogram %s = %s %s", name, value, dict(attributes or {}))

    def gauge(
        self,
        name: str,
        value: float,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        logger.info("metric.gauge %s = %s %s", name, value, dict(attributes or {}))
