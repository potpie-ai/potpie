"""Root-owned generic observability seam for product command orchestration."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping

SPAN_KIND_INTERNAL = "internal"


class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        del key, value

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        del attributes

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        del name, attributes

    def record_exception(self, exc: BaseException) -> None:
        del exc

    def set_error(self, message: str | None = None) -> None:
        del message


class ProductObservability:
    @contextmanager
    def span(self, *_args: Any, **_kwargs: Any) -> Iterator[_NoOpSpan]:
        yield _NoOpSpan()

    def counter(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def histogram(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def gauge(self, *_args: Any, **_kwargs: Any) -> None:
        return None


_OBSERVABILITY = ProductObservability()


def get_observability() -> ProductObservability:
    return _OBSERVABILITY


__all__ = ["SPAN_KIND_INTERNAL", "ProductObservability", "get_observability"]
