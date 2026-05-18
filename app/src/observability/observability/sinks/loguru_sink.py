"""loguru as an OPTIONAL sink (extra: observability[loguru]).

Filename is loguru_sink.py, NOT loguru.py — a module named loguru.py would
shadow the real `loguru` package (same reason the package is 'observability',
not 'logging').

loguru is imported lazily inside methods: importing this package never
requires loguru. If a host configures the 'loguru' sink without loguru
installed, resolution fails with a clear, actionable error (not at import).
"""

from __future__ import annotations

import logging

from ..config import ObservabilityConfig

_HINT = "loguru sink requires loguru — install observability[loguru]"


class _LoguruHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            from loguru import logger as L

            try:
                level = L.level(record.levelname).name
            except ValueError:
                level = record.levelno
            fields: dict = {}
            for attr in ("obs_context", "obs_fields"):
                data = getattr(record, attr, None)
                if isinstance(data, dict):
                    fields.update(data)
            L.bind(**fields).opt(depth=6, exception=record.exc_info).log(
                level, record.getMessage()
            )
        except Exception:
            self.handleError(record)


class LoguruSink:
    name = "loguru"

    def setup(self, config: ObservabilityConfig) -> None:
        try:
            import loguru  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(_HINT) from exc

    def build_handler(self, config: ObservabilityConfig) -> logging.Handler | None:
        return _LoguruHandler()

    def instrument(self, config: ObservabilityConfig) -> None:
        return None
