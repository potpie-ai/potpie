"""logfire log sink (extra: observability[logfire]) — NEW.

logfire does double duty: tracing (tracing.configure_tracing) AND structured
log export (this sink). Both share a single logfire.configure() call —
setup() here delegates to tracing.configure_tracing so logfire is configured
ONCE per process (EC2).

Edge cases:
 - Import logfire lazily.
 - No token / disabled -> setup() returns; build_handler returns None.
   tracing.configure_tracing already emits the one visible 'local-only' notice.
 - instrument() is no-op here: instrument_litellm/pydantic_ai already happen
   inside configure_tracing, so they aren't repeated.
"""

from __future__ import annotations

import logging

from ..config import ObservabilityConfig
from ..tracing import configure_tracing

_HINT = "logfire sink requires logfire — install observability[logfire]"


class LogfireSink:
    name = "logfire"

    def setup(self, config: ObservabilityConfig) -> None:
        if not config.logfire.enabled and not config.logfire.token:
            return
        try:
            import logfire  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(_HINT) from exc
        configure_tracing(config)

    def build_handler(
        self, config: ObservabilityConfig
    ) -> logging.Handler | None:
        if not config.logfire.enabled and not config.logfire.token:
            return None
        try:
            import logfire
        except ModuleNotFoundError:
            return None
        Handler = getattr(logfire, "LogfireLoggingHandler", None)
        if Handler is None:  # pragma: no cover — older logfire fallback
            class _MinimalHandler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    try:
                        import logfire as lf
                        fields: dict = {}
                        for attr in ("obs_context", "obs_fields"):
                            data = getattr(record, attr, None)
                            if isinstance(data, dict):
                                fields.update(data)
                        lf.log(
                            record.levelname.lower(),
                            record.getMessage(),
                            **fields,
                        )
                    except Exception:
                        self.handleError(record)

            return _MinimalHandler()
        return Handler()

    def instrument(self, config: ObservabilityConfig) -> None:
        return None

    def shutdown(self, config: ObservabilityConfig) -> None:
        # logfire batches spans/logs; flush before the process / sink dies.
        # API surface varies across versions; try the common spellings.
        try:
            import logfire
        except Exception:
            return
        for name in ("force_flush", "shutdown"):
            fn = getattr(logfire, name, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    continue
