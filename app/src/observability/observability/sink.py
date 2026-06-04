"""The ONE abstraction seam: the `Sink` Protocol + a lazy registry.

A Sink can receive logs and/or initialise a backend. Three methods so loguru,
Sentry and logfire all fit one contract:

  setup(config)          init the backend SDK (idempotent, fork-safe; EC2).
  build_handler(config)  return a logging.Handler, or None for sinks that only
                          init/instrument (e.g. tracing-only logfire).
  instrument(config)     optional tracing hooks (logfire). No-op for log sinks.

Built-ins are registered LAZILY: resolving a sink imports its module only
then, so `import observability` never pulls loguru/sentry_sdk/logfire.
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable, Protocol, runtime_checkable

from .config import ObservabilityConfig


@runtime_checkable
class Sink(Protocol):
    name: str

    def setup(self, config: ObservabilityConfig) -> None: ...

    def build_handler(
        self, config: ObservabilityConfig
    ) -> logging.Handler | None: ...

    def instrument(self, config: ObservabilityConfig) -> None: ...

    def shutdown(self, config: ObservabilityConfig) -> None:
        """Flush + release resources. Called on reconfigure and at process
        exit (atexit). Must be best-effort and never raise — backends may
        already be torn down during interpreter shutdown."""
        ...


# name -> "module:ClassName" (relative to this package), imported on resolve.
_BUILTIN: dict[str, str] = {
    "console": ".sinks.console:ConsoleSink",
    "json_stdout": ".sinks.json_stdout:JsonStdoutSink",
    "loguru": ".sinks.loguru_sink:LoguruSink",
    "sentry": ".sinks.sentry_sink:SentrySink",
    "logfire": ".sinks.logfire_sink:LogfireSink",
}


class SinkRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Sink]] = {}

    def register(self, name: str, factory: Callable[[], Sink]) -> None:
        self._factories[name] = factory

    def resolve(self, name: str) -> Sink:
        if name in self._factories:
            return self._factories[name]()
        spec = _BUILTIN.get(name)
        if spec is None:
            raise KeyError(
                f"Unknown sink {name!r}. Known: "
                f"{sorted(set(self._factories) | set(_BUILTIN))}"
            )
        mod_path, cls_name = spec.split(":")
        module = importlib.import_module(mod_path, __package__)
        return getattr(module, cls_name)()


registry = SinkRegistry()
