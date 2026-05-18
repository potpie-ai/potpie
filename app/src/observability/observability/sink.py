"""The ONE abstraction seam: the `Sink` Protocol + a small registry.

A Sink is anything that can receive logs and/or initialise a backend. Three
methods so loguru, Sentry, and logfire all fit one contract:

  setup(config)          init the backend SDK (sentry_sdk.init / logfire.
                          configure). MUST be idempotent and fork-safe (EC2).
  build_handler(config)  return a logging.Handler, or None for sinks that
                          only init/instrument (e.g. a tracing-only logfire).
  instrument(config)     optional tracing/instrumentation hooks (logfire
                          litellm/pydantic-ai). No-op for plain log sinks.

Why a Protocol and not a base class: keeps sinks dependency-free and lets a
host register its own without importing this package's internals.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from .config import ObservabilityConfig


@runtime_checkable
class Sink(Protocol):
    name: str

    def setup(self, config: ObservabilityConfig) -> None: ...

    def build_handler(
        self, config: ObservabilityConfig
    ) -> logging.Handler | None: ...

    def instrument(self, config: ObservabilityConfig) -> None: ...


class SinkRegistry:
    """Name -> Sink factory. Lets profiles/config refer to sinks by string
    and lets a host plug a custom sink. STUB (Phase 1): contract only."""

    def register(self, name: str, factory) -> None:
        raise NotImplementedError("Phase 1 scaffold — implemented in Phase 2")

    def resolve(self, name: str) -> Sink:
        raise NotImplementedError("Phase 1 scaffold — implemented in Phase 2")


registry = SinkRegistry()
