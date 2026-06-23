"""The three ports of the hexagonal daemon shell: Transport, Component, ServiceBackend."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from potpie.context_engine.domain.ports.daemon.operations import OperationRegistry
from potpie.context_engine.domain.ports.daemon.service import ReadyProbe, RestartPolicy, ServiceSpec


class HealthStatus(Enum):
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPED = "stopped"


@runtime_checkable
class Transport(Protocol):
    def bind(self, ctx: "ShellContext") -> None: ...  # type: ignore[name-defined]  # noqa: F821
    async def serve(self, ops: OperationRegistry) -> None: ...
    async def stop(self) -> None: ...
    def health(self) -> HealthStatus: ...


@runtime_checkable
class Component(Protocol):
    name: str

    async def on_start(self, ctx: "ShellContext") -> None: ...  # type: ignore[name-defined]  # noqa: F821
    async def on_stop(self) -> None: ...
    def health(self) -> HealthStatus: ...
    def operations(self) -> list: ...


@runtime_checkable
class ServiceBackend(Protocol):
    async def start(self, spec: ServiceSpec, ctx: "ShellContext") -> None: ...  # type: ignore[name-defined]  # noqa: F821
    async def stop(self, spec: ServiceSpec) -> None: ...
    async def probe(self, spec: ServiceSpec) -> HealthStatus: ...


__all__ = [
    "HealthStatus",
    "Transport",
    "Component",
    "ServiceBackend",
    "ServiceSpec",
    "ReadyProbe",
    "RestartPolicy",
]
