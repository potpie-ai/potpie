"""The three ports of the hexagonal daemon shell: Transport, Component, ServiceBackend."""

from __future__ import annotations

from asyncio import Event
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from domain.ports.daemon.operations import OperationRegistry, OperationSpec
from domain.ports.daemon.service import ReadyProbe, RestartPolicy, ServiceSpec


class HealthStatus(Enum):
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPED = "stopped"


class ShellEndpoints(Protocol):
    """Resolved managed-service endpoints exposed to daemon shell participants."""

    def set(self, name: str, endpoint: str) -> None: ...
    def get(self, name: str) -> str: ...
    def remove(self, name: str) -> None: ...
    def resolve(self, value: str) -> str: ...


class ShellContext(Protocol):
    """Context-engine-owned shape of the runtime context passed into shell ports."""

    config: dict[str, Any]
    data_dir: Path
    logger: Logger
    endpoints: ShellEndpoints
    shutdown: Event


@runtime_checkable
class Transport(Protocol):
    def bind(self, ctx: ShellContext) -> None: ...
    async def serve(self, ops: OperationRegistry) -> None: ...
    async def stop(self) -> None: ...
    def health(self) -> HealthStatus: ...


@runtime_checkable
class Component(Protocol):
    name: str

    async def on_start(self, ctx: ShellContext) -> None: ...
    async def on_stop(self) -> None: ...
    def health(self) -> HealthStatus: ...
    def operations(self) -> list[OperationSpec]: ...


@runtime_checkable
class ServiceBackend(Protocol):
    async def start(self, spec: ServiceSpec, ctx: ShellContext) -> None: ...
    async def stop(self, spec: ServiceSpec) -> None: ...
    async def probe(self, spec: ServiceSpec) -> HealthStatus: ...


__all__ = [
    "HealthStatus",
    "ShellContext",
    "ShellEndpoints",
    "Transport",
    "Component",
    "ServiceBackend",
    "ServiceSpec",
    "ReadyProbe",
    "RestartPolicy",
]
