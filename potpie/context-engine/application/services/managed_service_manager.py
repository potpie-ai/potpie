"""ServiceManager — orchestrates managed services through ServiceBackend plugins."""

from __future__ import annotations
import asyncio
import contextlib
from dataclasses import dataclass
from host.daemon_runtime.context import ShellContext
from domain.ports.daemon.service import ServiceSpec
from domain.ports.daemon.shell import HealthStatus, ServiceBackend
from host.daemon_runtime.registry import Registry


class ServiceNotFound(KeyError):
    pass


class DependencyCycle(RuntimeError):
    pass


class ReadyTimeout(RuntimeError):
    pass


@dataclass
class ServiceStatus:
    name: str
    status: HealthStatus
    endpoint: str | None


class ServiceManager:
    def __init__(
        self,
        specs: dict[str, ServiceSpec],
        backends: Registry[ServiceBackend],
        ctx: ShellContext,
    ) -> None:
        self._specs = specs
        self._backends = backends
        self._ctx = ctx
        self._instances: dict[str, ServiceBackend] = {}
        self._lock = asyncio.Lock()
        self._statuses: dict[str, HealthStatus] = {}

    async def up(self, name: str) -> str:
        async with self._lock:
            await self._up_resolving([name], stack=set())
        return self._specs[name].endpoint

    async def _up_resolving(self, names: list[str], stack: set[str]) -> None:
        for name in names:
            if (
                name in self._instances
                and self._statuses.get(name) is HealthStatus.READY
            ):
                continue
            if name not in self._specs:
                raise ServiceNotFound(f"unknown service {name!r}")
            if name in stack:
                raise DependencyCycle(f"service dependency cycle through {name!r}")
            stack.add(name)
            spec = self._specs[name]
            await self._up_resolving(spec.depends_on, stack)
            stack.remove(name)
            existing = self._instances.get(name)
            if existing is not None:
                # A previous start left a non-READY backend (e.g. ReadyTimeout). Stop it
                # before recreating so we don't orphan a process/container.
                with contextlib.suppress(Exception):
                    await existing.stop(spec)
                self._instances.pop(name, None)
            backend = self._backends.create(spec.backend)
            await backend.start(spec, self._ctx)
            self._instances[name] = backend
            self._statuses[name] = HealthStatus.STARTING
            ok = await self._wait_ready(name, spec)
            if not ok:
                self._statuses[name] = HealthStatus.DEGRADED
                raise ReadyTimeout(f"service {name!r} did not become ready")
            self._statuses[name] = HealthStatus.READY
            self._ctx.endpoints.set(name, spec.endpoint)

    async def _wait_ready(self, name: str, spec: ServiceSpec) -> bool:
        backend = self._instances[name]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + spec.ready.timeout_s
        while loop.time() < deadline:
            s = await backend.probe(spec)
            if s is HealthStatus.READY:
                return True
            await asyncio.sleep(spec.ready.interval_s)
        return False

    async def down(self, name: str) -> None:
        # TODO(v2): stop services that depend on `name` first; V1 does not check dependents.
        async with self._lock:
            backend = self._instances.pop(name, None)
            if backend is None:
                return
            spec = self._specs[name]
            try:
                await backend.stop(spec)
            finally:
                self._statuses[name] = HealthStatus.STOPPED
                self._ctx.endpoints.remove(name)

    def started_names(self) -> list[str]:
        """Names of services that have a started backend instance (running or degraded)."""
        return list(self._instances)

    def status(self, name: str | None = None) -> ServiceStatus | list[ServiceStatus]:
        if name is not None:
            spec = self._specs.get(name)
            return ServiceStatus(
                name=name,
                status=self._statuses.get(name, HealthStatus.STOPPED),
                endpoint=spec.endpoint if spec else None,
            )
        return [
            ServiceStatus(
                name=n,
                status=self._statuses.get(n, HealthStatus.STOPPED),
                endpoint=s.endpoint,
            )
            for n, s in self._specs.items()
        ]
