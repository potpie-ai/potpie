"""external ServiceBackend — does not start anything; only probes an already-running endpoint."""

from __future__ import annotations

from adapters.outbound.managed_services.subprocess_backend import (
    _cmd_probe,
    _http_probe,
    _tcp_probe,
)
from domain.ports.daemon.service import ServiceSpec
from domain.ports.daemon.shell import HealthStatus
from host.daemon_runtime.context import ShellContext


class ExternalBackend:
    name = "external"

    async def start(self, spec: ServiceSpec, ctx: ShellContext) -> None:
        return None

    async def stop(self, spec: ServiceSpec) -> None:
        return None

    async def probe(self, spec: ServiceSpec) -> HealthStatus:
        rp = spec.ready
        if rp.kind == "tcp":
            host, port = rp.target.split(":")
            return (
                HealthStatus.READY
                if await _tcp_probe(host, int(port), rp.interval_s)
                else HealthStatus.STARTING
            )
        if rp.kind == "http":
            return (
                HealthStatus.READY
                if await _http_probe(rp.target, rp.interval_s)
                else HealthStatus.STARTING
            )
        if rp.kind == "cmd":
            return (
                HealthStatus.READY
                if await _cmd_probe(rp.target, rp.interval_s)
                else HealthStatus.STARTING
            )
        return HealthStatus.STARTING
