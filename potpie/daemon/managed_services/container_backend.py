"""container ServiceBackend — runs services via the local docker CLI (no docker-py dep)."""

from __future__ import annotations

import asyncio
import shlex

from adapters.outbound.managed_services.subprocess_backend import (
    _argv_probe,
    _http_probe,
    _tcp_probe,
)
from domain.ports.daemon.service import ServiceSpec
from domain.ports.daemon.shell import HealthStatus
from host.daemon_runtime.context import ShellContext


class ContainerBackend:
    name = "container"

    def __init__(self, docker_cmd: str = "docker") -> None:
        self._docker = docker_cmd
        self._containers: dict[str, str] = {}  # service name -> container id

    async def start(self, spec: ServiceSpec, ctx: ShellContext) -> None:
        cfg = spec.config
        image = cfg["image"]
        argv = [self._docker, "run", "-d", "--rm", "--name", f"potpie_{spec.name}"]
        for host, container in (cfg.get("ports") or {}).items():
            argv += ["-p", f"{host}:{container}"]
        for k, v in (cfg.get("env") or {}).items():
            argv += ["-e", f"{k}={v}"]
        for src, dst in (cfg.get("volumes") or {}).items():
            argv += ["-v", f"{src}:{dst}"]
        argv += [image]
        if cfg.get("command"):
            argv += list(cfg["command"])
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"docker run failed: {err.decode().strip() or out.decode().strip()}"
            )
        self._containers[spec.name] = out.decode().strip()

    async def stop(self, spec: ServiceSpec) -> None:
        cid = self._containers.pop(spec.name, None)
        if cid is None:
            return
        for verb in ("stop", "rm"):
            proc = await asyncio.create_subprocess_exec(
                self._docker,
                verb,
                cid,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()  # ignore errors on stop/rm idempotency

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
            cid = self._containers.get(spec.name)
            if cid is None:
                return HealthStatus.STARTING
            argv = [self._docker, "exec", cid, *shlex.split(rp.target)]
            return (
                HealthStatus.READY
                if await _argv_probe(argv, rp.interval_s)
                else HealthStatus.STARTING
            )
        return HealthStatus.STARTING
