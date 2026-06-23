"""subprocess ServiceBackend — spawn any local binary, supervise it, probe readiness."""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import signal
import subprocess

from potpie.context_engine.domain.ports.daemon.service import ServiceSpec
from potpie.context_engine.domain.ports.daemon.shell import HealthStatus
from potpie.context_engine.host.daemon_runtime.context import ShellContext


class SubprocessBackend:
    name = "subprocess"

    def __init__(self) -> None:
        self._procs: dict[str, asyncio.subprocess.Process] = {}

    async def start(self, spec: ServiceSpec, ctx: ShellContext) -> None:
        cmd = spec.config["command"]
        env = spec.config.get("env")
        cwd = spec.config.get("cwd")
        log_path = ctx.data_dir / "logs" / f"service-{spec.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log_fp:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                cwd=cwd,
                stdout=log_fp,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        self._procs[spec.name] = proc

    async def stop(self, spec: ServiceSpec) -> None:
        proc = self._procs.pop(spec.name, None)
        if proc is None or proc.returncode is not None:
            return
        proc.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()

    async def probe(self, spec: ServiceSpec) -> HealthStatus:
        proc = self._procs.get(spec.name)
        if proc is None:
            return HealthStatus.STOPPED
        if proc.returncode is not None:
            return HealthStatus.DEGRADED
        rp = spec.ready
        if rp.kind == "tcp":
            host, port = rp.target.split(":")
            ok = await _tcp_probe(host, int(port), timeout_s=rp.interval_s)
            return HealthStatus.READY if ok else HealthStatus.STARTING
        if rp.kind == "http":
            ok = await _http_probe(rp.target, timeout_s=rp.interval_s)
            return HealthStatus.READY if ok else HealthStatus.STARTING
        if rp.kind == "cmd":
            ok = await _cmd_probe(rp.target, timeout_s=rp.interval_s)
            return HealthStatus.READY if ok else HealthStatus.STARTING
        return HealthStatus.DEGRADED


async def _tcp_probe(host: str, port: int, timeout_s: float) -> bool:
    try:
        fut = asyncio.open_connection(host, port)
        reader, writer = await asyncio.wait_for(fut, timeout=timeout_s)
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        return True
    except Exception:
        return False


async def _http_probe(url: str, timeout_s: float) -> bool:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(url)
        return response.status_code < 500
    except httpx.HTTPError:
        return False


async def _cmd_probe(target: str, timeout_s: float) -> bool:
    return await _argv_probe(shlex.split(target), timeout_s=timeout_s)


async def _argv_probe(argv: list[str], timeout_s: float) -> bool:
    check_proc = None
    try:
        check_proc = await asyncio.create_subprocess_exec(
            *argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        rc = await asyncio.wait_for(check_proc.wait(), timeout=timeout_s)
        return rc == 0
    except (OSError, TimeoutError):
        if check_proc is not None and check_proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                check_proc.kill()
            with contextlib.suppress(Exception):
                await check_proc.wait()
        return False
