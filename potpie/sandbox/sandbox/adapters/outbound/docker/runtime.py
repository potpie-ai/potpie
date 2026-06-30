"""Docker CLI runtime provider."""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
from pathlib import Path
from typing import AsyncIterator

from sandbox.domain.errors import RuntimeCommandRejected, RuntimeNotFound, RuntimeUnavailable
from sandbox.domain.models import (
    ExecChunk,
    ExecRequest,
    ExecResult,
    NetworkMode,
    Runtime,
    RuntimeCapabilities,
    RuntimeSpec,
    RuntimeState,
    new_id,
    utc_now,
)


class DockerRuntimeProvider:
    """Long-lived Docker container runtime.

    The provider deliberately uses the Docker CLI instead of the Python Docker
    SDK to keep this package dependency-light. It creates a sleeping container
    per runtime and executes commands through `docker exec`.
    """

    kind = "docker"
    capabilities = RuntimeCapabilities(preview_url=False, interactive_session=False)

    def __init__(self, *, docker_bin: str = "docker", name_prefix: str = "potpie-sandbox") -> None:
        self.docker_bin = docker_bin
        self.name_prefix = name_prefix
        self._runtimes: dict[str, Runtime] = {}

    async def create(self, workspace_id: str, spec: RuntimeSpec) -> Runtime:
        container_name = f"{self.name_prefix}-{new_id('rt').replace('_', '-')}"
        runtime = Runtime(
            id=new_id("rt"),
            workspace_id=workspace_id,
            backend_kind=self.kind,
            backend_runtime_id=container_name,
            spec=spec,
            state=RuntimeState.STARTING,
        )
        await asyncio.to_thread(self._create_container, runtime)
        runtime.state = RuntimeState.RUNNING
        runtime.last_started_at = utc_now()
        runtime.updated_at = utc_now()
        self._runtimes[runtime.id] = runtime
        return runtime

    async def get(self, runtime_id: str) -> Runtime | None:
        return self._runtimes.get(runtime_id)

    async def start(self, runtime: Runtime) -> Runtime:
        self._runtimes.setdefault(runtime.id, runtime)
        if runtime.backend_runtime_id is None:
            raise RuntimeUnavailable("Docker runtime has no container id")
        result = await asyncio.to_thread(self._run, [self.docker_bin, "start", runtime.backend_runtime_id], 60)
        if result.returncode != 0:
            raise RuntimeUnavailable(result.stderr.strip())
        runtime.state = RuntimeState.RUNNING
        runtime.last_started_at = utc_now()
        runtime.updated_at = utc_now()
        return runtime

    async def stop(self, runtime: Runtime) -> Runtime:
        self._runtimes.setdefault(runtime.id, runtime)
        if runtime.backend_runtime_id:
            await asyncio.to_thread(
                self._run,
                [self.docker_bin, "stop", "--time", "5", runtime.backend_runtime_id],
                30,
            )
        runtime.state = RuntimeState.STOPPED
        runtime.updated_at = utc_now()
        return runtime

    async def destroy(self, runtime: Runtime) -> None:
        self._runtimes.pop(runtime.id, None)
        if runtime.backend_runtime_id:
            await asyncio.to_thread(
                self._run,
                [self.docker_bin, "rm", "-f", runtime.backend_runtime_id],
                30,
            )
        runtime.state = RuntimeState.DELETED

    async def exec(self, runtime: Runtime, request: ExecRequest) -> ExecResult:
        return await asyncio.to_thread(self._exec_sync, runtime, request)

    async def exec_stream(
        self, runtime: Runtime, request: ExecRequest
    ) -> AsyncIterator[ExecChunk]:
        result = await self.exec(runtime, request)
        if result.stdout:
            yield ExecChunk(stream="stdout", data=result.stdout)
        if result.stderr:
            yield ExecChunk(stream="stderr", data=result.stderr)

    def _create_container(self, runtime: Runtime) -> None:
        spec = runtime.spec
        if not spec.mounts:
            raise RuntimeCommandRejected("Docker runtime requires at least one mount")
        if runtime.backend_runtime_id is None:
            raise RuntimeUnavailable("Docker runtime has no container name")

        cmd = [
            self.docker_bin,
            "create",
            "--name",
            runtime.backend_runtime_id,
            "--workdir",
            spec.workdir,
            "--network",
            self._docker_network(spec.network),
        ]
        for mount in spec.mounts:
            cmd.extend(["--mount", self._mount_arg(mount.source, mount.target, mount.writable)])
        for key, value in spec.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        if spec.resources:
            if spec.resources.memory_mb:
                cmd.extend(["--memory", f"{spec.resources.memory_mb}m"])
            if spec.resources.cpu:
                cmd.extend(["--cpus", str(spec.resources.cpu)])
        cmd.extend([spec.image, "tail", "-f", "/dev/null"])

        result = self._run(cmd, 120)
        if result.returncode != 0:
            raise RuntimeUnavailable(result.stderr.strip())
        start = self._run([self.docker_bin, "start", runtime.backend_runtime_id], 60)
        if start.returncode != 0:
            raise RuntimeUnavailable(start.stderr.strip())

    def _exec_sync(self, runtime: Runtime, request: ExecRequest) -> ExecResult:
        if runtime.backend_runtime_id is None:
            raise RuntimeUnavailable("Docker runtime has no container id")
        cmd = [self.docker_bin, "exec", "-i"]
        cwd = self._container_cwd(runtime.spec.workdir, request.cwd)
        cmd.extend(["-w", cwd])
        for key, value in request.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.append(runtime.backend_runtime_id)
        if request.shell:
            command = request.cmd[0] if len(request.cmd) == 1 else " ".join(shlex.quote(p) for p in request.cmd)
            cmd.extend(["sh", "-lc", command])
        else:
            cmd.extend(request.cmd)
        try:
            result = subprocess.run(
                cmd,
                input=request.stdin,
                capture_output=True,
                timeout=request.timeout_s,
                check=False,
            )
            stdout, stderr, truncated = self._limit_output(
                result.stdout or b"", result.stderr or b"", request.max_output_bytes
            )
            return ExecResult(
                exit_code=result.returncode,
                stdout=stdout,
                stderr=stderr,
                truncated=truncated,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                exit_code=124,
                stdout=exc.stdout or b"",
                stderr=exc.stderr or b"Command timed out",
                timed_out=True,
            )

    def _require(self, runtime_id: str) -> Runtime:
        runtime = self._runtimes.get(runtime_id)
        if runtime is None:
            raise RuntimeNotFound(runtime_id)
        return runtime

    @staticmethod
    def _docker_network(mode: NetworkMode) -> str:
        if mode is NetworkMode.NONE:
            return "none"
        return "bridge"

    @staticmethod
    def _mount_arg(source: str, target: str, writable: bool) -> str:
        path = Path(source)
        if path.is_absolute():
            parts = ["type=bind", f"source={path}", f"target={target}"]
        else:
            parts = ["type=volume", f"source={source}", f"target={target}"]
        if not writable:
            parts.append("readonly")
        return ",".join(parts)

    @staticmethod
    def _container_cwd(workdir: str, cwd: str | None) -> str:
        if cwd is None:
            return workdir
        if cwd.startswith("/"):
            return cwd
        return os.path.normpath(f"{workdir}/{cwd}")

    @staticmethod
    def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)

    @staticmethod
    def _limit_output(
        stdout: bytes, stderr: bytes, max_bytes: int | None
    ) -> tuple[bytes, bytes, bool]:
        if max_bytes is None or max_bytes <= 0:
            return stdout, stderr, False
        truncated = len(stdout) + len(stderr) > max_bytes
        remaining = max_bytes
        out = stdout[:remaining]
        remaining -= len(out)
        err = stderr[: max(0, remaining)]
        return out, err, truncated
