"""Explicit local subprocess runtime.

This adapter is useful for local development and for read-only analysis. It is
not a security boundary. Production write-capable execution should prefer Docker
or a managed sandbox provider.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
from pathlib import Path
from typing import AsyncIterator

from sandbox.domain.errors import InvalidWorkspacePath, RuntimeCommandRejected, RuntimeNotFound
from sandbox.domain.models import (
    ExecChunk,
    ExecRequest,
    ExecResult,
    Runtime,
    RuntimeCapabilities,
    RuntimeSpec,
    RuntimeState,
    new_id,
    utc_now,
)


class LocalSubprocessRuntimeProvider:
    kind = "local_subprocess"
    capabilities = RuntimeCapabilities()

    def __init__(self, *, allow_write: bool = False) -> None:
        self.allow_write = allow_write
        self._runtimes: dict[str, Runtime] = {}

    async def create(self, workspace_id: str, spec: RuntimeSpec) -> Runtime:
        self._validate_spec(spec)
        runtime = Runtime(
            id=new_id("rt"),
            workspace_id=workspace_id,
            backend_kind=self.kind,
            backend_runtime_id=None,
            spec=spec,
            state=RuntimeState.RUNNING,
        )
        self._runtimes[runtime.id] = runtime
        return runtime

    async def get(self, runtime_id: str) -> Runtime | None:
        return self._runtimes.get(runtime_id)

    async def start(self, runtime: Runtime) -> Runtime:
        self._runtimes.setdefault(runtime.id, runtime)
        runtime.state = RuntimeState.RUNNING
        runtime.last_started_at = utc_now()
        runtime.updated_at = utc_now()
        return runtime

    async def stop(self, runtime: Runtime) -> Runtime:
        self._runtimes.setdefault(runtime.id, runtime)
        runtime.state = RuntimeState.STOPPED
        runtime.updated_at = utc_now()
        return runtime

    async def destroy(self, runtime: Runtime) -> None:
        self._runtimes.pop(runtime.id, None)
        runtime.state = RuntimeState.DELETED

    async def exec(self, runtime: Runtime, request: ExecRequest) -> ExecResult:
        if request.command_kind.mutates_workspace and not self.allow_write:
            raise RuntimeCommandRejected(
                "LocalSubprocessRuntimeProvider was created with allow_write=False"
            )
        return await asyncio.to_thread(self._exec_sync, runtime, request)

    async def exec_stream(
        self, runtime: Runtime, request: ExecRequest
    ) -> AsyncIterator[ExecChunk]:
        result = await self.exec(runtime, request)
        if result.stdout:
            yield ExecChunk(stream="stdout", data=result.stdout)
        if result.stderr:
            yield ExecChunk(stream="stderr", data=result.stderr)

    def _exec_sync(self, runtime: Runtime, request: ExecRequest) -> ExecResult:
        workdir = self._resolve_cwd(runtime.spec, request.cwd)
        env = self._build_env(runtime.spec.env, request.env)
        cmd = self._command(request)
        try:
            result = subprocess.run(
                cmd,
                cwd=str(workdir),
                env=env,
                input=request.stdin,
                capture_output=True,
                timeout=request.timeout_s,
                shell=request.shell,
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
        except FileNotFoundError as exc:
            return ExecResult(exit_code=127, stderr=str(exc).encode())

    def _require(self, runtime_id: str) -> Runtime:
        runtime = self._runtimes.get(runtime_id)
        if runtime is None:
            raise RuntimeNotFound(runtime_id)
        return runtime

    def _validate_spec(self, spec: RuntimeSpec) -> None:
        if not spec.mounts:
            raise RuntimeCommandRejected("Local subprocess runtime requires a mount")
        source = Path(spec.mounts[0].source).resolve()
        if not source.exists() or not source.is_dir():
            raise InvalidWorkspacePath(f"Workspace path does not exist: {source}")

    def _resolve_cwd(self, spec: RuntimeSpec, cwd: str | None) -> Path:
        root = Path(spec.mounts[0].source).resolve()
        if cwd is None:
            return root
        requested = Path(cwd)
        if requested.is_absolute():
            raise InvalidWorkspacePath("Local runtime cwd must be relative")
        resolved = (root / requested).resolve()
        if os.path.commonpath([str(root), str(resolved)]) != str(root):
            raise InvalidWorkspacePath("cwd escapes workspace")
        return resolved

    @staticmethod
    def _build_env(
        base: object,
        override: object,
    ) -> dict[str, str]:
        env = {
            "PATH": os.getenv("PATH", "/usr/bin:/bin"),
            "HOME": os.getenv("HOME", "/tmp"),
            "LANG": "C.UTF-8",
            "TERM": "dumb",
        }
        env.update({str(k): str(v) for k, v in dict(base).items()})
        env.update({str(k): str(v) for k, v in dict(override).items()})
        return env

    @staticmethod
    def _command(request: ExecRequest) -> list[str] | str:
        if request.shell:
            if len(request.cmd) == 1:
                return request.cmd[0]
            return " ".join(shlex.quote(part) for part in request.cmd)
        return list(request.cmd)

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
