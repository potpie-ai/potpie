"""HTTP transport: serves OperationRegistry over UDS or TCP. Generic dispatch — knows nothing about specific operations."""

from __future__ import annotations

import asyncio
import json
import pathlib
import socket

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from potpie.daemon.http.errors import error_envelope, status_for_error
from potpie.daemon.runtime.service_manager import (
    DependencyCycle,
    ServiceNotFound,
)
from potpie_context_engine.domain.ports.daemon.operations import (
    AuthRequirement,
    OperationContext,
    OperationError,
    OperationRegistry,
    Principal,
)
from potpie_context_engine.domain.ports.daemon.shell import HealthStatus
from potpie.daemon.runtime.context import ShellContext
from potpie.daemon.runtime.health import HealthRegistrar
from potpie.daemon.runtime.ipc_auth import AuthFailure, IpcAuthGate


class HttpTransport:
    """Built-in transport. Serves operations over a UNIX domain socket (unix:/path) or TCP (tcp:host:port)."""

    def __init__(
        self,
        bind: str,
        auth: IpcAuthGate,
        health: HealthRegistrar,
        health_key: str = "transport:http",
    ) -> None:
        self._bind = bind
        self._auth = auth
        self._health = health
        self._health_key = health_key
        self._server: uvicorn.Server | None = None
        self._ctx: ShellContext | None = None
        self._sock: socket.socket | None = None
        self._bound_port: int | None = None

    def bind(self, ctx: ShellContext) -> None:
        self._ctx = ctx
        self._health.set(self._health_key, HealthStatus.STARTING)
        if self._bind.startswith("tcp:"):
            _, host, port = self._bind.split(":", 2)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, int(port)))
            sock.listen()
            sock.setblocking(False)
            self._sock = sock
            self._bound_port = sock.getsockname()[1]
        elif not self._bind.startswith("unix:"):
            raise ValueError(
                f"unsupported bind {self._bind!r}; use unix:/path or tcp:host:port"
            )

    def bound_port(self) -> int | None:
        return self._bound_port

    def health(self) -> HealthStatus:
        return self._health.get(self._health_key)

    async def serve(self, ops: OperationRegistry) -> None:
        if self._ctx is None:
            raise RuntimeError("bind() before serve()")
        app = self._build_app(ops)
        if self._bind.startswith("unix:"):
            sockpath = pathlib.Path(self._bind[len("unix:") :]).expanduser()
            sockpath.parent.mkdir(parents=True, exist_ok=True)
            if sockpath.exists():
                sockpath.unlink()
            config = uvicorn.Config(
                app=app, uds=str(sockpath), log_level="warning", lifespan="off"
            )
            server = uvicorn.Server(config)
            self._server = server
            watcher = asyncio.create_task(self._mark_ready_when_started())
            try:
                await server.serve()
            finally:
                watcher.cancel()
                if sockpath.exists():
                    sockpath.unlink()
                self._health.set(self._health_key, HealthStatus.STOPPED)
        else:  # tcp
            config = uvicorn.Config(app=app, log_level="warning", lifespan="off")
            server = uvicorn.Server(config)
            self._server = server
            watcher = asyncio.create_task(self._mark_ready_when_started())
            try:
                await server.serve(sockets=[self._sock])
            finally:
                watcher.cancel()
                if self._sock is not None:
                    self._sock.close()
                    self._sock = None
                self._health.set(self._health_key, HealthStatus.STOPPED)

    async def _mark_ready_when_started(self) -> None:
        """Flip health to READY only once uvicorn reports the socket is bound and serving."""
        try:
            while self._server is None or not self._server.started:
                await asyncio.sleep(0.01)
            self._health.set(self._health_key, HealthStatus.READY)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        elif self._sock is not None:
            # serve() was never called; release the listening socket bound in bind()
            self._sock.close()
            self._sock = None

    def _resolve_principal(self, request: Request) -> Principal | None:
        """UDS → trust loopback OS user. TCP → require token only when one is configured."""
        if self._bind.startswith("unix:"):
            return Principal(name="local")
        if not self._auth.token_configured:
            return Principal(name="local")
        tok = request.headers.get("x-potpie-token", "")
        try:
            return self._auth.authenticate_token(tok)
        except AuthFailure:
            return None

    def _build_app(self, ops: OperationRegistry) -> FastAPI:
        ctx = self._ctx
        if ctx is None:
            raise RuntimeError("bind() before _build_app()")
        app = FastAPI(title="potpied", version="0.1.0")
        transport = self

        @app.post("/op/{name}")
        async def dispatch(name: str, request: Request):
            try:
                op = ops.get(name)
            except KeyError:
                return JSONResponse(
                    status_code=404,
                    content=error_envelope(
                        OperationError("not_found", f"unknown operation {name!r}")
                    ),
                )
            principal = transport._resolve_principal(request)
            if op.auth == AuthRequirement.REQUIRED and principal is None:
                return JSONResponse(
                    status_code=401,
                    content=error_envelope(
                        OperationError("unauthorized", "authentication required")
                    ),
                )
            if principal is None:
                principal = Principal(name="anonymous")
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            try:
                inp = op.input_model.model_validate(raw)
            except ValidationError as ve:
                return JSONResponse(
                    status_code=400,
                    content=error_envelope(
                        OperationError(
                            "invalid_input",
                            "validation failed",
                            detail={"errors": json.loads(ve.json())},
                        )
                    ),
                )
            octx = OperationContext(
                principal=principal,
                request_id=request.headers.get("x-request-id", ""),
                deps=ctx.config.get("deps"),
            )
            try:
                out = await op.handler(inp, octx)
            except OperationError as oe:
                return JSONResponse(
                    status_code=status_for_error(oe.code),
                    content=error_envelope(oe),
                )
            except Exception:  # noqa: BLE001 — convert any handler error to a uniform envelope
                ctx.logger.exception("operation handler failed for %r", name)
                return JSONResponse(
                    status_code=500,
                    content=error_envelope(
                        OperationError("internal_error", "An internal error occurred")
                    ),
                )
            return JSONResponse(status_code=200, content=out.model_dump())

        @app.get("/op")
        async def list_ops():
            return {"operations": [o.name for o in ops.all()]}

        @app.get("/admin/health")
        async def admin_health():
            st = transport._health.snapshot().get(transport._health_key)
            return {"status": st.value if st else "unknown"}

        @app.get("/admin/services")
        async def admin_services():
            mgr = (ctx.config or {}).get("service_manager")
            if mgr is None:
                return {"services": []}
            return {
                "services": [
                    {"name": s.name, "status": s.status.value, "endpoint": s.endpoint}
                    for s in mgr.status()
                ]
            }

        @app.post("/admin/services/{name}/up")
        async def admin_service_up(name: str):
            mgr = (ctx.config or {}).get("service_manager")
            if mgr is None:
                return JSONResponse(
                    status_code=503,
                    content=error_envelope(
                        OperationError("unavailable", "service manager not available")
                    ),
                )
            try:
                ep = await mgr.up(name)
                return {"endpoint": ep}
            except (DependencyCycle, ServiceNotFound, ValidationError) as ex:
                return JSONResponse(
                    status_code=400,
                    content=error_envelope(OperationError("invalid_input", str(ex))),
                )
            except Exception:  # noqa: BLE001 — admin boundary maps startup failures
                ctx.logger.exception("service startup failed for %r", name)
                return JSONResponse(
                    status_code=503,
                    content=error_envelope(
                        OperationError("unavailable", "An internal error occurred")
                    ),
                )

        @app.post("/admin/services/{name}/down")
        async def admin_service_down(name: str):
            mgr = (ctx.config or {}).get("service_manager")
            if mgr is None:
                return JSONResponse(
                    status_code=503,
                    content=error_envelope(
                        OperationError("unavailable", "service manager not available")
                    ),
                )
            await mgr.down(name)
            return {"ok": True}

        return app
