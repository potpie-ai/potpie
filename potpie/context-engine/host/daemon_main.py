"""Runnable local Potpie daemon host."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import socket
import sys
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from starlette.concurrency import run_in_threadpool

from adapters.outbound.daemon_process.pidfile import (
    remove_pid_file,
    write_discovery,
    write_pid_file,
)
from adapters.outbound.pots.local_pot_store import default_home
from bootstrap.host_wiring import build_host_shell
from domain.errors import CapabilityNotImplemented, PotNotFound
from host.daemon_rpc import decode, encode

logger = logging.getLogger(__name__)


def create_app(*, token: str, base_url: str, pid: int, log_file: str) -> FastAPI:
    host = build_host_shell()
    home = default_home()
    pid_file = home / "daemon.pid"
    discovery_file = home / "daemon.json"

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        home.mkdir(parents=True, exist_ok=True)
        write_pid_file(pid_file, pid)
        write_discovery(
            discovery_file,
            transport="http",
            base_url=base_url,
            token=token,
            pid=pid,
            log_file=log_file,
        )
        try:
            yield
        finally:
            remove_pid_file(pid_file)
            remove_pid_file(discovery_file)

    app = FastAPI(title="potpie-daemon", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "mode": "daemon",
            "pid": pid,
            "backend": host.backend.profile,
        }

    @app.post("/rpc")
    async def rpc(payload: dict[str, Any], authorization: str | None = Header(None)) -> dict[str, Any]:
        _authorize(authorization, token)
        try:
            surface = str(payload["surface"])
            method = str(payload["method"])
            args = decode(payload.get("args") or [])
            kwargs = decode(payload.get("kwargs") or {})
            target = _resolve(host, surface)
            fn = getattr(target, method)
            result = await run_in_threadpool(fn, *args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return {"ok": True, "result": encode(result)}
        except Exception as exc:  # noqa: BLE001
            return _error_payload(exc)

    @app.post("/attr")
    def attr(payload: dict[str, Any], authorization: str | None = Header(None)) -> dict[str, Any]:
        _authorize(authorization, token)
        try:
            target = _resolve(host, str(payload["surface"]))
            return {"ok": True, "result": encode(getattr(target, str(payload["name"])))}
        except Exception as exc:  # noqa: BLE001
            return _error_payload(exc)

    # Local graph-explorer UI. Loopback-only (the daemon binds 127.0.0.1), so the
    # read-only /ui surface is left unauthenticated; /rpc and /attr stay token-gated.
    # The API router is included before the static mount so /ui/api/* wins over the
    # SPA catch-all at /ui.
    from fastapi.responses import RedirectResponse

    from adapters.inbound.http.ui import build_ui_api_router, mount_ui_static

    app.include_router(build_ui_api_router(host), prefix="/ui")
    mount_ui_static(app)

    @app.get("/", include_in_schema=False)
    def root() -> Any:
        return RedirectResponse(url="/ui")

    return app


def main() -> None:
    home = default_home()
    home.mkdir(parents=True, exist_ok=True)
    log_file = str(home / "daemon.log")
    port = int(os.getenv("POTPIE_DAEMON_PORT") or _free_port())
    token = os.getenv("POTPIE_DAEMON_TOKEN") or secrets.token_urlsafe(32)
    base_url = f"http://127.0.0.1:{port}"
    app = create_app(token=token, base_url=base_url, pid=os.getpid(), log_file=log_file)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info", access_log=False)


def _resolve(host: Any, path: str) -> Any:
    obj = host
    for part in path.split("."):
        if not part:
            raise ValueError("invalid empty RPC path part")
        obj = getattr(obj, part)
    return obj


def _authorize(header: str | None, token: str) -> None:
    expected = f"Bearer {token}"
    if header != expected:
        raise HTTPException(status_code=401, detail="invalid daemon token")


def _error_payload(exc: Exception) -> dict[str, Any]:
    logger.exception("daemon RPC failed")
    if isinstance(exc, CapabilityNotImplemented):
        error = {
            "code": "not_implemented",
            "message": str(exc),
            "capability": exc.capability,
            "detail": exc.detail,
            "recommended_next_action": exc.recommended_next_action,
        }
    elif isinstance(exc, PotNotFound):
        error = {"code": "pot_not_found", "message": str(exc)}
    elif isinstance(exc, ValueError):
        error = {"code": "validation_error", "message": str(exc)}
    else:
        error = {
            "code": "daemon_error",
            "message": str(exc) or exc.__class__.__name__,
        }
    return {"ok": False, "error": error}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)


__all__ = ["create_app", "main"]
