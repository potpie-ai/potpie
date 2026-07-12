"""Runnable root-owned daemon hosting one standalone ``ContextEngine``."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException

from potpie_context_engine import ContextEngine, EngineConfig, create_engine

from potpie.daemon.http.ui import build_ui_api_router, mount_ui_static
from potpie.daemon.process.pidfile import (
    remove_pid_file,
    write_discovery,
    write_pid_file,
)
from potpie.daemon.rpc import dispatch_rpc
from potpie.runtime.paths import product_data_dir
from potpie.runtime.settings import ProductSettings

logger = logging.getLogger(__name__)


def create_app(
    *,
    token: str,
    base_url: str,
    pid: int,
    log_file: str,
    engine: ContextEngine | None = None,
    data_dir: Path | None = None,
) -> FastAPI:
    settings = ProductSettings.load()
    home = (
        data_dir
        or (engine.config.data_dir if engine is not None else None)
        or settings.data_dir
    )
    engine = engine or create_engine(
        EngineConfig.persistent(data_dir=home, backend=settings.backend)
    )
    rpc_lock = asyncio.Lock()
    pid_file = home / "daemon.pid"
    discovery_file = home / "discovery.json"

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        home.mkdir(parents=True, exist_ok=True)
        write_pid_file(pid_file, pid)
        discovery = dict(
            transport="http",
            protocol_version="1",
            base_url=base_url,
            token=token,
            pid=pid,
            log_file=log_file,
            backend=engine.config.backend,
        )
        write_discovery(discovery_file, **discovery)
        try:
            yield
        finally:
            await engine.aclose()
            remove_pid_file(pid_file)
            remove_pid_file(discovery_file)

    app = FastAPI(title="potpie-daemon", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "transport": "http",
            "protocol_version": "1",
            "pid": pid,
        }

    @app.post("/rpc")
    async def rpc(
        payload: dict[str, Any], authorization: str | None = Header(None)
    ) -> dict[str, Any]:
        _authorize(authorization, token)
        async with rpc_lock:
            return await dispatch_rpc(engine, payload)

    # The local UI remains root-owned. Until its handlers migrate to the async
    # runtime in Commit 6, bind it to the same engine's internal legacy shell;
    # no second product/engine facade is constructed in the daemon process.
    app.include_router(build_ui_api_router(engine._shell), prefix="/ui")
    mount_ui_static(app)
    return app


def main() -> None:
    from potpie_context_engine.bootstrap.logging_setup import configure_logging

    configure_logging()
    from potpie.daemon.telemetry.sentry_runtime import configure_daemon_sentry

    configure_daemon_sentry()
    home = product_data_dir()
    home.mkdir(parents=True, exist_ok=True)
    log_file = str(home / "daemon.log")
    port = int(os.getenv("POTPIE_DAEMON_PORT") or _free_port())
    token = os.getenv("POTPIE_DAEMON_TOKEN") or secrets.token_urlsafe(32)
    base_url = f"http://127.0.0.1:{port}"
    app = create_app(
        token=token,
        base_url=base_url,
        pid=os.getpid(),
        log_file=log_file,
        data_dir=home,
    )
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info", access_log=False)


def _authorize(header: str | None, token: str) -> None:
    expected = f"Bearer {token}"
    if header is None or not secrets.compare_digest(header, expected):
        raise HTTPException(status_code=401, detail="invalid daemon token")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


__all__ = ["create_app", "main"]


if __name__ == "__main__":
    main()
