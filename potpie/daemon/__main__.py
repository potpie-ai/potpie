"""Canonical local daemon process launched with ``python -m potpie.daemon``."""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from potpie.daemon.discovery import (
    read_daemon_credential,
    remove_daemon_runtime_records,
)
from potpie.daemon.http.ui import build_ui_api_router, mount_ui_static
from potpie.runtime import CanonicalDaemonRuntime, RuntimeEndpoint
from potpie.runtime.clients import TypedEngineOperationHandler
from potpie.runtime.composition import LocalRuntimeComposition, build_local_runtime
from potpie.runtime.local_engine import build_local_resource_manager
from potpie.runtime.server import run_foreground
from potpie_context_engine.bootstrap.logging_setup import configure_logging

_ENV_ENDPOINT_KIND = "POTPIE_DAEMON_ENDPOINT_KIND"
_ENV_ENDPOINT_ADDRESS = "POTPIE_DAEMON_ENDPOINT_ADDRESS"
_ENV_ENDPOINT_PORT = "POTPIE_DAEMON_ENDPOINT_PORT"
_ENV_INSTANCE_ID = "POTPIE_DAEMON_INSTANCE_ID"
_ENV_UI_PORT = "POTPIE_DAEMON_UI_PORT"


def main() -> None:
    """Run exactly one canonical daemon instance in the current process."""

    configure_logging()
    from potpie.daemon.telemetry.sentry_runtime import configure_daemon_sentry

    configure_daemon_sentry()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


async def _run() -> None:
    home = Path(_required_env("CONTEXT_ENGINE_HOME")).resolve()
    instance_id = _required_env(_ENV_INSTANCE_ID)
    endpoint = _endpoint_from_environment()
    ui_port = int(_required_env(_ENV_UI_PORT))
    ui_url = f"http://127.0.0.1:{ui_port}"
    bearer_token = read_daemon_credential(home)

    composition = build_local_runtime()
    resource_manager = build_local_resource_manager(composition.engine)
    ui_server = _build_ui_server(composition=composition, port=ui_port)
    ui_task = asyncio.create_task(ui_server.serve())
    runtime = CanonicalDaemonRuntime(
        endpoint=endpoint,
        bearer_token=bearer_token,
        operation_handler=TypedEngineOperationHandler(resource_manager),
        ownership_lock_path=home / "daemon.runtime.lock",
        instance_id=instance_id,
        shutdown_resources=resource_manager.shutdown,
        backend_profile=str(composition.root.backend.profile),
        ui_url=ui_url,
    )
    try:
        await _wait_for_ui_start(ui_server, ui_task)
        await run_foreground(runtime)
    finally:
        ui_server.should_exit = True
        with contextlib.suppress(Exception):
            await ui_task
        with contextlib.suppress(Exception):
            await runtime.stop()
        remove_daemon_runtime_records(
            home,
            expected_instance_id=instance_id,
            expected_pid=os.getpid(),
        )


def _build_ui_server(
    *, composition: LocalRuntimeComposition, port: int
) -> uvicorn.Server:
    app = FastAPI(title="potpie-daemon-ui")
    app.include_router(
        build_ui_api_router(
            pots=composition.root.pots,
            graph=composition.engine.graph,
            backend=composition.engine.backend,
        ),
        prefix="/ui",
    )
    mount_ui_static(app)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    return server


async def _wait_for_ui_start(
    server: uvicorn.Server,
    task: asyncio.Task[None],
) -> None:
    while not server.started:
        if task.done():
            await task
            raise RuntimeError("daemon UI server exited before readiness")
        await asyncio.sleep(0.01)


def _endpoint_from_environment() -> RuntimeEndpoint:
    kind = _required_env(_ENV_ENDPOINT_KIND)
    address = _required_env(_ENV_ENDPOINT_ADDRESS)
    if kind == "uds":
        return RuntimeEndpoint(kind="uds", address=address)
    if kind == "tcp":
        return RuntimeEndpoint(
            kind="tcp",
            address=address,
            port=int(_required_env(_ENV_ENDPOINT_PORT)),
        )
    raise RuntimeError("unsupported daemon endpoint kind")


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"required daemon environment is missing: {name}")
    return value


if __name__ == "__main__":
    main()


__all__ = ["main"]
