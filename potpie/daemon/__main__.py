"""Canonical local daemon process launched with ``python -m potpie.daemon``."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

import uvicorn

from potpie import build_info
from potpie.daemon.discovery import (
    remove_daemon_runtime_records,
    write_daemon_credential,
)
from potpie.daemon.http.ui import build_ui_app
from potpie.runtime import CanonicalDaemonRuntime, DaemonBuild, RuntimeEndpoint
from potpie.runtime.clients import LocalEngineClient, TypedEngineOperationHandler
from potpie.runtime.composition import LocalRuntimeComposition, build_local_runtime
from potpie.runtime.local_engine import build_local_resource_manager
from potpie.runtime.resource_manager import ContextResourceManager, ContextSelector
from potpie.runtime.server import run_foreground
from potpie_context_engine.bootstrap.logging_setup import configure_logging

_ENV_ENDPOINT_KIND = "POTPIE_DAEMON_ENDPOINT_KIND"
_ENV_ENDPOINT_ADDRESS = "POTPIE_DAEMON_ENDPOINT_ADDRESS"
_ENV_ENDPOINT_PORT = "POTPIE_DAEMON_ENDPOINT_PORT"
_ENV_INSTANCE_ID = "POTPIE_DAEMON_INSTANCE_ID"
_ENV_UI_PORT = "POTPIE_DAEMON_UI_PORT"
_ENV_BEARER_TOKEN = "POTPIE_DAEMON_BEARER_TOKEN"  # noqa: S105 - env key name

logger = logging.getLogger(__name__)


def main() -> None:
    """Run exactly one canonical daemon instance in the current process."""

    configure_logging()
    from potpie.daemon.telemetry.sentry_runtime import configure_daemon_sentry

    configure_daemon_sentry()
    _configure_daemon_product_analytics()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


def _configure_daemon_product_analytics() -> None:
    try:
        from potpie.cli.telemetry import settings, spool
        from potpie.cli.telemetry.context import bind_daemon_telemetry_context
        from potpie.cli.telemetry.product_analytics import configure_product_analytics

        configure_product_analytics(settings.load_product_analytics_settings())
        # Events are spooled like the CLI's; the daemon has no command exit to
        # ship them at, so each one starts the detached flusher right away.
        spool.launch_after_append()
        bind_daemon_telemetry_context()
    except Exception:  # noqa: BLE001 — analytics must never block daemon start
        return


async def _run() -> None:
    home = Path(_required_env("CONTEXT_ENGINE_HOME")).resolve()
    instance_id = _required_env(_ENV_INSTANCE_ID)
    endpoint = _endpoint_from_environment()
    ui_port = int(_required_env(_ENV_UI_PORT))
    ui_url = f"http://127.0.0.1:{ui_port}"
    bearer_token = _consume_required_env(_ENV_BEARER_TOKEN)
    # Read at boot: an editable install reads its stamp from git, and the
    # checkout may move on while this process keeps serving the code it
    # started with.
    stamp = build_info.build_stamp()

    composition = build_local_runtime()
    resource_manager = build_local_resource_manager(composition.engine)
    ui_server = _build_ui_server(
        composition=composition,
        port=ui_port,
        bearer_token=bearer_token,
        resource_manager=resource_manager,
    )
    ui_task = asyncio.create_task(ui_server.serve())
    runtime = CanonicalDaemonRuntime(
        endpoint=endpoint,
        bearer_token=bearer_token,
        operation_handler=TypedEngineOperationHandler(
            resource_manager,
            coordinator=composition.coordinator,
            context_free_handler=composition.graph_metadata,
        ),
        ownership_lock_path=home / "daemon.runtime.lock",
        instance_id=instance_id,
        shutdown_resources=_then_stop_embedded_graph_servers(resource_manager.shutdown),
        after_ownership_acquired=lambda _ownership: write_daemon_credential(
            home, bearer_token
        ),
        before_ownership_release=lambda _ownership: remove_daemon_runtime_records(
            home,
            expected_instance_id=instance_id,
            expected_pid=os.getpid(),
        ),
        backend_profile=str(composition.root.backend.profile),
        ui_url=ui_url,
        version=build_info.distribution_version(),
        build=DaemonBuild(
            rev=stamp.get("rev"),
            dirty=stamp.get("dirty"),
            built_at=stamp.get("built_at"),
        ),
        coordinator=composition.coordinator,
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


def _then_stop_embedded_graph_servers(
    release_resources: Callable[[], Awaitable[object]],
) -> Callable[[], Awaitable[object]]:
    """Release the engine resources, then the embedded graph server under them.

    The ``falkordb_lite`` server is a separate, daemonized process; nothing else
    stops it when the daemon exits, so without this every ``potpie daemon
    stop`` strands a ``redis-server`` holding the db file open until the
    machine reboots. It runs inside the runtime's resource shutdown, before the
    ownership lock is released, so a daemon started right after this one can
    never attach to a server that is about to be stopped.
    """

    async def shutdown() -> object:
        try:
            return await release_resources()
        finally:
            await asyncio.to_thread(_stop_embedded_graph_servers)

    return shutdown


def _stop_embedded_graph_servers() -> None:
    """Stop the embedded graph servers this process started; never raise.

    Shutdown must not be the reason a stop fails, so the import is local and
    every failure is logged rather than propagated.
    """
    try:
        from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
            shutdown_embedded_servers,
        )

        stopped = shutdown_embedded_servers()
        if stopped:
            logger.info("stopped %d embedded graph server(s)", stopped)
    except Exception:  # noqa: BLE001
        logger.debug("embedded graph server shutdown failed", exc_info=True)


def _build_ui_server(
    *,
    composition: LocalRuntimeComposition,
    port: int,
    bearer_token: str,
    resource_manager: ContextResourceManager,
) -> uvicorn.Server:
    # The explorer takes the same per-boot secret as the typed endpoint: the
    # CLI reads it from the owner-only credential file and spends it on a
    # single-use browser handoff, so no second secret store is needed.
    app = build_ui_app(
        pots=composition.root.pots,
        graph=composition.engine.graph,
        backend=composition.engine.backend,
        bearer_token=bearer_token,
        engine_client=_ui_engine_client(composition, resource_manager),
    )
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


def _ui_engine_client(
    composition: LocalRuntimeComposition, resource_manager: ContextResourceManager
) -> Callable[[str], LocalEngineClient]:
    """Typed clients for explorer routes, on the daemon's own manager and locks.

    The explorer's commit routes send the same typed operations the CLI does,
    so selection, archived-pot refusal and operation locking are shared with
    every daemon request rather than reimplemented for the browser.
    """

    def build(pot_id: str) -> LocalEngineClient:
        return LocalEngineClient(
            selector=ContextSelector(kind="explicit", value=pot_id),
            authentication={"kind": "daemon_ui"},
            resource_manager=resource_manager,
            coordinator=composition.coordinator,
            context_free_handler=composition.graph_metadata,
        )

    return build


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


def _consume_required_env(name: str) -> str:
    value = os.environ.pop(name, "").strip()
    if not value:
        raise RuntimeError(f"required daemon environment is missing: {name}")
    return value


if __name__ == "__main__":
    main()


__all__ = ["main"]
