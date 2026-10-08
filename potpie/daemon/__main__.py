"""Canonical local daemon process launched with ``python -m potpie.daemon``."""

from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from potpie.daemon.discovery import (
    remove_daemon_runtime_records,
    write_daemon_credential,
)
from potpie.daemon.http.ui import build_ui_api_router, mount_ui_static
from potpie.runtime import CanonicalDaemonRuntime, RuntimeEndpoint
from potpie.runtime.clients import TypedEngineOperationHandler
from potpie.runtime.composition import LocalRuntimeComposition, build_local_runtime
from potpie.runtime.local_engine import build_local_resource_manager
from potpie.runtime.server import run_foreground
from potpie_context_engine.bootstrap.logging_setup import configure_logging
from potpie.daemon.windows_worker import WindowsLadybugOperationHandler

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
    _configure_daemon_process_diagnostics()
    # Ladybug Win64 wheels need OpenSSL DLLs registered before any import of
    # ladybug in this child process (add_dll_directory is not inherited).
    if os.name == "nt":
        from potpie_context_engine.adapters.outbound.graph.ladybug_windows_bootstrap import (
            ensure_ladybug_openssl,
        )

        status = ensure_ladybug_openssl()
        if not status.get("ok") and not status.get("skipped"):
            # Fail closed with a clear log line; readiness handshake will fail.
            import sys

            print(
                f"ladybug openssl bootstrap failed: {status.get('error')}",
                file=sys.stderr,
            )
    from potpie.daemon.telemetry.sentry_runtime import configure_daemon_sentry

    configure_daemon_sentry()
    _configure_daemon_product_analytics()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    except BaseException:
        logger.critical("daemon process terminated unexpectedly", exc_info=True)
        raise


def _configure_daemon_product_analytics() -> None:
    try:
        from potpie.cli.telemetry import settings
        from potpie.cli.telemetry.context import bind_daemon_telemetry_context
        from potpie.cli.telemetry.product_analytics import configure_product_analytics

        configure_product_analytics(settings.load_product_analytics_settings())
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

    composition = build_local_runtime()
    _warm_daemon_embedder(composition)
    resource_manager = build_local_resource_manager(composition.engine)
    operation_handler = TypedEngineOperationHandler(
        resource_manager,
        coordinator=composition.coordinator,
        context_free_handler=composition.graph_metadata,
    )
    operation_handler = WindowsLadybugOperationHandler(
        fallback=operation_handler,
        backend_profile=str(getattr(composition.root.backend, "profile", "")),
        backend=composition.root.backend,
    )
    ui_server = _build_ui_server(composition=composition, port=ui_port)
    ui_task = asyncio.create_task(ui_server.serve())
    runtime = CanonicalDaemonRuntime(
        endpoint=endpoint,
        bearer_token=bearer_token,
        operation_handler=operation_handler,
        ownership_lock_path=home / "daemon.runtime.lock",
        instance_id=instance_id,
        shutdown_resources=resource_manager.shutdown,
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


def _configure_daemon_process_diagnostics() -> None:
    """Make native model failures diagnosable and less thread-sensitive on Windows."""

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if os.name == "nt":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        logger.warning("could not enable daemon fault diagnostics", exc_info=True)


def _warm_daemon_embedder(composition: object) -> None:
    """Load and probe a lazy embedder before the daemon reports readiness.

    The Windows sentence-transformers stack can terminate the hosting process
    in native code while loading a model. A native termination cannot be
    caught by this process, so Windows semantic work is delegated to the
    isolated operation worker instead of risking daemon startup.
    """

    if sys.platform == "win32":
        logger.warning(
            "daemon embedder warmup skipped on Windows; semantic search uses "
            "an isolated worker"
        )
        return

    engine = getattr(composition, "engine", None)
    backend = getattr(engine, "backend", None)
    embedder = getattr(backend, "embedder", None)
    prepare = getattr(embedder, "prepare", None)
    if not callable(prepare):
        return

    embedder_name = str(getattr(embedder, "name", type(embedder).__name__))
    logger.info(
        "daemon embedder warmup starting",
        extra={"embedder": embedder_name, "pid": os.getpid()},
    )
    try:
        metadata = prepare()
    except Exception:
        logger.exception(
            "daemon embedder warmup failed",
            extra={"embedder": embedder_name, "pid": os.getpid()},
        )
        raise
    dimensions = metadata.get("dimensions") if isinstance(metadata, dict) else None
    logger.info(
        "daemon embedder warmup complete",
        extra={
            "embedder": embedder_name,
            "dimensions": dimensions,
            "pid": os.getpid(),
        },
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


def _consume_required_env(name: str) -> str:
    value = os.environ.pop(name, "").strip()
    if not value:
        raise RuntimeError(f"required daemon environment is missing: {name}")
    return value


if __name__ == "__main__":
    main()


__all__ = ["main"]
