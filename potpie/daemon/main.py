"""Runnable local Potpie daemon host."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import socket
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from starlette.concurrency import run_in_threadpool

from potpie.daemon.http.ui import build_ui_api_router, mount_ui_static
from potpie.daemon.process.pidfile import (
    remove_pid_file,
    write_discovery,
    write_pid_file,
)
from adapters.outbound.pots.local_pot_store import default_home
from bootstrap.logging_setup import configure_logging
from bootstrap.observability_context import correlation_scope
from bootstrap.observability_runtime import get_observability
from potpie.runtime import build_potpie_host_shell
from domain.errors import CapabilityNotImplemented, PotNotFound
from domain.ports.observability import SPAN_KIND_SERVER
from potpie.daemon.rpc import decode, encode, validate_rpc_attr, validate_rpc_method

logger = logging.getLogger(__name__)

_LOCKED_RPC_METHODS: frozenset[tuple[str, str]] = frozenset(
    {
        ("backend", "provision"),
        ("backend.mutation", "apply"),
        ("backend.mutation", "invalidate"),
        ("backend.mutation", "reset_pot"),
        ("config", "set"),
        ("graph", "mutate"),
        ("graph", "record"),
        ("graph_workbench", "commit"),
        ("graph_workbench", "inbox_add"),
        ("graph_workbench", "inbox_claim"),
        ("graph_workbench", "inbox_close"),
        ("graph_workbench", "inbox_mark_applied"),
        ("graph_workbench", "inbox_mark_rejected"),
        ("graph_workbench", "propose"),
        ("pots", "add_source"),
        ("pots", "archive_pot"),
        ("pots", "clear_repo_default"),
        ("pots", "create_pot"),
        ("pots", "init"),
        ("pots", "remove_source"),
        ("pots", "rename_pot"),
        ("pots", "reset_pot"),
        ("pots", "set_repo_default"),
        ("pots", "use_pot"),
        ("setup", "run"),
        ("skills", "add"),
        ("skills", "install"),
        ("skills", "nudge"),
        ("skills", "remove"),
        ("skills", "update"),
    }
)


def create_app(*, token: str, base_url: str, pid: int, log_file: str) -> FastAPI:
    host = build_potpie_host_shell()
    rpc_lock = asyncio.Lock()
    home = default_home()
    pid_file = home / "daemon.pid"
    discovery_file = home / "discovery.json"

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        home.mkdir(parents=True, exist_ok=True)
        write_pid_file(pid_file, pid)
        discovery = dict(
            transport="http",
            base_url=base_url,
            token=token,
            pid=pid,
            log_file=log_file,
            backend=host.backend.profile,
        )
        write_discovery(discovery_file, **discovery)
        try:
            yield
        finally:
            remove_pid_file(pid_file)
            remove_pid_file(discovery_file)

    app = FastAPI(title="potpie-daemon", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        with correlation_scope(
            source="daemon_health", trace_id=_daemon_trace_id("health")
        ):
            with get_observability().span(
                "daemon.health",
                kind=SPAN_KIND_SERVER,
                attributes={"backend": host.backend.profile, "pid": pid},
            ):
                return {
                    "ok": True,
                    "mode": "daemon",
                    "pid": pid,
                    "backend": host.backend.profile,
                }

    @app.post("/rpc")
    async def rpc(
        payload: dict[str, Any], authorization: str | None = Header(None)
    ) -> dict[str, Any]:
        _authorize(authorization, token)
        with correlation_scope(source="daemon_rpc", trace_id=_daemon_trace_id("rpc")):
            with get_observability().span("daemon.rpc", kind=SPAN_KIND_SERVER) as span:
                try:
                    surface = str(payload["surface"])
                    method = str(payload["method"])
                    validate_rpc_method(surface, method)
                    span.set_attributes({"rpc.surface": surface, "rpc.method": method})
                    args = _decode_rpc_args(payload.get("args") or [])
                    kwargs = _decode_rpc_kwargs(payload.get("kwargs") or {})
                    if kwargs.get("pot_id"):
                        span.set_attribute("pot_id", kwargs["pot_id"])
                    target = _resolve(host, surface)
                    if _requires_rpc_lock(surface, method):
                        async with rpc_lock:
                            result = await _invoke_rpc_method(
                                target, method, args, kwargs
                            )
                    else:
                        result = await _invoke_rpc_method(target, method, args, kwargs)
                    return {"ok": True, "result": encode(result)}
                except Exception as exc:  # noqa: BLE001
                    span.record_exception(exc)
                    span.set_error(exc.__class__.__name__)
                    return _error_payload(exc)

    @app.post("/attr")
    async def attr(
        payload: dict[str, Any], authorization: str | None = Header(None)
    ) -> dict[str, Any]:
        _authorize(authorization, token)
        with correlation_scope(source="daemon_attr", trace_id=_daemon_trace_id("attr")):
            with get_observability().span("daemon.attr", kind=SPAN_KIND_SERVER) as span:
                try:
                    surface = str(payload["surface"])
                    name = str(payload["name"])
                    validate_rpc_attr(surface, name)
                    span.set_attributes({"rpc.surface": surface, "rpc.attr": name})
                    target = _resolve(host, surface)
                    return {"ok": True, "result": encode(getattr(target, name))}
                except Exception as exc:  # noqa: BLE001
                    span.record_exception(exc)
                    span.set_error(exc.__class__.__name__)
                    return _error_payload(exc)

    app.include_router(build_ui_api_router(host), prefix="/ui")
    mount_ui_static(app)

    return app


def main() -> None:
    configure_logging()
    from potpie.daemon.telemetry.sentry_runtime import configure_daemon_sentry

    configure_daemon_sentry()
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
        if not part or part.startswith("_"):
            raise ValueError("invalid RPC path")
        obj = getattr(obj, part)
    return obj


def _requires_rpc_lock(surface: str, method: str) -> bool:
    return (surface, method) in _LOCKED_RPC_METHODS


def _decode_rpc_args(value: Any) -> list[Any]:
    args = decode(value)
    if isinstance(args, tuple):
        return list(args)
    if isinstance(args, list):
        return args
    raise ValueError("invalid RPC args")


def _decode_rpc_kwargs(value: Any) -> dict[str, Any]:
    kwargs = decode(value)
    if isinstance(kwargs, dict):
        return kwargs
    raise ValueError("invalid RPC kwargs")


async def _invoke_rpc_method(
    target: Any,
    method: str,
    args: list[Any],
    kwargs: dict[str, Any],
) -> Any:
    fn = getattr(target, method)
    result = await run_in_threadpool(fn, *args, **kwargs)
    if asyncio.iscoroutine(result):
        result = await result
    return result


def _authorize(header: str | None, token: str) -> None:
    expected = f"Bearer {token}"
    if header is None or not secrets.compare_digest(header, expected):
        raise HTTPException(status_code=401, detail="invalid daemon token")


def _error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, CapabilityNotImplemented):
        logger.debug("daemon RPC returned expected capability error: %s", exc)
        error = {
            "code": "not_implemented",
            "message": str(exc),
            "capability": exc.capability,
            "detail": exc.detail,
            "recommended_next_action": exc.recommended_next_action,
        }
    elif isinstance(exc, PotNotFound):
        logger.debug("daemon RPC returned expected missing pot error: %s", exc)
        error = {"code": "pot_not_found", "message": str(exc)}
    elif isinstance(exc, ValueError):
        logger.debug("daemon RPC returned expected validation error: %s", exc)
        # Domain validation errors may carry structured guidance (e.g.
        # UnknownGraphViewError's did_you_mean) that the CLI error envelope
        # surfaces; keep it on the wire like CapabilityNotImplemented above.
        error = {
            "code": "validation_error",
            "message": str(exc),
            "detail": getattr(exc, "detail", None),
            "recommended_next_action": getattr(exc, "recommended_next_action", None),
        }
    else:
        logger.exception("daemon RPC failed")
        from potpie.daemon.telemetry.sentry_runtime import (
            capture_unexpected_daemon_error,
        )

        capture_unexpected_daemon_error(
            exc,
            error_code="daemon_error",
            error_kind="unexpected",
        )
        error = {
            "code": "daemon_error",
            "message": str(exc) or exc.__class__.__name__,
        }
    return {"ok": False, "error": error}


def _daemon_trace_id(operation: str) -> str:
    return f"daemon_{operation}_{secrets.token_hex(8)}"


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
