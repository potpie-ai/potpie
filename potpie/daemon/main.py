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

from potpie.daemon.http.ui import UiAuth, build_ui_api_router, mount_ui_static
from potpie.daemon.process.pidfile import (
    remove_pid_file,
    write_discovery,
    write_pid_file,
)
from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home
from potpie_context_engine.bootstrap.logging_setup import configure_logging
from potpie_context_engine.bootstrap.observability_context import correlation_scope
from potpie_context_engine.bootstrap.observability_runtime import get_observability
from potpie_context_engine.bootstrap.host_wiring import build_host_shell
from potpie_context_core.errors import (
    CapabilityNotImplemented,
    ContextEngineDisabled,
    PotNotFound,
)
from potpie_context_engine.domain.ports.observability import SPAN_KIND_SERVER
from potpie.daemon.rpc import decode, encode
from potpie.daemon.surfaces import RPC_SURFACES, surface_contract

logger = logging.getLogger(__name__)

#: Kept as a name here because it is imported as one. The set itself moved to
#: :mod:`potpie.daemon.surfaces` so it is a declaration a test can walk and an
#: endpoint can publish, rather than a literal buried in an HTTP module that a
#: second repository copies by hand.
_ALLOWED_RPC_SURFACES = RPC_SURFACES


def _stop_embedded_graph_servers() -> None:
    """Take the embedded graph server down with the daemon that started it.

    The ``falkordb_lite`` server is a separate, daemonized process; nothing else
    stops it when this one exits, so without this every ``potpie daemon stop``
    strands a ``redis-server`` holding the db file open until the machine
    reboots. Import locally and never raise: shutdown must not be the reason a
    stop fails.
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


def create_app(*, token: str, base_url: str, pid: int, log_file: str) -> FastAPI:
    host = build_host_shell()
    rpc_lock = asyncio.Lock()
    home = default_home()
    pid_file = home / "daemon.pid"
    discovery_file = home / "discovery.json"
    legacy_discovery_file = home / "daemon.json"

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        home.mkdir(parents=True, exist_ok=True)
        write_pid_file(pid_file, pid)
        discovery = dict(
            transport="http",
            base_url=base_url,
            token=token,
            pid=pid,
            log_file=log_file,
        )
        write_discovery(discovery_file, **discovery)
        write_discovery(legacy_discovery_file, **discovery)
        try:
            yield
        finally:
            remove_pid_file(pid_file)
            remove_pid_file(discovery_file)
            remove_pid_file(legacy_discovery_file)
            _stop_embedded_graph_servers()

    app = FastAPI(title="potpie-daemon", lifespan=lifespan)
    # ``/ui/api`` reads the same graph ``/rpc`` does — and, for a managed pot,
    # spends this daemon's stored remote token — so it takes the same daemon
    # token. Held on the app because both halves of the browser handoff (the
    # router mints, the shell sets the cookie) need to reach it.
    app.state.ui_auth = UiAuth(token=token)

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

    @app.get("/surfaces")
    def surfaces(authorization: str | None = Header(None)) -> dict[str, Any]:
        """What this host implements, so a client need not assume its own copy.

        The whole reason this exists: the allowlist is maintained twice, in two
        repositories, and the copies drifted without anything noticing. A client
        that can ask stops guessing — and a client asking a host that predates
        this endpoint gets a 404, which it must read as "does not say" rather
        than "implements nothing" (see
        :func:`potpie.cli.hosts.advertised_surfaces`).

        Behind the daemon token like ``/rpc``, and deliberately *not* folded
        into ``/health``: that route is unauthenticated, and the inventory of
        what this process serves is not something to hand to anything that can
        reach the port.
        """
        _authorize(authorization, token)
        return surface_contract()

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
                    _validate_rpc_target(surface, method)
                    span.set_attributes({"rpc.surface": surface, "rpc.method": method})
                    args = decode(payload.get("args") or [])
                    kwargs = decode(payload.get("kwargs") or {})
                    if kwargs.get("pot_id"):
                        span.set_attribute("pot_id", kwargs["pot_id"])
                    async with rpc_lock:
                        target = _resolve(host, surface)
                        fn = getattr(target, method)
                        result = await run_in_threadpool(fn, *args, **kwargs)
                        if asyncio.iscoroutine(result):
                            result = await result
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
                    _validate_rpc_target(surface, name)
                    span.set_attributes({"rpc.surface": surface, "rpc.attr": name})
                    async with rpc_lock:
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


def _validate_rpc_target(surface: str, member: str) -> None:
    """Refuse anything outside the declared contract, in the right vocabulary.

    The two refusals are not the same event and must not arrive as the same
    answer. An undeclared *surface* is this host saying "I do not serve that" —
    a capability answer, which the CLI renders at exit 2 with a next action.
    Reporting it as ``ValueError`` sent it over the wire as ``validation_error``
    (exit 1, "the caller got it wrong", no repair offered), which is how a
    managed deployment missing ``resources`` came to blame the user for asking.

    An undeclared or underscore-prefixed *member* really is a caller mistake and
    stays a ``ValueError``. Nothing about the allowlist's security role changes:
    either way the path is refused before :func:`_resolve` walks a single
    attribute.
    """
    if surface not in _ALLOWED_RPC_SURFACES:
        raise CapabilityNotImplemented(
            surface,
            detail=f"this host does not implement the '{surface}' RPC surface",
            recommended_next_action=(
                "run this command against a host that does, e.g. "
                "'potpie --host local ...'"
            ),
        )
    if not member or member.startswith("_"):
        raise ValueError(f"invalid RPC member: {member}")


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
    elif isinstance(exc, ContextEngineDisabled):
        # "Your backend is down" is an answer, not a daemon fault. Without this
        # branch the whole family (including GraphSubstrateUnavailable and
        # PotTeardownFailed) fell into the ``else``: a traceback in the daemon
        # log and a Sentry event tagged ``is_expected=false`` for an entirely
        # ordinary outcome, and the subclass's specific repair dropped in favour
        # of the client's generic "run potpie doctor".
        #
        # Order is free against the branches above — CapabilityNotImplemented
        # and PotNotFound are siblings under ContextEngineError, not subclasses
        # of this one — but a subclass added under either would need to stay
        # ahead of this branch to keep its own shape.
        logger.debug("daemon RPC returned expected unavailability: %s", exc)
        error = {
            "code": "unavailable",
            "message": str(exc),
            "recommended_next_action": getattr(exc, "recommended_next_action", None),
        }
    elif isinstance(exc, ValueError):
        logger.debug("daemon RPC returned expected validation error: %s", exc)
        # Domain validation errors may carry structured guidance (e.g.
        # UnknownGraphViewError's did_you_mean) that the CLI error envelope
        # surfaces; keep it on the wire like CapabilityNotImplemented above.
        # ``error_code`` is the domain's own stable code where it has one
        # (``ResourceStoreError.code``); the envelope ``code`` stays
        # ``validation_error`` so the client keeps routing these to ValueError.
        error = {
            "code": "validation_error",
            "error_code": getattr(exc, "code", None),
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
