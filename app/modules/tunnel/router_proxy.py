"""
Router Service for wildcard workspace tunnels.

When Host matches *.{TUNNEL_WILDCARD_DOMAIN} and the first label is 16 hex chars
(workspace_id), looks up workspace presence in Redis and proxies the request to
https://{tunnel_id}.cfargotunnel.com.
"""

import os
import re
from typing import Callable

import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.modules.tunnel.tunnel_service import (
    get_tunnel_service,
    _tunnel_wildcard_enabled,
    _tunnel_wildcard_domain,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

ROUTER_PROXY_TIMEOUT = 10.0
WORKSPACE_ID_PATTERN = re.compile(r"^[a-f0-9]{16}$")


async def _proxy_request_to_local(
    request: Request,
    workspace_id: str,
    local_port: int,
) -> Response:
    """Dev mode: Forward request directly to localhost:{local_port}."""
    path = request.url.path or "/"
    if request.url.query:
        path = f"{path}?{request.url.query}"
    url = f"http://127.0.0.1:{local_port}{path}"
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("connection", None)
    headers.pop("transfer-encoding", None)
    headers.pop("keep-alive", None)
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=ROUTER_PROXY_TIMEOUT) as client:
            response = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=body,
            )
        out_headers = dict(response.headers)
        out_headers["X-Potpie-Router"] = "dev-mode-local"
        out_headers["X-Local-Port"] = str(local_port)
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=out_headers,
        )
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={"error": "LOCAL_TIMEOUT", "workspace_id": workspace_id, "local_port": local_port},
        )
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": "LOCAL_ERROR", "workspace_id": workspace_id, "detail": str(e)},
        )


async def _proxy_request_to_tunnel(
    request: Request,
    tunnel_id: str,
    workspace_id: str,
) -> Response:
    """Forward request to https://{tunnel_id}.cfargotunnel.com and return the response."""
    base = f"https://{tunnel_id}.cfargotunnel.com"
    path = request.url.path or "/"
    if request.url.query:
        path = f"{path}?{request.url.query}"
    url = f"{base}{path}"
    headers = dict(request.headers)
    # Drop hop-by-hop and host so upstream sets its own
    headers.pop("host", None)
    headers.pop("connection", None)
    headers.pop("transfer-encoding", None)
    headers.pop("keep-alive", None)
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=ROUTER_PROXY_TIMEOUT) as client:
            response = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=body,
            )
        # Upstream 5xx -> 502 UPSTREAM_ERROR per plan
        if response.status_code >= 500:
            return JSONResponse(
                status_code=502,
                content={
                    "error": "UPSTREAM_ERROR",
                    "workspace_id": workspace_id,
                    "status": response.status_code,
                },
            )
        if response.status_code == 404:
            logger.warning(
                "[RouterProxy] upstream 404 for workspace_id=%s path=%s (tunnel may be disconnected or path not served)",
                workspace_id,
                request.url.path,
            )
        # Signal that Potpie router proxied this; upstream 404 = tunnel connector likely disconnected
        out_headers = dict(response.headers)
        out_headers["X-Potpie-Router"] = "proxied"
        out_headers["X-Upstream-Status"] = str(response.status_code)
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=out_headers,
        )
    except httpx.TimeoutException:
        logger.warning(
            f"[RouterProxy] Timeout proxying to workspace_id={workspace_id} tunnel_id={tunnel_id}"
        )
        return JSONResponse(
            status_code=504,
            content={"error": "TUNNEL_TIMEOUT", "workspace_id": workspace_id},
        )
    except Exception as e:
        logger.warning(f"[RouterProxy] Upstream error workspace_id={workspace_id}: {e}")
        return JSONResponse(
            status_code=502,
            content={
                "error": "UPSTREAM_ERROR",
                "workspace_id": workspace_id,
                "detail": str(e),
            },
        )


class WildcardTunnelRouterMiddleware(BaseHTTPMiddleware):
    """
    If Host is *.{TUNNEL_WILDCARD_DOMAIN} and first label is 16 hex (workspace_id),
    resolve tunnel_id from Redis and proxy to https://{tunnel_id}.cfargotunnel.com.
    Otherwise pass through to the next handler.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not _tunnel_wildcard_enabled():
            return await call_next(request)
        domain = _tunnel_wildcard_domain()
        if not domain:
            return await call_next(request)
        host = (request.headers.get("host") or "").strip().split(":")[0]
        if not host.endswith("." + domain):
            return await call_next(request)
        # First label = workspace_id
        workspace_id = host.split(".")[0]
        if not WORKSPACE_ID_PATTERN.fullmatch(workspace_id):
            return JSONResponse(
                status_code=400,
                content={"error": "INVALID_HOST"},
            )
        tunnel_service = get_tunnel_service()
        presence = tunnel_service.get_workspace_presence(workspace_id)
        tunnel_id = presence.get("tunnel_id") if presence else None
        # Fallback: if no presence (e.g. before first heartbeat), use provisioned workspace tunnel record
        if not tunnel_id:
            record = tunnel_service.get_workspace_tunnel_record(workspace_id)
            if record and record.get("status") == "active" and record.get("tunnel_id"):
                tunnel_id = record["tunnel_id"]
                logger.info(
                    "[RouterProxy] using tunnel_id from workspace record (no presence yet) workspace_id=%s tunnel_id=%s",
                    workspace_id,
                    tunnel_id,
                )
        if not tunnel_id:
            logger.warning(
                "[RouterProxy] workspace_id=%s offline (no presence and no tunnel record); ensure extension provisions and sends heartbeat",
                workspace_id,
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": "WORKSPACE_OFFLINE",
                    "workspace_id": workspace_id,
                },
            )
        logger.info(
            "[RouterProxy] proxying host=%s workspace_id=%s tunnel_id=%s path=%s",
            host,
            workspace_id,
            tunnel_id,
            request.url.path,
        )
        # Dev mode: bypass tunnel and proxy directly to local_port
        if os.getenv("DEV_MODE_BYPASS_TUNNEL") == "true":
            # Get local_port from presence or default to 8013
            local_port = presence.get("local_port") if presence else None
            if not local_port:
                record = tunnel_service.get_workspace_tunnel_record(workspace_id)
                local_port = record.get("local_port") if record else 8013
            logger.info(
                "[RouterProxy] DEV_MODE: bypassing tunnel, proxying directly to localhost:%s",
                local_port,
            )
            return await _proxy_request_to_local(request, workspace_id, local_port)
        return await _proxy_request_to_tunnel(request, tunnel_id, workspace_id)
