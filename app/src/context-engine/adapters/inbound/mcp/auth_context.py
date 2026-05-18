"""Request-scoped API key for embedded Streamable HTTP MCP (set by ASGI middleware)."""

from __future__ import annotations

from contextvars import ContextVar, Token

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

_api_key_var: ContextVar[str | None] = ContextVar("potpie_mcp_api_key", default=None)
_client_name_var: ContextVar[str | None] = ContextVar("potpie_mcp_client_name", default=None)


def set_mcp_request_context(
    *,
    api_key: str | None,
    client_name: str | None,
) -> tuple[Token, Token]:
    return (
        _api_key_var.set(api_key),
        _client_name_var.set(client_name),
    )


def reset_mcp_request_context(tokens: tuple[Token, Token]) -> None:
    _api_key_var.reset(tokens[0])
    _client_name_var.reset(tokens[1])


def get_mcp_api_key() -> str:
    key = (_api_key_var.get() or "").strip()
    if not key:
        raise ValueError(
            "X-API-Key header is required for Potpie MCP. "
            "Create a key in Potpie Key management and add it to your MCP client headers."
        )
    return key


def get_mcp_client_name() -> str | None:
    name = (_client_name_var.get() or "").strip()
    return name or None


class PotpieMcpApiKeyMiddleware(BaseHTTPMiddleware):
    """Capture X-API-Key (and optional client name) for MCP tool handlers."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
        client_name = (
            request.headers.get("x-potpie-client-name")
            or request.headers.get("X-Potpie-Client-Name")
            or request.headers.get("mcp-client-name")
        )
        tokens = set_mcp_request_context(
            api_key=api_key.strip() if api_key else None,
            client_name=client_name.strip() if client_name else None,
        )
        try:
            return await call_next(request)
        finally:
            reset_mcp_request_context(tokens)
