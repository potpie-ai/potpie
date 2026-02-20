"""
Middleware for Socket.IO /ws requests.

Auth is handled by first-message pattern in the Socket.IO layer (auth event),
so we allow all /ws requests through at upgrade time; no token validation here.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class SocketAuthMiddleware(BaseHTTPMiddleware):
    """
    Pass-through for /ws. Auth is required via the first-message 'auth' event
    (or token at connect for backward compatibility) in socket_server.
    """

    async def dispatch(self, request: Request, call_next):
        return await call_next(request)
