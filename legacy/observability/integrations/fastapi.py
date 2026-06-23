from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware

from observability import log_context


class LoggingContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        ctx = {
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
        }
        with log_context(**ctx):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
