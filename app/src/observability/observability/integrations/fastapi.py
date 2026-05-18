"""FastAPI/Starlette request-context middleware (extra: observability[fastapi]).

Ports app/modules/utils/logging_middleware.py: binds request_id (X-Request-ID
or generated), path, method, and user_id (if auth set request.state.user)
into log_context for the request; echoes X-Request-ID on the response.

starlette is imported lazily via module __getattr__ so importing the package
never requires starlette; `from ...fastapi import LoggingContextMiddleware`
still works and triggers the import only then.

KNOWN LIMITATION (port-parity, EC3): with BaseHTTPMiddleware the context is
bound in the dispatch task; for streaming/SSE responses the body is produced
in a different task, so logs emitted *during* streaming may not carry the
context. This matches the original implementation; a streaming-safe rebind is
tracked for later, not changed in this port.
"""

from __future__ import annotations


def _build_middleware():
    import uuid

    from starlette.middleware.base import BaseHTTPMiddleware

    from .. import log_context

    class LoggingContextMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
            user_id = None
            state_user = getattr(request.state, "user", None)
            if state_user:
                user_id = state_user.get("user_id")
            ctx = {
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
            }
            if user_id:
                ctx["user_id"] = user_id
            with log_context(**ctx):
                response = await call_next(request)
                response.headers["X-Request-ID"] = request_id
                return response

    return LoggingContextMiddleware


def __getattr__(name: str):
    if name == "LoggingContextMiddleware":
        cls = _build_middleware()
        globals()[name] = cls
        return cls
    raise AttributeError(name)
