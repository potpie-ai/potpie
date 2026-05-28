"""
FastAPI Middleware for Automatic Logging Context Injection

This middleware automatically adds request-level context (user_id, request_id, path)
to all logs within a request, without requiring manual log_context() calls in every route.

Also sets Logfire trace metadata (Baggage) so Pydantic AI / LiteLLM spans created
during the request are queryable in Logfire by user_id and request_id.
"""

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.modules.utils.logger import log_context, setup_logger

logger = setup_logger(__name__)


def _get_logfire_trace_metadata():
    """Lazy import to avoid loading logfire when disabled."""
    try:
        from app.modules.intelligence.tracing.logfire_tracer import (
            is_logfire_enabled,
            logfire_trace_metadata,
        )
        if is_logfire_enabled():
            return logfire_trace_metadata
    except ImportError:
        pass
    return None


class LoggingContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to automatically inject request-level context into all logs.

    This ensures that every log entry within a request automatically includes:
    - request_id: Unique identifier for the request
    - path: The API endpoint path
    - user_id: The authenticated user (if available)

    When Logfire is enabled, also sets trace metadata (user_id, request_id) so
    all spans (including Pydantic AI / LiteLLM) are queryable in Logfire SQL.
    """

    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Extract user_id from request state (set by AuthService.check_auth)
        user_id = None
        if hasattr(request.state, "user") and request.state.user:
            user_id = request.state.user.get("user_id")

        # Create context with request-level information
        context = {
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
        }

        # Add user_id if available
        if user_id:
            context["user_id"] = user_id

        # Logfire: set trace metadata so HTTP and any LLM spans are queryable by user_id / request_id
        logfire_metadata = _get_logfire_trace_metadata()
        trace_kwargs = {"request_id": request_id}
        if user_id:
            trace_kwargs["user_id"] = user_id

        if logfire_metadata and trace_kwargs:
            with logfire_metadata(**trace_kwargs):
                with log_context(**context):
                    response = await call_next(request)
                    response.headers["X-Request-ID"] = request_id
                    return response
        else:
            with log_context(**context):
                response = await call_next(request)
                response.headers["X-Request-ID"] = request_id
                return response
