"""
FastAPI Middleware for Automatic Logging Context Injection

This middleware automatically adds request-level context (user_id, request_id, path)
to all logs within a request, without requiring manual log_context() calls in every route.

Best Practice: Use this for automatic context, and log_context() for domain-specific IDs
(conversation_id, project_id) that are only available in specific routes.
"""

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.modules.utils.logger import log_context, setup_logger

logger = setup_logger(__name__)


class LoggingContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to automatically inject request-level context into all logs.

    This ensures that every log entry within a request automatically includes:
    - request_id: Unique identifier for the request
    - path: The API endpoint path
    - user_id: The authenticated user (if available)

    Domain-specific IDs (conversation_id, project_id) should still be added
    manually using log_context() in routes where they're available.
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

        # Add context to all logs in this request
        with log_context(**context):
            # Add request_id to response headers for tracing
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id

            return response
