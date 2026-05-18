"""FastAPI/Starlette request-context middleware (extra: observability[fastapi]).

Ported from current app/modules/utils/logging_middleware.py. Binds request_id
(from X-Request-ID or generated), path, and user_id (if auth set it on
request.state) into log_context for the request scope; echoes X-Request-ID on
the response.

EDGE CASES:
 - request.state.user may be absent/typed differently — guard, never raise
   from middleware.
 - Streaming/SSE responses: context must stay bound for the whole stream, not
   just until headers flush (the codebase streams chat responses).
 - This only covers HTTP. Background/Celery work needs integrations/celery.py
   (EC3) — a request-only middleware is exactly why the audit found weak
   correlation on queued runs.
"""

from __future__ import annotations


class LoggingContextMiddleware:
    """STUB (Phase 1): contract only. Ported in Phase 2."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        raise NotImplementedError("Phase 1 scaffold — ported in Phase 2")
