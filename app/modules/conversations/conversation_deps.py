"""Dependencies for conversation routes (async Redis/session from app state)."""

from fastapi import Request
from fastapi import HTTPException

from app.modules.conversations.utils.redis_streaming import AsyncRedisStreamManager
from app.modules.conversations.session.session_service import AsyncSessionService


def get_async_redis_stream_manager(request: Request) -> AsyncRedisStreamManager:
    """Return the app-scoped AsyncRedisStreamManager. Raises 503 if not available."""
    manager = getattr(request.app.state, "async_redis_stream_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Async Redis stream manager not available",
        )
    return manager


def get_async_session_service(
    request: Request,
) -> AsyncSessionService:
    """Return an AsyncSessionService using the app-scoped async Redis manager."""
    redis_manager = get_async_redis_stream_manager(request)
    return AsyncSessionService(redis_manager)
