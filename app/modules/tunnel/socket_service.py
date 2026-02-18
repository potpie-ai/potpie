"""
WorkspaceSocketService: RPC-style tool execution over Socket.IO.

Resolves workspace_id -> socket_id via Redis, emits tool_call to the extension,
waits for tool_response on Redis pub/sub (rpc:resp:{correlation_id}).
"""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional
from uuid import uuid4

import redis.asyncio as aioredis

from app.core.config_provider import ConfigProvider
from app.modules.tunnel.socket_server import (
    RPC_RESPONSE_CHANNEL_PREFIX,
    WORKSPACE_NAMESPACE,
    WORKSPACE_SOCKET_KEY_PREFIX,
    sio,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

WORKSPACE_TOOL_CALL_TIMEOUT = float(os.getenv("WORKSPACE_TOOL_CALL_TIMEOUT", "30"))


def _workspace_socket_key(workspace_id: str) -> str:
    return f"{WORKSPACE_SOCKET_KEY_PREFIX}{workspace_id}"


_socket_service_instance: Optional["WorkspaceSocketService"] = None
_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="socket_rpc")
    return _executor


class WorkspaceSocketService:
    """Service to execute tool calls over Socket.IO with workspace isolation."""

    def __init__(self) -> None:
        config = ConfigProvider()
        self._redis_url = config.get_redis_url()
        self._async_redis: Optional[aioredis.Redis] = None

    async def _get_redis(self) -> Optional[aioredis.Redis]:
        if self._async_redis is not None:
            return self._async_redis
        if not self._redis_url:
            return None
        try:
            self._async_redis = aioredis.from_url(self._redis_url, decode_responses=True)
            await self._async_redis.ping()
            return self._async_redis
        except Exception as e:
            logger.warning("[WorkspaceSocketService] Async Redis unavailable: %s", e)
            return None

    async def _get_socket_id(self, workspace_id: str) -> Optional[str]:
        """Resolve workspace_id to socket id from Redis."""
        redis_client = await self._get_redis()
        if not redis_client:
            return None
        key = _workspace_socket_key(workspace_id)
        sid = await redis_client.get(key)
        return sid

    def is_workspace_online(self, workspace_id: str) -> bool:
        """Synchronous check: is there a socket registered for this workspace_id?"""
        if not self._redis_url:
            return False
        try:
            import redis
            sync_redis = redis.from_url(self._redis_url, decode_responses=True)
            key = _workspace_socket_key(workspace_id)
            return sync_redis.exists(key) > 0
        except Exception:
            return False

    async def execute_tool_call(
        self,
        workspace_id: str,
        endpoint: str,
        payload: Dict[str, Any],
        timeout: float = WORKSPACE_TOOL_CALL_TIMEOUT,
    ) -> Dict[str, Any]:
        """
        Emit tool_call to the extension for workspace_id, wait for tool_response via Redis.
        Raises TunnelConnectionError if workspace offline or timeout.
        """
        from app.modules.tunnel.tunnel_service import TunnelConnectionError

        socket_id = await self._get_socket_id(workspace_id)
        if not socket_id:
            raise TunnelConnectionError(
                f"No socket connected for workspace_id={workspace_id}. "
                "Ensure the VS Code extension is running and registered.",
                last_error="workspace offline",
            )

        correlation_id = str(uuid4())
        channel = f"{RPC_RESPONSE_CHANNEL_PREFIX}{correlation_id}"
        redis_client = await self._get_redis()
        if not redis_client:
            raise TunnelConnectionError(
                "Redis unavailable for RPC response.",
                last_error="redis unavailable",
            )

        event_payload = {
            "correlation_id": correlation_id,
            "endpoint": endpoint,
            "payload": payload,
            "timeout": timeout,
        }
        try:
            await sio.emit(
                "tool_call",
                event_payload,
                room=socket_id,
                namespace=WORKSPACE_NAMESPACE,
            )
        except Exception as e:
            from app.modules.tunnel.tunnel_service import TunnelConnectionError
            logger.warning("[WorkspaceSocketService] emit failed: %s", e)
            raise TunnelConnectionError(
                f"Failed to send tool call to workspace {workspace_id}: {e}",
                last_error=str(e),
            )

        async def _wait_one_response():
            pubsub = redis_client.pubsub()
            await pubsub.subscribe(channel)
            try:
                async for message in pubsub.listen():
                    if message["type"] != "message":
                        continue
                    data = message.get("data")
                    if not data:
                        continue
                    try:
                        if isinstance(data, str):
                            out = json.loads(data)
                        else:
                            out = data
                    except json.JSONDecodeError:
                        continue
                    return out
            finally:
                await pubsub.unsubscribe(channel)
                await pubsub.close()

        try:
            return await asyncio.wait_for(_wait_one_response(), timeout=timeout + 2.0)
        except asyncio.TimeoutError:
            from app.modules.tunnel.tunnel_service import TunnelConnectionError
            raise TunnelConnectionError(
                f"Tool call timed out for workspace_id={workspace_id} (endpoint={endpoint})",
                last_error="timeout",
            )

    async def _execute_tool_call_with_timeout(
        self,
        workspace_id: str,
        endpoint: str,
        payload: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        """Wrapper that applies asyncio.wait_for for timeout."""
        try:
            return await asyncio.wait_for(
                self.execute_tool_call(workspace_id, endpoint, payload, timeout),
                timeout=timeout + 2.0,
            )
        except asyncio.TimeoutError:
            from app.modules.tunnel.tunnel_service import TunnelConnectionError
            raise TunnelConnectionError(
                f"Tool call timed out for workspace_id={workspace_id}",
                last_error="timeout",
            )

    def execute_tool_call_sync(
        self,
        workspace_id: str,
        endpoint: str,
        payload: Dict[str, Any],
        timeout: float = WORKSPACE_TOOL_CALL_TIMEOUT,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for execute_tool_call. Runs the async method in a
        dedicated thread with a new event loop. Use from sync tool code.
        """
        def _run() -> Dict[str, Any]:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._execute_tool_call_with_timeout(
                        workspace_id, endpoint, payload, timeout
                    )
                )
            finally:
                loop.close()

        future = _get_executor().submit(_run)
        try:
            return future.result(timeout=timeout + 5.0)
        except Exception as e:
            from app.modules.tunnel.tunnel_service import TunnelConnectionError
            if isinstance(e, TunnelConnectionError):
                raise
            raise TunnelConnectionError(
                f"Tool call failed for workspace_id={workspace_id}: {e}",
                last_error=str(e),
            )


def get_socket_service() -> WorkspaceSocketService:
    """Return singleton WorkspaceSocketService."""
    global _socket_service_instance
    if _socket_service_instance is None:
        _socket_service_instance = WorkspaceSocketService()
    return _socket_service_instance
