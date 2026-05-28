"""
WorkspaceSocketService: RPC-style tool execution over Socket.IO.

Resolves workspace_id -> socket_id via Redis, emits tool_call to the extension,
waits for tool_response on Redis pub/sub (rpc:resp:{correlation_id}).

Deduplicates identical tool calls (same workspace_id, endpoint, payload) within
a 4-second window so the extension receives a single tool_call even when the
backend invokes execute_tool_call twice.
"""

import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import uuid4

import redis
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
TOOL_CALL_DEDUP_TTL_SECONDS = 4.0


@dataclass
class _DedupEntry:
    """Shared state for deduplicated tool calls."""

    event: threading.Event
    result: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    expiry_at: float = 0.0


def _dedup_key(workspace_id: str, endpoint: str, payload: Dict[str, Any]) -> str:
    """Stable key for deduplication."""
    payload_str = json.dumps(payload, sort_keys=True)
    return f"{workspace_id}|{endpoint}|{payload_str}"


_tool_call_dedup_lock = threading.Lock()
_tool_call_dedup_map: Dict[str, _DedupEntry] = {}


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
        # Reusable sync Redis pool for is_workspace_online — avoids creating a new
        # connection on every status check (which was the previous behaviour).
        self._sync_redis: Optional[redis.Redis] = None

    def _get_sync_redis(self) -> Optional[redis.Redis]:
        """Return a cached sync Redis client (connection-pooled)."""
        if self._sync_redis is not None:
            return self._sync_redis
        if not self._redis_url:
            return None
        try:
            client = redis.from_url(self._redis_url, decode_responses=True)
            client.ping()
            self._sync_redis = client
            return self._sync_redis
        except Exception as e:
            logger.warning("[WorkspaceSocketService] Sync Redis unavailable: %s", e)
            return None

    async def _get_redis(self) -> Optional[aioredis.Redis]:
        if self._async_redis is not None:
            try:
                await self._async_redis.ping()
                return self._async_redis
            except Exception:
                # Cached client is broken — drop it and reconnect below
                self._async_redis = None
        if not self._redis_url:
            return None
        try:
            self._async_redis = aioredis.from_url(self._redis_url, decode_responses=True)
            await self._async_redis.ping()
            return self._async_redis
        except Exception as e:
            logger.warning("[WorkspaceSocketService] Async Redis unavailable: %s", e)
            self._async_redis = None
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
        sync_redis = self._get_sync_redis()
        if not sync_redis:
            return False
        try:
            key = _workspace_socket_key(workspace_id)
            return sync_redis.exists(key) > 0
        except Exception as e:
            logger.warning("[WorkspaceSocketService] Redis error in is_workspace_online: %s", e)
            # Invalidate cached client so next call re-connects
            self._sync_redis = None
            return False

    def _cleanup_expired_dedup(self) -> None:
        """Remove expired entries from the dedup map."""
        now = time.monotonic()
        with _tool_call_dedup_lock:
            expired = [k for k, v in _tool_call_dedup_map.items() if now >= v.expiry_at]
            for k in expired:
                del _tool_call_dedup_map[k]

    async def _execute_tool_call_impl(
        self,
        workspace_id: str,
        endpoint: str,
        payload: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        """
        Internal: emit tool_call and wait for tool_response. No deduplication.
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

        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)

        try:
            await sio.emit(
                "tool_call",
                event_payload,
                room=socket_id,
                namespace=WORKSPACE_NAMESPACE,
            )
        except Exception as e:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
            logger.warning("[WorkspaceSocketService] emit failed: %s", e)
            raise TunnelConnectionError(
                f"Failed to send tool call to workspace {workspace_id}: {e}",
                last_error=str(e),
            )

        async def _wait_one_response():
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
            raise TunnelConnectionError(
                f"Tool call timed out for workspace_id={workspace_id} (endpoint={endpoint})",
                last_error="timeout",
            )

    async def execute_tool_call(
        self,
        workspace_id: str,
        endpoint: str,
        payload: Dict[str, Any],
        timeout: float = WORKSPACE_TOOL_CALL_TIMEOUT,
    ) -> Dict[str, Any]:
        """
        Emit tool_call to the extension for workspace_id, wait for tool_response via Redis.
        Deduplicates identical calls (same workspace_id, endpoint, payload) within 4 seconds
        so only one tool_call is sent to the extension; duplicate callers share the result.
        Raises TunnelConnectionError if workspace offline or timeout.
        """
        from app.modules.tunnel.tunnel_service import TunnelConnectionError

        key = _dedup_key(workspace_id, endpoint, payload)
        now = time.monotonic()
        expiry_at = now + TOOL_CALL_DEDUP_TTL_SECONDS

        self._cleanup_expired_dedup()

        with _tool_call_dedup_lock:
            existing = _tool_call_dedup_map.get(key)
            if existing is not None and now < existing.expiry_at:
                entry = existing
                is_first = False
                logger.debug(
                    "[WorkspaceSocketService] Dedup: sharing in-flight result for endpoint=%s workspace_id=%s",
                    endpoint,
                    workspace_id,
                )
            else:
                entry = _DedupEntry(event=threading.Event(), expiry_at=expiry_at)
                _tool_call_dedup_map[key] = entry
                is_first = True

        if is_first:
            try:
                result = await self._execute_tool_call_impl(
                    workspace_id, endpoint, payload, timeout
                )
                entry.result = result
            except Exception as e:
                entry.error = e
            finally:
                entry.event.set()
                with _tool_call_dedup_lock:
                    _tool_call_dedup_map.pop(key, None)

        timeout_sec = max(0.1, timeout + 2.0)
        if not entry.event.wait(timeout=timeout_sec):
            raise TunnelConnectionError(
                f"Tool call timed out (dedup wait) for workspace_id={workspace_id}",
                last_error="timeout",
            )
        if entry.error is not None:
            raise entry.error
        assert entry.result is not None
        return entry.result

    async def _execute_tool_call_with_timeout(
        self,
        workspace_id: str,
        endpoint: str,
        payload: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        """Delegate to execute_tool_call; timeout is already enforced there."""
        return await self.execute_tool_call(workspace_id, endpoint, payload, timeout)

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
