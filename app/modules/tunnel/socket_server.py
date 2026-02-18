"""
Socket.IO server for workspace tunnel.

Extension clients connect to namespace /workspace, then send auth event with
Firebase ID token (first-message auth). Alternatively token at connect (handshake
or query/header) is supported for backward compatibility.
Uses Redis for workspace_id -> socket_id and sid -> user_id (auth store).
"""

import asyncio
import json
import os
from typing import Any, Awaitable, Callable, Optional

import redis.asyncio as aioredis
import socketio
from firebase_admin import auth as firebase_auth

from app.core.config_provider import ConfigProvider
from app.modules.tunnel.socket_models import (
    ToolResponseEvent,
    WorkspaceHeartbeatPayload,
    WorkspaceRegisterPayload,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

WORKSPACE_SOCKET_KEY_PREFIX = "workspace:socket:"
SOCKET_WORKSPACE_KEY_PREFIX = "socket:workspace:"
WORKSPACE_SOCKET_TTL = int(os.getenv("WORKSPACE_SOCKET_TTL", "300"))
RPC_RESPONSE_CHANNEL_PREFIX = "rpc:resp:"
AUTH_SID_KEY_PREFIX = "auth:sid:"
AUTH_TIMEOUT_SECS = int(os.getenv("SOCKET_AUTH_TIMEOUT_SECS", "15"))

# In-memory auth store when Redis unavailable (single-instance)
_in_memory_auth_sids: dict[str, str] = {}
# Auth timeout tasks: sid -> Task (cancel when auth succeeds)
_auth_timeout_tasks: dict[str, asyncio.Task] = {}


def _get_redis_url() -> Optional[str]:
    config = ConfigProvider()
    return config.get_redis_url()


redis_url = _get_redis_url()
client_manager: Optional[socketio.AsyncRedisManager] = None
if redis_url:
    try:
        client_manager = socketio.AsyncRedisManager(redis_url)
        logger.info("[SocketServer] Using AsyncRedisManager for Socket.IO")
    except Exception as e:
        logger.warning("[SocketServer] Redis manager failed, using in-memory: %s", e)
        client_manager = None

sio = socketio.AsyncServer(
    async_mode="asgi",
    client_manager=client_manager,
    cors_allowed_origins="*",
)
_raw_socket_asgi = socketio.ASGIApp(sio, socketio_path="socket.io")


# Mount path used in main.py; Engine.IO expects path relative to this (e.g. /socket.io/).
_SOCKET_MOUNT_PATH = "/ws"


class _SocketAsgiWebSocketScopeFix:
    """
    ASGI wrapper for the Socket.IO app when mounted at /ws:

    1. Path rewrite: Starlette's Mount does not set scope["path"] to the
       remainder; the child app still sees path="/ws/socket.io/...". Engine.IO
       expects path="/socket.io/...". We strip the mount prefix so the inner app
       sees the correct path and accepts the connection instead of returning 404.

    2. WebSocket 404 fix: When the inner app (engineio) returns "not found", it
       sends http.response.start/body; for a websocket scope ASGI requires
       websocket.http.response.start/body. We rewrite those message types so
       uvicorn does not raise RuntimeError.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        # Strip mount prefix so Engine.IO sees path /socket.io/... not /ws/socket.io/...
        path = scope.get("path") or ""
        if path.startswith(_SOCKET_MOUNT_PATH):
            path = path[len(_SOCKET_MOUNT_PATH) :] or "/"
            scope = {**scope, "path": path}

        if scope.get("type") != "websocket":
            await self._app(scope, receive, send)
            return

        async def wrapped_send(message: dict[str, Any]) -> None:
            msg_type = message.get("type")
            if msg_type == "http.response.start":
                message = {**message, "type": "websocket.http.response.start"}
            elif msg_type == "http.response.body":
                message = {**message, "type": "websocket.http.response.body"}
            await send(message)

        await self._app(scope, receive, wrapped_send)


socket_asgi = _SocketAsgiWebSocketScopeFix(_raw_socket_asgi)

# Async Redis for workspace_id -> sid mapping (separate from Socket.IO's Redis adapter)
_async_redis: Optional[aioredis.Redis] = None


async def _get_async_redis() -> Optional[aioredis.Redis]:
    global _async_redis
    if _async_redis is not None:
        return _async_redis
    if not redis_url:
        return None
    try:
        _async_redis = aioredis.from_url(redis_url, decode_responses=True)
        await _async_redis.ping()  # type: ignore[misc]
        return _async_redis
    except Exception as e:
        logger.warning("[SocketServer] Async Redis unavailable: %s", e)
        return None


def _workspace_socket_key(workspace_id: str) -> str:
    return f"{WORKSPACE_SOCKET_KEY_PREFIX}{workspace_id}"


def _auth_sid_key(sid: str) -> str:
    return f"{AUTH_SID_KEY_PREFIX}{sid}"


def _socket_workspace_key(sid: str) -> str:
    return f"{SOCKET_WORKSPACE_KEY_PREFIX}{sid}"


async def _refresh_auth_ttl(sid: str) -> None:
    """Refresh TTL for auth:sid:{sid} so long-lived connections stay authenticated."""
    redis_client = await _get_async_redis()
    if redis_client:
        key = _auth_sid_key(sid)
        await redis_client.expire(key, WORKSPACE_SOCKET_TTL)


async def _set_authenticated(sid: str, user_id: str) -> None:
    """Mark socket as authenticated. Redis or in-memory."""
    redis_client = await _get_async_redis()
    if redis_client:
        key = _auth_sid_key(sid)
        await redis_client.setex(key, WORKSPACE_SOCKET_TTL, user_id)
    else:
        _in_memory_auth_sids[sid] = user_id


async def _get_authenticated(sid: str) -> Optional[str]:
    """Return user_id if socket is authenticated, else None."""
    redis_client = await _get_async_redis()
    if redis_client:
        key = _auth_sid_key(sid)
        return await redis_client.get(key)
    return _in_memory_auth_sids.get(sid)


async def _clear_authenticated(sid: str) -> None:
    """Remove auth state for socket (on disconnect)."""
    redis_client = await _get_async_redis()
    if redis_client:
        key = _auth_sid_key(sid)
        await redis_client.delete(key)
    else:
        _in_memory_auth_sids.pop(sid, None)


def _cancel_auth_timeout(sid: str) -> None:
    """Cancel auth timeout task for sid if present."""
    task = _auth_timeout_tasks.pop(sid, None)
    if task and not task.done():
        task.cancel()


async def _auth_timeout_task(sid: str) -> None:
    """Disconnect sid if not authenticated after AUTH_TIMEOUT_SECS."""
    try:
        await asyncio.sleep(AUTH_TIMEOUT_SECS)
        user_id = await _get_authenticated(sid)
        if user_id is None:
            logger.warning("[SocketServer] Auth timeout for sid=%s, disconnecting", sid)
            await sio.disconnect(sid, namespace=WORKSPACE_NAMESPACE)
    except asyncio.CancelledError:
        pass
    finally:
        _auth_timeout_tasks.pop(sid, None)


async def _verify_token(token: Optional[str]) -> Optional[dict]:
    """Verify Firebase ID token; return decoded claims or None."""
    if not token or not isinstance(token, str):
        return None
    if os.getenv("isDevelopmentMode") == "enabled":
        return {"user_id": os.getenv("defaultUsername", "dev-user"), "email": "dev@potpie.ai"}
    try:
        decoded = firebase_auth.verify_id_token(token)
        if "user_id" not in decoded and "uid" in decoded:
            decoded["user_id"] = decoded["uid"]
        return decoded
    except Exception as e:
        logger.debug("[SocketServer] Token verification failed: %s", e)
        return None


WORKSPACE_NAMESPACE = "/workspace"


def _token_from_environ(environ: dict) -> Optional[str]:
    """Get token from upgrade request: query string or Authorization header (same as middleware)."""
    if not environ:
        return None
    # Query string (e.g. ?token=xxx&EIO=4&transport=websocket)
    qs = environ.get("QUERY_STRING") or environ.get("query_string")
    if qs:
        from urllib.parse import parse_qs
        parsed = parse_qs(qs if isinstance(qs, str) else qs.decode())
        tokens = parsed.get("token", [])
        if tokens:
            return tokens[0]
    # Authorization header
    auth = environ.get("HTTP_AUTHORIZATION") or environ.get("Authorization")
    if auth:
        auth_str = auth.decode() if isinstance(auth, bytes) else auth
        if auth_str.lower().startswith("bearer "):
            return auth_str[7:].strip()
    return None


@sio.event(namespace=WORKSPACE_NAMESPACE)
async def connect(sid: str, environ: dict, auth: Optional[dict] = None):
    """
    Allow connection. If token provided at connect (handshake or query/header), verify
    and store so socket is authenticated immediately. Otherwise require 'auth' event;
    auth timeout will disconnect if no auth within AUTH_TIMEOUT_SECS.
    """
    token = None
    if auth and isinstance(auth, dict):
        token = auth.get("token")
    if not token and environ:
        token = _token_from_environ(environ)

    if token:
        user = await _verify_token(token)
        if user:
            user_id = user.get("user_id") or user.get("uid", "")
            await _set_authenticated(sid, user_id)
            logger.info("[SocketServer] Connected sid=%s user_id=%s (token at connect)", sid, user_id)
            return True
        logger.warning("[SocketServer] Connect: invalid token for sid=%s, allowing (send auth event)", sid)

    # No token or invalid: allow connect; client must send 'auth' event (auth timeout applies)
    task = asyncio.create_task(_auth_timeout_task(sid))
    _auth_timeout_tasks[sid] = task
    return True


@sio.event(namespace=WORKSPACE_NAMESPACE)
async def auth(sid: str, data: dict):
    """
    First-message auth: client sends Firebase ID token. Verify and emit auth_success
    or disconnect. In dev mode, missing/invalid token still gets default user.
    """
    _cancel_auth_timeout(sid)
    token = (data or {}).get("token") if isinstance(data, dict) else None

    if os.getenv("isDevelopmentMode") == "enabled":
        user_id = os.getenv("defaultUsername", "dev-user")
        if token:
            user = await _verify_token(token)
            if user:
                user_id = user.get("user_id") or user.get("uid", user_id)
        await _set_authenticated(sid, user_id)
        await sio.emit("auth_success", {"uid": user_id}, room=sid, namespace=WORKSPACE_NAMESPACE)
        logger.info("[SocketServer] auth sid=%s user_id=%s (dev mode)", sid, user_id)
        return

    if not token or not isinstance(token, str):
        await sio.emit("auth_failure", {"reason": "missing token"}, room=sid, namespace=WORKSPACE_NAMESPACE)
        await sio.disconnect(sid, namespace=WORKSPACE_NAMESPACE)
        return
    user = await _verify_token(token)
    if not user:
        await sio.emit("auth_failure", {"reason": "invalid token"}, room=sid, namespace=WORKSPACE_NAMESPACE)
        await sio.disconnect(sid, namespace=WORKSPACE_NAMESPACE)
        return
    user_id = user.get("user_id") or user.get("uid", "")
    await _set_authenticated(sid, user_id)
    await sio.emit("auth_success", {"uid": user_id}, room=sid, namespace=WORKSPACE_NAMESPACE)
    logger.info("[SocketServer] auth sid=%s user_id=%s", sid, user_id)


@sio.event(namespace=WORKSPACE_NAMESPACE)
async def token_refresh(sid: str, data: dict):
    """Refresh auth with a new Firebase ID token (~55 min). Disconnect if invalid."""
    token = (data or {}).get("token") if isinstance(data, dict) else None
    if not token or not isinstance(token, str):
        await sio.disconnect(sid, namespace=WORKSPACE_NAMESPACE)
        return
    user = await _verify_token(token)
    if not user:
        await sio.disconnect(sid, namespace=WORKSPACE_NAMESPACE)
        return
    user_id = user.get("user_id") or user.get("uid", "")
    await _set_authenticated(sid, user_id)
    logger.debug("[SocketServer] token_refresh sid=%s user_id=%s", sid, user_id)


async def _require_auth(sid: str) -> bool:
    """Return True if sid is authenticated; else disconnect and return False."""
    user_id = await _get_authenticated(sid)
    if user_id is not None:
        return True
    logger.warning("[SocketServer] Rejecting unauthenticated sid=%s", sid)
    await sio.disconnect(sid, namespace=WORKSPACE_NAMESPACE)
    return False


@sio.event(namespace=WORKSPACE_NAMESPACE)
async def register_workspace(sid: str, data: dict):
    """Store workspace_id -> sid in Redis with TTL; optionally store workspace metadata for GET /tunnel/workspace.
    Emit register_success with workspace_id so client can verify; register_failure if Redis unavailable."""
    if not await _require_auth(sid):
        return
    try:
        payload = WorkspaceRegisterPayload.model_validate(data)
    except Exception as e:
        logger.warning("[SocketServer] register_workspace invalid payload from sid=%s: %s", sid, e)
        await sio.emit("register_failure", {"reason": "invalid payload"}, room=sid, namespace=WORKSPACE_NAMESPACE)
        return
    auth_user_id = await _get_authenticated(sid)
    if auth_user_id is not None and payload.user_id != auth_user_id:
        logger.warning("[SocketServer] register_workspace user_id mismatch sid=%s", sid)
        await sio.emit("register_failure", {"reason": "user_id mismatch"}, room=sid, namespace=WORKSPACE_NAMESPACE)
        await sio.disconnect(sid, namespace=WORKSPACE_NAMESPACE)
        return
    redis_client = await _get_async_redis()
    if not redis_client:
        logger.warning("[SocketServer] Redis unavailable, cannot register workspace")
        await sio.emit("register_failure", {"reason": "redis_unavailable"}, room=sid, namespace=WORKSPACE_NAMESPACE)
        return
    key = _workspace_socket_key(payload.workspace_id)
    await redis_client.setex(key, WORKSPACE_SOCKET_TTL, sid)
    # Reverse map sid -> workspace_id for O(1) disconnect cleanup
    rev_key = _socket_workspace_key(sid)
    await redis_client.setex(rev_key, WORKSPACE_SOCKET_TTL, payload.workspace_id)
    try:
        from app.modules.tunnel.tunnel_service import get_tunnel_service
        get_tunnel_service().set_workspace_tunnel_record(
            workspace_id=payload.workspace_id,
            user_id=payload.user_id,
            repo_url=payload.repo_url,
            status="active",
        )
    except Exception as e:
        logger.debug("[SocketServer] Failed to store workspace record: %s", e)
    await sio.emit("register_success", {"workspace_id": payload.workspace_id}, room=sid, namespace=WORKSPACE_NAMESPACE)
    logger.info(
        "[SocketServer] Registered workspace_id=%s sid=%s repo_url=%s",
        payload.workspace_id,
        sid,
        payload.repo_url[:50] if payload.repo_url else "",
    )


@sio.event(namespace=WORKSPACE_NAMESPACE)
async def heartbeat(sid: str, data: dict):
    """Refresh TTL for workspace_id and for auth:sid so the connection stays authenticated."""
    if not await _require_auth(sid):
        return
    try:
        payload = WorkspaceHeartbeatPayload.model_validate(data)
    except Exception:
        return
    await _refresh_auth_ttl(sid)
    redis_client = await _get_async_redis()
    if not redis_client:
        return
    key = _workspace_socket_key(payload.workspace_id)
    await redis_client.expire(key, WORKSPACE_SOCKET_TTL)


@sio.event(namespace=WORKSPACE_NAMESPACE)
async def tool_response(sid: str, data: dict):
    """Publish response to Redis channel rpc:resp:{correlation_id} for the waiting RPC caller."""
    if not await _require_auth(sid):
        return
    try:
        payload = ToolResponseEvent.model_validate(data)
    except Exception as e:
        logger.warning("[SocketServer] tool_response invalid payload from sid=%s: %s", sid, e)
        return
    redis_client = await _get_async_redis()
    if not redis_client:
        logger.warning("[SocketServer] Redis unavailable, tool_response dropped")
        return
    channel = f"{RPC_RESPONSE_CHANNEL_PREFIX}{payload.correlation_id}"
    message = payload.model_dump(mode="json")
    await redis_client.publish(channel, json.dumps(message))
    logger.debug("[SocketServer] Published tool_response correlation_id=%s", payload.correlation_id)


@sio.event(namespace=WORKSPACE_NAMESPACE)
async def disconnect(sid: str, _reason: Optional[str] = None):
    """Remove auth state and workspace_id -> sid mapping using reverse map socket:workspace:{sid} for O(1) cleanup."""
    _cancel_auth_timeout(sid)
    await _clear_authenticated(sid)
    redis_client = await _get_async_redis()
    if not redis_client:
        return
    rev_key = _socket_workspace_key(sid)
    workspace_id = await redis_client.get(rev_key)
    if workspace_id:
        await redis_client.delete(_workspace_socket_key(workspace_id))
        await redis_client.delete(rev_key)
        logger.info("[SocketServer] Unregistered workspace_id=%s on disconnect", workspace_id)
