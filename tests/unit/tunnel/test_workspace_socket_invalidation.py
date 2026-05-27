"""Tests for stale workspace socket cleanup."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.unit


def test_invalidate_workspace_socket_deletes_forward_and_reverse_keys():
    from app.modules.tunnel.tunnel_service import TunnelService

    svc = TunnelService.__new__(TunnelService)
    svc.redis_client = MagicMock()
    svc._in_memory_workspace_tunnel_records = {}

    svc.redis_client.get.return_value = "sid-abc"
    svc.redis_client.delete.return_value = 1
    svc.redis_client.get.side_effect = [
        "sid-abc",  # forward key lookup
        json.dumps(
            {"user_id": "user-1", "repo_url": "owner/repo", "status": "active"}
        ),
    ]

    removed = svc.invalidate_workspace_socket("b2c2c1ae2cfffd87")

    assert removed is True
    assert svc.redis_client.delete.call_count == 2


def test_get_tunnel_url_false_after_invalidation():
    from app.modules.tunnel.tunnel_service import TunnelService

    svc = TunnelService.__new__(TunnelService)
    svc.redis_client = MagicMock()
    svc._in_memory_workspace_tunnel_records = {}
    svc.get_workspace_id = MagicMock(return_value="wid123")
    svc.redis_client.get.return_value = None
    svc.redis_client.delete.return_value = 1

    from app.modules.tunnel.socket_service import get_socket_service

    socket_svc = get_socket_service()
    socket_svc.is_workspace_online = MagicMock(return_value=False)

    svc.get_workspace_socket_status = lambda workspace_id: socket_svc.is_workspace_online(
        workspace_id
    )

    url = svc.get_tunnel_url("user-1", "conv-1", repository="owner/repo")
    assert url is None


@pytest.mark.asyncio
async def test_rpc_timeout_does_not_invalidate_workspace_socket(monkeypatch):
    from app.modules.tunnel import socket_service
    from app.modules.tunnel.tunnel_service import TunnelConnectionError

    svc = socket_service.WorkspaceSocketService.__new__(
        socket_service.WorkspaceSocketService
    )
    svc._get_socket_id = AsyncMock(return_value="sid-abc")

    pubsub = MagicMock()
    pubsub.subscribe = AsyncMock()
    pubsub.unsubscribe = AsyncMock()
    pubsub.close = AsyncMock()
    redis_client = MagicMock()
    redis_client.pubsub.return_value = pubsub
    svc._get_redis = AsyncMock(return_value=redis_client)

    svc._publish_tool_call = AsyncMock()
    invalidate = MagicMock()
    monkeypatch.setattr(
        socket_service, "_invalidate_workspace_after_rpc_failure", invalidate
    )

    async def fake_wait_for(awaitable, timeout):
        close = getattr(awaitable, "close", None)
        if close:
            close()
        raise socket_service.asyncio.TimeoutError()

    monkeypatch.setattr(socket_service.asyncio, "wait_for", fake_wait_for)

    with pytest.raises(TunnelConnectionError) as exc:
        await svc._execute_tool_call_impl(
            "b2c2c1ae2cfffd87",
            "/api/terminal/execute",
            {"command": "gcc -g add_numbers.c"},
            timeout=0.1,
        )

    assert exc.value.last_error == "timeout"
    invalidate.assert_not_called()


@pytest.mark.asyncio
async def test_backend_loop_mismatch_does_not_invalidate_workspace_socket(monkeypatch):
    from app.modules.tunnel import socket_service
    from app.modules.tunnel.tunnel_service import TunnelConnectionError

    svc = socket_service.WorkspaceSocketService.__new__(
        socket_service.WorkspaceSocketService
    )
    svc._get_socket_id = AsyncMock(return_value="sid-abc")

    pubsub = MagicMock()
    pubsub.subscribe = AsyncMock()
    pubsub.unsubscribe = AsyncMock()
    pubsub.close = AsyncMock()
    redis_client = MagicMock()
    redis_client.pubsub.return_value = pubsub
    svc._get_redis = AsyncMock(return_value=redis_client)
    svc._publish_tool_call = AsyncMock(
        side_effect=RuntimeError("got Future <Future pending> attached to a different loop")
    )

    invalidate = MagicMock()
    monkeypatch.setattr(
        socket_service, "_invalidate_workspace_after_rpc_failure", invalidate
    )

    with pytest.raises(TunnelConnectionError) as exc:
        await svc._execute_tool_call_impl(
            "b2c2c1ae2cfffd87",
            "/api/debug/start-session",
            {"program": "/tmp/zipmap_repro"},
            timeout=0.1,
        )

    assert exc.value.last_error == "backend_loop_mismatch"
    invalidate.assert_not_called()


@pytest.mark.asyncio
async def test_emit_failure_still_invalidates_workspace_socket(monkeypatch):
    from app.modules.tunnel import socket_service
    from app.modules.tunnel.tunnel_service import TunnelConnectionError

    svc = socket_service.WorkspaceSocketService.__new__(
        socket_service.WorkspaceSocketService
    )
    svc._get_socket_id = AsyncMock(return_value="sid-abc")

    pubsub = MagicMock()
    pubsub.subscribe = AsyncMock()
    pubsub.unsubscribe = AsyncMock()
    pubsub.close = AsyncMock()
    redis_client = MagicMock()
    redis_client.pubsub.return_value = pubsub
    svc._get_redis = AsyncMock(return_value=redis_client)
    svc._publish_tool_call = AsyncMock(side_effect=RuntimeError("redis unavailable"))

    invalidate = MagicMock()
    monkeypatch.setattr(
        socket_service, "_invalidate_workspace_after_rpc_failure", invalidate
    )

    with pytest.raises(TunnelConnectionError) as exc:
        await svc._execute_tool_call_impl(
            "b2c2c1ae2cfffd87",
            "/api/debug/start-session",
            {"program": "/tmp/zipmap_repro"},
            timeout=0.1,
        )

    assert exc.value.last_error == "emit_failed"
    invalidate.assert_called_once_with(
        "b2c2c1ae2cfffd87", reason="emit_failed", socket_id="sid-abc"
    )


@pytest.mark.asyncio
async def test_publish_tool_call_uses_fresh_redis_publisher(monkeypatch):
    from app.modules.tunnel import socket_service

    svc = socket_service.WorkspaceSocketService.__new__(
        socket_service.WorkspaceSocketService
    )
    published = []

    class Publisher:
        async def publish(self, channel, payload):
            published.append((channel, json.loads(payload)))

        async def aclose(self):
            pass

    manager = MagicMock()
    manager.redis_url = "redis://redis:6379/0"
    manager.channel = "socketio"
    manager.redis_options = {}
    manager.host_id = "socket-server-host"
    manager.json = json
    monkeypatch.setattr(socket_service.sio, "manager", manager)
    monkeypatch.setattr(socket_service.sio, "emit", AsyncMock())
    monkeypatch.setattr(
        socket_service.aioredis,
        "from_url",
        MagicMock(return_value=Publisher()),
    )

    await svc._publish_tool_call("sid-abc", {"correlation_id": "corr-1"})

    assert len(published) == 1
    channel, message = published[0]
    assert channel == "socketio"
    assert message["method"] == "emit"
    assert message["event"] == "tool_call"
    assert message["namespace"] == socket_service.WORKSPACE_NAMESPACE
    assert message["room"] == "sid-abc"
    assert message["data"] == [{"correlation_id": "corr-1"}]
    assert message["host_id"] != manager.host_id
    socket_service.sio.emit.assert_not_awaited()
