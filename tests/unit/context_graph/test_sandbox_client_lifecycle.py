"""SandboxClient.detach_repo_from_pot / destroy_pot_sandbox — store-driven cleanup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.unit


@dataclass
class _Workspace:
    id: str
    user_id: str
    project_id: str
    repo_name: str

    @property
    def request(self) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            user_id=self.user_id,
            project_id=self.project_id,
            repo=SimpleNamespace(repo_name=self.repo_name),
        )


@dataclass
class _FakeStore:
    workspaces: list[_Workspace] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    async def list_workspaces(self) -> list[_Workspace]:
        return list(self.workspaces)


@dataclass
class _FakeService:
    store: _FakeStore
    destroy_calls: list[str] = field(default_factory=list)
    destroy_pot_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def _store(self) -> _FakeStore:
        return self.store

    async def destroy_workspace(self, workspace_id: str) -> None:
        self.destroy_calls.append(workspace_id)
        self.store.workspaces = [w for w in self.store.workspaces if w.id != workspace_id]

    async def destroy_pot_container(self, **kwargs: Any) -> dict[str, int]:
        self.destroy_pot_calls.append(kwargs)
        return {"workspaces": len(self.destroy_calls), "repo_caches": 0}


class _FakeClient:
    """Tiny stand-in that exposes ``_service`` like the real SandboxClient."""

    def __init__(self, service: _FakeService) -> None:
        self._service = service


@pytest.mark.asyncio
class TestDetachRepoFromPot:
    async def test_destroys_only_matching_workspaces(self) -> None:
        # Import real method off the real class so we test the canonical impl.
        from sandbox.api.client import SandboxClient

        store = _FakeStore(
            workspaces=[
                _Workspace(id="ws-1", user_id="u1", project_id="pot-1", repo_name="a/x"),
                _Workspace(id="ws-2", user_id="u1", project_id="pot-1", repo_name="b/y"),
                _Workspace(id="ws-3", user_id="u1", project_id="pot-2", repo_name="a/x"),
                _Workspace(id="ws-4", user_id="u2", project_id="pot-1", repo_name="a/x"),
            ]
        )
        service = _FakeService(store=store)
        client = _FakeClient(service=service)  # type: ignore[assignment]

        removed = await SandboxClient.detach_repo_from_pot(
            client,  # type: ignore[arg-type]
            user_id="u1",
            project_id="pot-1",
            repo="a/x",
        )
        assert removed == 1
        assert service.destroy_calls == ["ws-1"]

    async def test_destroy_failures_are_skipped(self) -> None:
        from sandbox.api.client import SandboxClient

        store = _FakeStore(
            workspaces=[
                _Workspace(id="ws-1", user_id="u", project_id="p", repo_name="r/r"),
                _Workspace(id="ws-2", user_id="u", project_id="p", repo_name="r/r"),
            ]
        )
        service = _FakeService(store=store)

        async def _maybe_destroy(workspace_id):
            if workspace_id == "ws-1":
                raise RuntimeError("backend dead")
            service.destroy_calls.append(workspace_id)

        service.destroy_workspace = _maybe_destroy  # type: ignore[assignment]
        client = _FakeClient(service=service)

        removed = await SandboxClient.detach_repo_from_pot(
            client,  # type: ignore[arg-type]
            user_id="u",
            project_id="p",
            repo="r/r",
        )
        # ws-2 still got destroyed even though ws-1 raised.
        assert removed == 1
        assert service.destroy_calls == ["ws-2"]


@pytest.mark.asyncio
class TestDestroyPotSandbox:
    async def test_delegates_to_service(self) -> None:
        from sandbox.api.client import SandboxClient

        service = _FakeService(store=_FakeStore())
        client = _FakeClient(service=service)
        result = await SandboxClient.destroy_pot_sandbox(
            client,  # type: ignore[arg-type]
            user_id="u1",
            project_id="pot-1",
            delete_repo_caches=True,
        )
        assert service.destroy_pot_calls == [
            {"user_id": "u1", "project_id": "pot-1", "delete_repo_caches": True}
        ]
        assert result["workspaces"] == 0
