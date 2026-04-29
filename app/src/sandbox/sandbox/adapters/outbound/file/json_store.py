"""Small JSON metadata store for local development and single-node deployments."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from sandbox.adapters.outbound.memory.store import InMemorySandboxStore
from sandbox.domain.models import (
    Mount,
    NetworkMode,
    RepoIdentity,
    ResourceHints,
    Runtime,
    RuntimeSpec,
    RuntimeState,
    Workspace,
    WorkspaceLocation,
    WorkspaceMode,
    WorkspaceRequest,
    WorkspaceState,
    WorkspaceStorageKind,
)


class JsonSandboxStore(InMemorySandboxStore):
    """Persist sandbox metadata to one JSON file.

    This is not a replacement for Postgres in production. It is useful for local
    sandbox development where process restarts should remember workspace/runtime
    ids without adding a database dependency.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = Path(path)
        self._persist_lock = asyncio.Lock()
        self._loaded = False

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        await self._ensure_loaded()
        return await super().get_workspace(workspace_id)

    async def find_workspace_by_key(self, key: str) -> Workspace | None:
        await self._ensure_loaded()
        return await super().find_workspace_by_key(key)

    async def save_workspace(self, workspace: Workspace) -> None:
        await self._ensure_loaded()
        await super().save_workspace(workspace)
        await self._flush()

    async def delete_workspace(self, workspace_id: str) -> None:
        await self._ensure_loaded()
        await super().delete_workspace(workspace_id)
        await self._flush()

    async def list_workspaces(self) -> list[Workspace]:
        await self._ensure_loaded()
        return await super().list_workspaces()

    async def get_runtime(self, runtime_id: str) -> Runtime | None:
        await self._ensure_loaded()
        return await super().get_runtime(runtime_id)

    async def find_runtime_by_workspace(
        self, workspace_id: str, backend_kind: str | None = None
    ) -> Runtime | None:
        await self._ensure_loaded()
        return await super().find_runtime_by_workspace(workspace_id, backend_kind)

    async def save_runtime(self, runtime: Runtime) -> None:
        await self._ensure_loaded()
        await super().save_runtime(runtime)
        await self._flush()

    async def delete_runtime(self, runtime_id: str) -> None:
        await self._ensure_loaded()
        await super().delete_runtime(runtime_id)
        await self._flush()

    async def list_runtimes(self) -> list[Runtime]:
        await self._ensure_loaded()
        return await super().list_runtimes()

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._persist_lock:
            if self._loaded:
                return
            if not self._path.exists():
                self._loaded = True
                return
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for item in raw.get("workspaces", []):
                await super().save_workspace(_workspace_from_json(item))
            for item in raw.get("runtimes", []):
                await super().save_runtime(_runtime_from_json(item))
            self._loaded = True

    async def _flush(self) -> None:
        async with self._persist_lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            workspaces = [_json_ready(w) for w in await super().list_workspaces()]
            runtimes = [_json_ready(r) for r in await super().list_runtimes()]
            payload = {"workspaces": workspaces, "runtimes": runtimes}
            tmp = self._path.with_suffix(f"{self._path.suffix}.tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self._path)


def _json_ready(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {
            k: _json_ready(v)
            for k, v in obj.items()
            if k != "auth_token"
        }
    if isinstance(obj, list | tuple):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _location(raw: dict[str, Any]) -> WorkspaceLocation:
    return WorkspaceLocation(
        kind=WorkspaceStorageKind(raw["kind"]),
        local_path=raw.get("local_path"),
        remote_path=raw.get("remote_path"),
        docker_volume=raw.get("docker_volume"),
        backend_workspace_id=raw.get("backend_workspace_id"),
    )


def _request(raw: dict[str, Any]) -> WorkspaceRequest:
    return WorkspaceRequest(
        user_id=raw["user_id"],
        project_id=raw["project_id"],
        repo=RepoIdentity(**raw["repo"]),
        base_ref=raw["base_ref"],
        mode=WorkspaceMode(raw.get("mode", WorkspaceMode.EDIT.value)),
        conversation_id=raw.get("conversation_id"),
        task_id=raw.get("task_id"),
        branch_name=raw.get("branch_name"),
        create_branch=bool(raw.get("create_branch", False)),
        pinned=bool(raw.get("pinned", False)),
    )


def _workspace_from_json(raw: dict[str, Any]) -> Workspace:
    return Workspace(
        id=raw["id"],
        key=raw["key"],
        repo_cache_id=raw.get("repo_cache_id"),
        request=_request(raw["request"]),
        location=_location(raw["location"]),
        backend_kind=raw["backend_kind"],
        state=WorkspaceState(raw.get("state", WorkspaceState.READY.value)),
        dirty=bool(raw.get("dirty", False)),
        pinned_until=_dt(raw.get("pinned_until")),
        last_used_at=_dt(raw.get("last_used_at")) or datetime.utcnow(),
        size_bytes=raw.get("size_bytes"),
        created_at=_dt(raw.get("created_at")) or datetime.utcnow(),
        updated_at=_dt(raw.get("updated_at")) or datetime.utcnow(),
        metadata=dict(raw.get("metadata") or {}),
    )


def _runtime_from_json(raw: dict[str, Any]) -> Runtime:
    spec_raw = raw["spec"]
    mounts = tuple(Mount(**m) for m in spec_raw.get("mounts", []))
    resources = (
        ResourceHints(**spec_raw["resources"])
        if spec_raw.get("resources") is not None
        else None
    )
    spec = RuntimeSpec(
        image=spec_raw["image"],
        workdir=spec_raw["workdir"],
        mounts=mounts,
        env=dict(spec_raw.get("env") or {}),
        resources=resources,
        network=NetworkMode(spec_raw.get("network", NetworkMode.LIMITED.value)),
        placement=dict(spec_raw.get("placement") or {}),
        labels=dict(spec_raw.get("labels") or {}),
    )
    return Runtime(
        id=raw["id"],
        workspace_id=raw["workspace_id"],
        backend_kind=raw["backend_kind"],
        backend_runtime_id=raw.get("backend_runtime_id"),
        spec=spec,
        state=RuntimeState(raw.get("state", RuntimeState.RUNNING.value)),
        last_started_at=_dt(raw.get("last_started_at")),
        last_used_at=_dt(raw.get("last_used_at")) or datetime.utcnow(),
        expires_at=_dt(raw.get("expires_at")),
        created_at=_dt(raw.get("created_at")) or datetime.utcnow(),
        updated_at=_dt(raw.get("updated_at")) or datetime.utcnow(),
    )

