"""Compose sandbox providers and service."""

from __future__ import annotations

from dataclasses import dataclass

from sandbox.adapters.outbound.docker.runtime import DockerRuntimeProvider
from sandbox.adapters.outbound.file.json_store import JsonSandboxStore
from sandbox.adapters.outbound.local.git_workspace import LocalGitWorkspaceProvider
from sandbox.adapters.outbound.local.subprocess_runtime import LocalSubprocessRuntimeProvider
from sandbox.adapters.outbound.memory.locks import InMemoryLockManager
from sandbox.adapters.outbound.memory.store import InMemorySandboxStore
from sandbox.application.services.sandbox_service import SandboxService
from sandbox.bootstrap.settings import SandboxSettings, settings_from_env
from sandbox.domain.ports.locks import LockManager
from sandbox.domain.ports.runtimes import RuntimeProvider
from sandbox.domain.ports.stores import SandboxStore
from sandbox.domain.ports.workspaces import WorkspaceProvider


@dataclass
class SandboxContainer:
    workspace_provider: WorkspaceProvider
    runtime_provider: RuntimeProvider
    store: SandboxStore
    locks: LockManager
    service: SandboxService


def build_sandbox_container(
    settings: SandboxSettings | None = None,
    *,
    store: SandboxStore | None = None,
    locks: LockManager | None = None,
    workspace_provider: WorkspaceProvider | None = None,
    runtime_provider: RuntimeProvider | None = None,
) -> SandboxContainer:
    s = settings or settings_from_env()
    if workspace_provider is None:
        workspace_provider = _workspace_provider(s)
    if runtime_provider is None:
        runtime_provider = _runtime_provider(s, workspace_provider)
    metadata_store: SandboxStore
    if store is not None:
        metadata_store = store
    elif s.metadata_path:
        metadata_store = JsonSandboxStore(s.metadata_path)
    else:
        metadata_store = InMemorySandboxStore()
    lock_manager = locks or InMemoryLockManager()
    service = SandboxService(
        workspace_provider=workspace_provider,
        runtime_provider=runtime_provider,
        store=metadata_store,
        locks=lock_manager,
    )
    return SandboxContainer(
        workspace_provider=workspace_provider,
        runtime_provider=runtime_provider,
        store=metadata_store,
        locks=lock_manager,
        service=service,
    )


_DAYTONA_MISSING_HINT = (
    "Daytona SDK is not installed in this environment. Either:\n"
    "  • install it:  uv add daytona  (or:  pip install 'potpie-sandbox[daytona]')\n"
    "  • or set local-only providers in your .env:\n"
    "        SANDBOX_WORKSPACE_PROVIDER=local\n"
    "        SANDBOX_RUNTIME_PROVIDER=local_subprocess"
)


def _require_daytona_sdk() -> None:
    """Fail fast at bootstrap if the SDK is missing.

    Without this, the missing import surfaces deep inside the first agent
    tool call as a bare ``ModuleNotFoundError``, which is hard to diagnose.
    """
    try:
        import daytona  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(_DAYTONA_MISSING_HINT) from exc


def _workspace_provider(settings: SandboxSettings) -> WorkspaceProvider:
    if settings.provider == "local":
        return LocalGitWorkspaceProvider(settings.repos_base_path)
    if settings.provider == "daytona":
        _require_daytona_sdk()
        from sandbox.adapters.outbound.daytona.provider import DaytonaWorkspaceProvider

        return DaytonaWorkspaceProvider(
            snapshot=settings.daytona_snapshot,
            workspace_root=settings.daytona_workspace_root,
            snapshot_dockerfile=settings.daytona_snapshot_dockerfile,
            snapshot_build_timeout_s=settings.daytona_snapshot_build_timeout_s,
            snapshot_heartbeat_s=settings.daytona_snapshot_heartbeat_s,
        )
    raise ValueError(f"Unsupported SANDBOX_WORKSPACE_PROVIDER={settings.provider!r}")


def _runtime_provider(
    settings: SandboxSettings, workspace_provider: WorkspaceProvider
) -> RuntimeProvider:
    if settings.runtime == "local_subprocess":
        return LocalSubprocessRuntimeProvider(allow_write=settings.local_allow_write)
    if settings.runtime == "docker":
        return DockerRuntimeProvider()
    if settings.runtime == "daytona":
        _require_daytona_sdk()
        from sandbox.adapters.outbound.daytona.provider import (
            DaytonaRuntimeProvider,
            DaytonaWorkspaceProvider,
        )

        if not isinstance(workspace_provider, DaytonaWorkspaceProvider):
            raise ValueError("Daytona runtime requires Daytona workspace provider")
        return DaytonaRuntimeProvider(workspace_provider)
    raise ValueError(f"Unsupported SANDBOX_RUNTIME_PROVIDER={settings.runtime!r}")

