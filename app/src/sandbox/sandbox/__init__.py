"""Potpie sandbox core.

This package is intentionally independent from the legacy app modules. It uses
ports for workspace persistence and runtime execution so local `.repos`,
Docker-backed execution, and Daytona-managed sandboxes can share one service
contract.

Most callers should import from this top-level module:

    from sandbox import SandboxClient, WorkspaceHandle, WorkspaceMode

The lower-level domain types (`WorkspaceRequest`, `ExecRequest`, …) are still
exported for advanced cases (custom adapters, tests) but the client surface
should suffice for product code.
"""

from sandbox.adapters.outbound.memory.eviction import NoOpEvictionPolicy
from sandbox.api import (
    FileEntry,
    GitStatus,
    Hit,
    SandboxClient,
    WorkspaceHandle,
)
from sandbox.api.client import SandboxOpError
from sandbox.application.services.sandbox_service import SandboxService
from sandbox.bootstrap.container import SandboxContainer, build_sandbox_container
from sandbox.bootstrap.settings import SandboxSettings, settings_from_env
from sandbox.domain.models import (
    Capabilities,
    CommandKind,
    ExecRequest,
    ExecResult,
    NetworkMode,
    PullRequest,
    PullRequestRequest,
    RepoCache,
    RepoCacheRequest,
    RepoIdentity,
    RuntimeRequest,
    WorkspaceMode,
    WorkspaceRequest,
)
from sandbox.domain.ports.eviction import EvictionPolicy, EvictionResult
from sandbox.domain.ports.git_platform import GitPlatformProvider
from sandbox.domain.ports.repos import RepoCacheProvider

__all__ = [
    "Capabilities",
    "CommandKind",
    "EvictionPolicy",
    "EvictionResult",
    "ExecRequest",
    "ExecResult",
    "FileEntry",
    "GitPlatformProvider",
    "GitStatus",
    "Hit",
    "NetworkMode",
    "NoOpEvictionPolicy",
    "PullRequest",
    "PullRequestRequest",
    "RepoCache",
    "RepoCacheProvider",
    "RepoCacheRequest",
    "RepoIdentity",
    "RuntimeRequest",
    "SandboxClient",
    "SandboxContainer",
    "SandboxOpError",
    "SandboxService",
    "SandboxSettings",
    "WorkspaceHandle",
    "WorkspaceMode",
    "WorkspaceRequest",
    "build_sandbox_container",
    "settings_from_env",
]
