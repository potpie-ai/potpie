"""Shared fixtures for sandbox integration tests.

These tests build a real ``SandboxClient`` against either a hermetic
on-disk fixture repo (local provider) or a live Daytona stack (Daytona
provider). The local fixtures need no network and run by default; the
Daytona ones are skipped unless ``DAYTONA_API_URL`` + ``DAYTONA_API_KEY``
are set.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import AsyncIterator, Iterator

import pytest


# Daytona's docker-compose stack uses ``proxy.localhost`` toolbox URLs that
# macOS doesn't auto-resolve. Patch in-process so live tests need no host
# changes; harmless when the daytona stack isn't in use.
_LOCAL_HOSTS = {"proxy.localhost"}


def _install_localhost_resolver() -> None:
    real = socket.getaddrinfo

    def patched(host, *args, **kwargs):
        if isinstance(host, str) and (
            host in _LOCAL_HOSTS or host.endswith(".proxy.localhost")
        ):
            try:
                return real("127.0.0.1", *args, **kwargs)
            except socket.gaierror:
                pass
        return real(host, *args, **kwargs)

    socket.getaddrinfo = patched  # type: ignore[assignment]


_install_localhost_resolver()


def _git_available() -> bool:
    return shutil.which("git") is not None


def _run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"{cmd}: {result.stderr}"


@pytest.fixture
def repos_base(tmp_path: Path) -> Path:
    """Empty `.repos/` root for the test's RepoCache + worktrees."""
    base = tmp_path / ".repos"
    base.mkdir()
    return base


@pytest.fixture
def metadata_path(tmp_path: Path) -> Path:
    """Path the JSON sandbox store flushes to."""
    return tmp_path / "metadata.json"


@pytest.fixture
def upstream_repo(tmp_path: Path) -> Path:
    """Single-commit hermetic git repo used as the workspace's upstream.

    Lives outside ``.repos/`` so it acts as a real "remote" — the cache
    clones from this path with ``git clone --bare``. We pre-create
    ``main`` with a couple of files so the agent flow has something to
    read, write, and commit.
    """
    if not _git_available():
        pytest.skip("git CLI not on PATH")
    repo = tmp_path / "upstream"
    repo.mkdir()
    _run(["git", "init", "-b", "main"], repo)
    _run(["git", "config", "user.email", "fixture@example.com"], repo)
    _run(["git", "config", "user.name", "Fixture"], repo)
    (repo / "README.md").write_text("hello sandbox\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('alive')\n", encoding="utf-8")
    _run(["git", "add", "-A"], repo)
    _run(["git", "commit", "-m", "initial"], repo)
    # Receiving pushes into a non-bare repo with the working tree on
    # the same branch is rejected by default; flip the receive policy
    # so the test's ``client.push`` doesn't hit that wall if exercised.
    _run(
        ["git", "config", "receive.denyCurrentBranch", "updateInstead"],
        repo,
    )
    return repo


@pytest.fixture
async def local_client(
    repos_base: Path, metadata_path: Path
) -> AsyncIterator["SandboxClient"]:  # type: ignore[name-defined]
    """A real ``SandboxClient`` wired to the local provider.

    ``local_allow_write=True`` so the subprocess runtime accepts WRITE
    commands — the read-only default exists to keep this backend safe in
    production where Docker / Daytona are preferred for write workloads.
    """
    from sandbox import SandboxClient, SandboxSettings, build_sandbox_container

    settings = SandboxSettings(
        provider="local",
        runtime="local_subprocess",
        repos_base_path=str(repos_base),
        metadata_path=str(metadata_path),
        local_allow_write=True,
    )
    container = build_sandbox_container(settings)
    yield SandboxClient.from_container(container)


@pytest.fixture
def daytona_env_present() -> bool:
    return bool(os.getenv("DAYTONA_API_URL") and os.getenv("DAYTONA_API_KEY"))


@pytest.fixture
async def daytona_client(
    daytona_env_present: bool, metadata_path: Path
) -> AsyncIterator["SandboxClient"]:  # type: ignore[name-defined]
    """Real ``SandboxClient`` wired to a live Daytona stack.

    Each test gets its own metadata.json so workspaces created in one
    test aren't visible to another via the persisted store. The actual
    Daytona sandboxes still leak across tests until one of them deletes
    them — every test that creates a sandbox is expected to clean up in
    a ``finally`` block.
    """
    if not daytona_env_present:
        pytest.skip(
            "DAYTONA_API_URL and DAYTONA_API_KEY must be set to run Daytona "
            "integration tests"
        )

    from daytona import Daytona, DaytonaConfig

    from sandbox import SandboxClient
    from sandbox.adapters.outbound.daytona.provider import (
        DaytonaRuntimeProvider,
        DaytonaWorkspaceProvider,
    )
    from sandbox.adapters.outbound.file.json_store import JsonSandboxStore
    from sandbox.adapters.outbound.memory.locks import InMemoryLockManager
    from sandbox.application.services.sandbox_service import SandboxService
    from sandbox.bootstrap.container import SandboxContainer
    from sandbox.adapters.outbound.memory.eviction import NoOpEvictionPolicy

    config = DaytonaConfig(
        api_url=os.environ["DAYTONA_API_URL"],
        api_key=os.environ["DAYTONA_API_KEY"],
    )
    workspace_provider = DaytonaWorkspaceProvider(
        client_factory=lambda: Daytona(config),
        snapshot=os.getenv("DAYTONA_SNAPSHOT") or "potpie/agent-sandbox:0.1.0",
        workspace_root=os.getenv("DAYTONA_WORKSPACE_ROOT", "/home/agent/work"),
    )
    runtime_provider = DaytonaRuntimeProvider(workspace_provider)
    store = JsonSandboxStore(metadata_path)
    locks = InMemoryLockManager()
    service = SandboxService(
        workspace_provider=workspace_provider,
        runtime_provider=runtime_provider,
        store=store,
        locks=locks,
        # Daytona has no separate RepoCacheProvider port today (P4 in
        # the roadmap promotes it). Pass None so the service skips the
        # ensure_repo_cache step inside acquire_session instead of
        # raising RuntimeError.
        repo_cache_provider=None,
    )
    container = SandboxContainer(
        workspace_provider=workspace_provider,
        runtime_provider=runtime_provider,
        store=store,
        locks=locks,
        service=service,
        eviction=NoOpEvictionPolicy(),
        repo_cache_provider=None,
    )
    yield SandboxClient.from_container(container)
