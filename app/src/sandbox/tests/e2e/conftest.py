"""Shared fixtures for sandbox end-to-end tests.

These fixtures build a real `SandboxService` against a real local git repo so
the tests exercise the full flow: workspace provisioning, runtime attach, exec,
and teardown. Tests select a runtime backend (local subprocess, docker,
daytona) via a parametrized fixture or the `runtime_backend` marker.
"""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import pytest


# Daytona's docker-compose stack emits toolbox URLs under `proxy.localhost`,
# which macOS does not auto-resolve to 127.0.0.1. Daytona ships a sudo-required
# dnsmasq script for this; here we keep the patch in-process so tests run with
# zero host changes. Honors a real /etc/hosts entry if one already exists.
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

from sandbox.adapters.outbound.docker.runtime import DockerRuntimeProvider
from sandbox.adapters.outbound.file.json_store import JsonSandboxStore
from sandbox.adapters.outbound.local.git_workspace import LocalGitWorkspaceProvider
from sandbox.adapters.outbound.local.subprocess_runtime import LocalSubprocessRuntimeProvider
from sandbox.adapters.outbound.memory.locks import InMemoryLockManager
from sandbox.application.services.sandbox_service import SandboxService


def _run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"{cmd}: {result.stderr}"


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    """Create a tiny on-disk git repo with one commit on `main`.

    Used as the `repo_url` for workspace requests so the tests do not require
    network access or GitHub credentials.
    """
    repo = tmp_path / "source"
    repo.mkdir()
    _run(["git", "init", "-b", "main"], repo)
    _run(["git", "config", "user.email", "e2e@example.com"], repo)
    _run(["git", "config", "user.name", "E2E User"], repo)
    (repo / "README.md").write_text("hello e2e\n", encoding="utf-8")
    (repo / "app.py").write_text("print('alive')\n", encoding="utf-8")
    _run(["git", "add", "-A"], repo)
    _run(["git", "commit", "-m", "initial"], repo)
    return repo


@pytest.fixture
def repos_base(tmp_path: Path) -> Path:
    base = tmp_path / ".repos"
    base.mkdir()
    return base


@pytest.fixture
def metadata_path(tmp_path: Path) -> Path:
    return tmp_path / "metadata.json"


@pytest.fixture
def workspace_provider(repos_base: Path) -> LocalGitWorkspaceProvider:
    return LocalGitWorkspaceProvider(repos_base)


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=20, check=False
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def _daytona_dashboard_up(url: str) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _daytona_configured() -> bool:
    """True if either an explicit API key is provided, or the dev stack is reachable."""
    if os.getenv("DAYTONA_API_KEY"):
        return True
    dashboard = os.getenv("DAYTONA_DASHBOARD_URL", f"http://localhost:{os.getenv('DAYTONA_DASHBOARD_PORT', '3010')}")
    return _daytona_dashboard_up(dashboard)


def docker_image() -> str:
    return os.getenv("SANDBOX_DOCKER_IMAGE", "busybox:latest")


@pytest.fixture
def local_service(
    workspace_provider: LocalGitWorkspaceProvider, metadata_path: Path
) -> SandboxService:
    return SandboxService(
        workspace_provider=workspace_provider,
        runtime_provider=LocalSubprocessRuntimeProvider(allow_write=True),
        store=JsonSandboxStore(metadata_path),
        locks=InMemoryLockManager(),
    )


@pytest.fixture
def docker_service(
    workspace_provider: LocalGitWorkspaceProvider, metadata_path: Path
) -> SandboxService:
    if not _docker_available():
        pytest.skip("Docker daemon not available")
    return SandboxService(
        workspace_provider=workspace_provider,
        runtime_provider=DockerRuntimeProvider(name_prefix="potpie-sandbox-e2e"),
        store=JsonSandboxStore(metadata_path),
        locks=InMemoryLockManager(),
    )


def _daytona_credentials() -> tuple[str, str, str | None]:
    """Return ``(api_url, api_key, organization_id)`` for the local Daytona stack.

    Honors explicit env vars first; otherwise mints a key by scripting the dex
    OIDC password login against the docker-compose dev stack.
    """
    dashboard_default = (
        f"http://localhost:{os.getenv('DAYTONA_DASHBOARD_PORT', '3010')}"
    )
    api_url = os.getenv("DAYTONA_API_URL", f"{dashboard_default}/api")
    api_key = os.getenv("DAYTONA_API_KEY")
    org_id = os.getenv("DAYTONA_ORGANIZATION_ID")
    if api_key:
        return api_url, api_key, org_id
    import importlib.util

    helper = (
        Path(__file__).resolve().parents[2] / "scripts" / "daytona_local.py"
    )
    spec = importlib.util.spec_from_file_location("_potpie_daytona_local", helper)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    dashboard = os.getenv("DAYTONA_DASHBOARD_URL", f"http://localhost:{os.getenv('DAYTONA_DASHBOARD_PORT', '3010')}")
    api_key, org_id = module.mint_dev_api_key(dashboard=dashboard)
    return api_url, api_key, org_id


@pytest.fixture(scope="session")
def daytona_credentials() -> tuple[str, str, str | None]:
    if not _daytona_configured():
        pytest.skip(
            "Daytona dev stack not reachable at DAYTONA_DASHBOARD_URL / "
            "DAYTONA_API_KEY not set"
        )
    return _daytona_credentials()


@pytest.fixture
def daytona_service(
    metadata_path: Path, daytona_credentials: tuple[str, str, str | None]
) -> SandboxService:
    from daytona import Daytona, DaytonaConfig

    from sandbox.adapters.outbound.daytona.provider import (
        DaytonaRuntimeProvider,
        DaytonaWorkspaceProvider,
    )

    api_url, api_key, _org_id = daytona_credentials
    # Note: organization_id is only used for JWT auth; passing it alongside an
    # API key flips the SDK into JWT mode and the request fails. The API key
    # is already scoped to its owning org.
    config = DaytonaConfig(api_url=api_url, api_key=api_key)

    workspace_provider = DaytonaWorkspaceProvider(
        client_factory=lambda: Daytona(config),
        snapshot=os.getenv("DAYTONA_SNAPSHOT") or "potpie/agent-sandbox:0.1.0",
        workspace_root=os.getenv("DAYTONA_WORKSPACE_ROOT", "/home/agent/work"),
    )
    return SandboxService(
        workspace_provider=workspace_provider,
        runtime_provider=DaytonaRuntimeProvider(workspace_provider),
        store=JsonSandboxStore(metadata_path),
        locks=InMemoryLockManager(),
    )
