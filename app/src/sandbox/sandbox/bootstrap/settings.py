"""Environment-backed sandbox settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_AGENT_SNAPSHOT = "potpie/agent-sandbox:0.1.0"
DEFAULT_AGENT_DOCKER_IMAGE = "potpie/agent-sandbox:0.1.0"

# Bundled Dockerfile that defines the agent-sandbox snapshot. Resolved relative
# to this file so the path is correct whether the package is run from source
# or installed into a venv.
_DEFAULT_AGENT_DOCKERFILE = (
    Path(__file__).resolve().parents[2] / "images" / "agent-sandbox" / "Dockerfile"
)


@dataclass(frozen=True, slots=True)
class SandboxSettings:
    provider: str = "local"
    runtime: str = "local_subprocess"
    repos_base_path: str = ".repos"
    metadata_path: str | None = None
    docker_image: str = DEFAULT_AGENT_DOCKER_IMAGE
    local_allow_write: bool = False
    daytona_snapshot: str | None = DEFAULT_AGENT_SNAPSHOT
    daytona_workspace_root: str = "/home/agent/work"
    daytona_snapshot_dockerfile: str | None = None
    daytona_snapshot_build_timeout_s: float = 20 * 60
    daytona_snapshot_heartbeat_s: float = 15


def settings_from_env() -> SandboxSettings:
    base = os.getenv("SANDBOX_REPOS_BASE_PATH", ".repos")
    metadata = os.getenv("SANDBOX_METADATA_PATH")
    if metadata is None:
        metadata = str(Path(base) / ".sandbox" / "metadata.json")
    dockerfile = os.getenv("DAYTONA_SNAPSHOT_DOCKERFILE")
    if dockerfile is None and _DEFAULT_AGENT_DOCKERFILE.exists():
        dockerfile = str(_DEFAULT_AGENT_DOCKERFILE)
    return SandboxSettings(
        provider=os.getenv("SANDBOX_WORKSPACE_PROVIDER", "local").strip().lower(),
        runtime=os.getenv("SANDBOX_RUNTIME_PROVIDER", "local_subprocess").strip().lower(),
        repos_base_path=base,
        metadata_path=metadata,
        docker_image=os.getenv("SANDBOX_DOCKER_IMAGE", DEFAULT_AGENT_DOCKER_IMAGE),
        local_allow_write=_bool_env("SANDBOX_LOCAL_ALLOW_WRITE", False),
        daytona_snapshot=os.getenv("DAYTONA_SNAPSHOT") or DEFAULT_AGENT_SNAPSHOT,
        daytona_workspace_root=os.getenv(
            "DAYTONA_WORKSPACE_ROOT", "/home/agent/work"
        ),
        daytona_snapshot_dockerfile=dockerfile,
        daytona_snapshot_build_timeout_s=_float_env(
            "DAYTONA_SNAPSHOT_BUILD_TIMEOUT_S", 20 * 60
        ),
        daytona_snapshot_heartbeat_s=_float_env(
            "DAYTONA_SNAPSHOT_HEARTBEAT_S", 15
        ),
    )


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default

