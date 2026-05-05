"""Environment-backed sandbox settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# Bumped from 0.1.0 to 0.2.1 in Phase 5: 0.1.0 was baked from the
# pre-migration Dockerfile and lacks `potpie-parse` on PATH, so any
# parse against a 0.1.0 sandbox crashes with `command not found`. The
# 0.2.x line installs the `potpie-sandbox` package (which carries the
# parser_runner shim) plus the parsing_rs wheel.
DEFAULT_AGENT_SNAPSHOT = "potpie/agent-sandbox:0.2.1"
DEFAULT_AGENT_DOCKER_IMAGE = "potpie/agent-sandbox:0.2.1"

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
    # Sandbox creation knobs. ``name_prefix`` is what the Daytona dashboard
    # displays — without it ops sees only the random sandbox id, which makes
    # incident response painful. The actual name appended at sandbox creation
    # time encodes the (user, project) pair so similar workloads cluster.
    daytona_sandbox_name_prefix: str = "potpie"
    # Daytona's default is "disabled" (negative) — archived sandboxes linger
    # forever. Setting a minute count means "delete N minutes after stopping".
    # ``None`` keeps Daytona's default; ``0`` deletes immediately on stop.
    daytona_auto_delete_minutes: int | None = None
    # Comma-separated CIDR allow-list applied at sandbox creation. Empty
    # string = no allow-list (open egress where the org tier permits it).
    # Tier-1 / Tier-2 orgs already restrict by default and override this;
    # the setting matters mainly on Tier-3+ where sandboxes default to
    # open internet, plus LLM-driven workloads that benefit from explicit
    # whitelisting (github.com, pypi.org, registry.npmjs.org).
    daytona_network_allow_list: str = ""
    daytona_network_block_all: bool = False
    # Snapshot resources (vCPU / GiB RAM / GiB disk). Daytona bakes these
    # into the snapshot at build time — a sandbox spawned from the snapshot
    # can't override them. Defaults are sized for medium repos running
    # pytest + linters; the SDK's own defaults (1 / 1 / 3) OOM on anything
    # beyond toy projects.
    daytona_snapshot_cpu: int = 2
    daytona_snapshot_memory_gb: int = 4
    daytona_snapshot_disk_gb: int = 10
    # Persist the bare clone in a Daytona Volume so it survives sandbox
    # destruction. Off by default — flip via env once the volume layout
    # is validated against the live infrastructure. With the flag on,
    # the per-user volume is mounted at ``daytona_volume_mount_path``
    # under a per-project subpath; the bare clone lives at
    # ``<mount>/.bare`` and the next sandbox for the same project
    # re-attaches the volume instead of re-cloning. Worktrees stay on
    # the sandbox's local fs because FUSE-backed volumes are slower
    # than node-local storage and worktrees are throwaway.
    daytona_use_volume_for_bare: bool = False
    # Volume names are org-scoped unique. We key on user, not project,
    # because the docs explicitly recommend one shared volume per
    # environment with per-tenant ``subpath`` values — keeps us under
    # the 100-volume-per-org cap as the project count grows.
    daytona_volume_name_prefix: str = "potpie-bare"
    # Mount path is fixed inside the sandbox so all bare paths are
    # predictable. Must NOT collide with system dirs Daytona rejects
    # (/proc, /sys, /etc, /bin, /sbin, /lib, /lib64, /boot, /dev).
    daytona_volume_mount_path: str = "/home/agent/work/.bare-cache"


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
        daytona_sandbox_name_prefix=os.getenv(
            "DAYTONA_SANDBOX_NAME_PREFIX", "potpie"
        ),
        daytona_auto_delete_minutes=_optional_int_env("DAYTONA_AUTO_DELETE_MINUTES"),
        daytona_network_allow_list=os.getenv("DAYTONA_NETWORK_ALLOW_LIST", ""),
        daytona_network_block_all=_bool_env("DAYTONA_NETWORK_BLOCK_ALL", False),
        daytona_snapshot_cpu=_int_env("DAYTONA_SNAPSHOT_CPU", 2),
        daytona_snapshot_memory_gb=_int_env("DAYTONA_SNAPSHOT_MEMORY_GB", 4),
        daytona_snapshot_disk_gb=_int_env("DAYTONA_SNAPSHOT_DISK_GB", 10),
        daytona_use_volume_for_bare=_bool_env(
            "DAYTONA_USE_VOLUME_FOR_BARE", False
        ),
        daytona_volume_name_prefix=os.getenv(
            "DAYTONA_VOLUME_NAME_PREFIX", "potpie-bare"
        ),
        daytona_volume_mount_path=os.getenv(
            "DAYTONA_VOLUME_MOUNT_PATH", "/home/agent/work/.bare-cache"
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


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _optional_int_env(name: str) -> int | None:
    """Like :func:`_int_env` but returns ``None`` for unset / blank.

    Used for knobs whose "leave the SDK default alone" branch needs a
    distinct sentinel — e.g. ``auto_delete_interval``, where ``0`` means
    "delete immediately" and ``-1`` means "never," so we can't reuse a
    numeric default.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None
