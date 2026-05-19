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

_VALID_PROVIDERS = frozenset({"local", "daytona"})
_VALID_RUNTIMES = frozenset({"local_subprocess", "docker", "daytona"})
_VALID_EVICTION_POLICIES = frozenset({"tiered", "noop"})

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
    # Local subprocess explicitly *is not* a security boundary (see
    # ``LocalSubprocessRuntimeProvider`` docstring): a caller who selects
    # ``runtime=local_subprocess`` has already accepted that. Defaulting
    # writes off would silently break the edit/commit/push flow without
    # adding any real safety. Read-only callers can flip this back via
    # ``SANDBOX_LOCAL_ALLOW_WRITE=false``.
    local_allow_write: bool = True
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
    # ---- Storage limit + eviction --------------------------------------
    # Tiered eviction is the default (bounded disk out of the box). Set
    # ``noop`` to opt out (tests/standalone). ``storage_limit_gb`` is an
    # absolute budget for the local ``.repos``; when unset the budget is
    # ``storage_disk_fraction`` of the filesystem total — pressure is
    # always our-footprint / budget, never whole-disk fill, so an empty
    # cache reads ~0 and never flaps. High/low water are the evict
    # trigger and the sweep stop target. ``evict_dirty`` gates the
    # last-resort tier that drops un-pushed edits. ``max_sandboxes_per_
    # user`` is the Daytona per-user count cap the inspector reports on
    # (``None`` ⇒ count scope not enforced; Daytona's own caps still
    # apply). ``storage_sweep_interval_secs`` is read by the Celery beat
    # entry, kept here so all knobs live in one place.
    eviction_policy: str = "tiered"
    storage_limit_gb: float | None = None
    storage_disk_fraction: float = 0.5
    storage_high_water: float = 0.85
    storage_low_water: float = 0.70
    evict_dirty: bool = True
    daytona_max_sandboxes_per_user: int | None = None
    storage_sweep_interval_secs: float = 300.0

    def __post_init__(self) -> None:
        """Fail fast on misconfiguration.

        Previously an unknown ``SANDBOX_WORKSPACE_PROVIDER`` /
        ``SANDBOX_RUNTIME_PROVIDER`` surfaced only when the bootstrap
        tried to materialise the adapter — deep inside the first agent
        call inside a Celery task. Validating here turns it into a
        process-start error with a clear message.
        """
        if self.provider not in _VALID_PROVIDERS:
            raise ValueError(
                f"Unsupported SANDBOX_WORKSPACE_PROVIDER={self.provider!r}; "
                f"expected one of {sorted(_VALID_PROVIDERS)}"
            )
        if self.runtime not in _VALID_RUNTIMES:
            raise ValueError(
                f"Unsupported SANDBOX_RUNTIME_PROVIDER={self.runtime!r}; "
                f"expected one of {sorted(_VALID_RUNTIMES)}"
            )
        # Daytona's runtime is wired to its workspace provider (the
        # sandbox is the runtime), so mixing it with the local
        # workspace provider can't work — flag it early.
        if self.runtime == "daytona" and self.provider != "daytona":
            raise ValueError(
                "SANDBOX_RUNTIME_PROVIDER=daytona requires "
                "SANDBOX_WORKSPACE_PROVIDER=daytona (the Daytona runtime "
                "lives inside the Daytona workspace)"
            )
        if self.eviction_policy not in _VALID_EVICTION_POLICIES:
            raise ValueError(
                f"Unsupported SANDBOX_EVICTION_POLICY={self.eviction_policy!r}; "
                f"expected one of {sorted(_VALID_EVICTION_POLICIES)}"
            )
        if not 0.0 < self.storage_low_water < self.storage_high_water <= 1.0:
            raise ValueError(
                "require 0 < SANDBOX_STORAGE_LOW_WATER < "
                "SANDBOX_STORAGE_HIGH_WATER <= 1; got "
                f"low={self.storage_low_water} high={self.storage_high_water}"
            )
        if not 0.0 < self.storage_disk_fraction <= 1.0:
            raise ValueError(
                "require 0 < SANDBOX_STORAGE_DISK_FRACTION <= 1; got "
                f"{self.storage_disk_fraction}"
            )


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
        local_allow_write=_bool_env("SANDBOX_LOCAL_ALLOW_WRITE", True),
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
        eviction_policy=os.getenv("SANDBOX_EVICTION_POLICY", "tiered")
        .strip()
        .lower(),
        storage_limit_gb=_optional_float_env("SANDBOX_STORAGE_LIMIT_GB"),
        storage_disk_fraction=_float_env("SANDBOX_STORAGE_DISK_FRACTION", 0.5),
        storage_high_water=_float_env("SANDBOX_STORAGE_HIGH_WATER", 0.85),
        storage_low_water=_float_env("SANDBOX_STORAGE_LOW_WATER", 0.70),
        evict_dirty=_bool_env("SANDBOX_EVICT_DIRTY", True),
        daytona_max_sandboxes_per_user=_optional_int_env(
            "SANDBOX_DAYTONA_MAX_SANDBOXES_PER_USER"
        ),
        storage_sweep_interval_secs=_float_env(
            "SANDBOX_STORAGE_SWEEP_INTERVAL_SECS", 300.0
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


def _optional_float_env(name: str) -> float | None:
    """Like :func:`_float_env` but returns ``None`` for unset / blank.

    Used for ``SANDBOX_STORAGE_LIMIT_GB``: ``None`` means "no explicit
    budget — fall back to the disk-fraction default", which is a
    distinct mode from any numeric value.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None
