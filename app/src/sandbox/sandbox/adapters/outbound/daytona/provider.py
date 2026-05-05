"""Daytona-backed workspace and runtime providers.

The Daytona SDK is optional. This module imports it lazily so the sandbox core
can be installed without managed-sandbox dependencies.

Layout inside one Daytona sandbox::

    <workspace_root>/
        <repo_safe>/
            .bare/                 # bare clone — no working tree, no checked-out branch
            worktrees/
                <branch_safe>/     # one per active branch

One Daytona sandbox is created per ``(user_id, project_id)`` and re-used across
every workspace request for that pair. Branches share the underlying bare clone
via ``git worktree`` so multiple agent runs on different branches don't pay for
multiple Daytona sandboxes. The clone is bare so no branch is ever "checked
out at the cache" — every branch is free for ``worktree add``.

The runtime provider issues ``sandbox.process.exec`` against the same shared
sandbox; ``Runtime.spec.workdir`` is the per-workspace worktree path, so two
runtimes for two branches stay isolated even though they target the same
backend container.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator, Callable
from urllib.parse import quote, urlparse

logger = logging.getLogger(__name__)

from sandbox.domain.errors import RepoAuthFailed, RepoCacheUnavailable, RuntimeNotFound, RuntimeUnavailable
from sandbox.domain.models import (
    Capabilities,
    ExecChunk,
    ExecRequest,
    ExecResult,
    Mount,
    Runtime,
    RuntimeCapabilities,
    RuntimeSpec,
    RuntimeState,
    Workspace,
    WorkspaceLocation,
    WorkspaceMode,
    WorkspaceRequest,
    WorkspaceState,
    WorkspaceStorageKind,
    new_id,
    utc_now,
)


class DaytonaWorkspaceProvider:
    """Workspace provider where one Daytona sandbox hosts many branch worktrees.

    Sandboxes are keyed on ``(user_id, project_id)``. The first request for a
    project creates the sandbox and clones the repo; subsequent requests on
    the same project reuse the sandbox and add a fresh worktree per branch.
    """

    kind = "daytona"

    # Defaults sized for the project-sandbox lifecycle (Phase 2 of the
    # Daytona migration): one Daytona sandbox per (user, project)
    # kept alive across conversations, recreated lazily on health-check
    # failure. The previous 30-minute auto-stop was tuned for one-shot
    # agent runs and is too aggressive once we lean on the sandbox as
    # the durable home of a project's working tree.
    DEFAULT_AUTO_STOP_MINUTES = 24 * 60          # 24 hours
    DEFAULT_AUTO_ARCHIVE_MINUTES = 30 * 24 * 60  # 30 days

    def __init__(
        self,
        *,
        client_factory: Callable[[], Any] | None = None,
        snapshot: str | None = None,
        workspace_root: str = "/home/daytona/workspace",
        snapshot_dockerfile: str | Path | None = None,
        snapshot_build_timeout_s: float = 20 * 60,
        snapshot_heartbeat_s: float = 15,
        auto_stop_minutes: int | None = None,
        auto_archive_minutes: int | None = None,
        sandbox_name_prefix: str = "potpie",
        auto_delete_minutes: int | None = None,
        network_allow_list: str = "",
        network_block_all: bool = False,
        snapshot_cpu: int | None = None,
        snapshot_memory_gb: int | None = None,
        snapshot_disk_gb: int | None = None,
        use_volume_for_bare: bool = False,
        volume_name_prefix: str = "potpie-bare",
        volume_mount_path: str = "/home/agent/work/.bare-cache",
    ) -> None:
        self._client_factory = client_factory or _default_daytona_client
        self.snapshot = snapshot
        self.workspace_root = workspace_root.rstrip("/")
        self.snapshot_dockerfile = (
            Path(snapshot_dockerfile) if snapshot_dockerfile else None
        )
        self.snapshot_build_timeout_s = snapshot_build_timeout_s
        self.snapshot_heartbeat_s = snapshot_heartbeat_s
        # TTL knobs are constructor-overridable AND env-overridable so
        # tests can pin a tight value while operators tune via env. Env
        # takes precedence only when the constructor passes ``None`` —
        # a caller that explicitly sets the param wins (per least-surprise).
        self.auto_stop_minutes = (
            auto_stop_minutes
            if auto_stop_minutes is not None
            else _env_int("DAYTONA_AUTO_STOP_MINUTES", self.DEFAULT_AUTO_STOP_MINUTES)
        )
        self.auto_archive_minutes = (
            auto_archive_minutes
            if auto_archive_minutes is not None
            else _env_int(
                "DAYTONA_AUTO_ARCHIVE_MINUTES", self.DEFAULT_AUTO_ARCHIVE_MINUTES
            )
        )
        self.sandbox_name_prefix = sandbox_name_prefix
        self.auto_delete_minutes = auto_delete_minutes
        # Empty string disables the allow-list. Daytona accepts a
        # comma-separated CIDR list directly; we don't parse it here.
        self.network_allow_list = network_allow_list.strip()
        self.network_block_all = network_block_all
        self.snapshot_cpu = snapshot_cpu
        self.snapshot_memory_gb = snapshot_memory_gb
        self.snapshot_disk_gb = snapshot_disk_gb
        # Bare-cache volume knobs. ``use_volume_for_bare`` is the master
        # switch — off by default so existing deploys are unaffected.
        # When on, ``_ensure_volume`` runs per-user (volume is shared
        # across the user's projects via ``subpath``) and ``_bare_path``
        # returns a path inside the mount so the next sandbox for the
        # same project re-attaches the cached clone.
        self.use_volume_for_bare = use_volume_for_bare
        self.volume_name_prefix = volume_name_prefix
        self.volume_mount_path = volume_mount_path.rstrip("/")
        self._client: Any | None = None
        self._sandboxes: dict[str, Any] = {}                          # sandbox_id -> SDK object
        self._project_sandbox_ids: dict[tuple[str, str], str] = {}    # (user, project) -> sandbox_id
        self._bare_repos: set[tuple[str, str]] = set()                # (sandbox_id, repo_name)
        self._project_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._by_id: dict[str, Workspace] = {}
        self._by_key: dict[str, str] = {}
        self._snapshot_ensured = False
        self._snapshot_lock = threading.Lock()
        # Cache resolved volume ids per user so we don't pay a round-trip
        # to Daytona on every sandbox creation. Volumes are stable and
        # outlive sandboxes; a process-wide cache is safe.
        self._user_volume_ids: dict[str, str] = {}
        self._volume_lock = threading.Lock()

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    async def get_or_create_workspace(self, request: WorkspaceRequest) -> Workspace:
        key = request.key()
        existing = self._lookup_existing(key)
        if existing is not None:
            return existing

        project_key = (request.user_id, request.project_id)
        lock = self._project_locks.setdefault(project_key, asyncio.Lock())
        async with lock:
            existing = self._lookup_existing(key)
            if existing is not None:
                return existing
            workspace = await asyncio.to_thread(
                self._create_workspace_sync, request, project_key
            )
            self._by_id[workspace.id] = workspace
            self._by_key[workspace.key] = workspace.id
            return workspace

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        return self._by_id.get(workspace_id)

    async def delete_workspace(self, workspace: Workspace) -> None:
        """Remove the worktree but keep the per-project sandbox alive.

        The sandbox is shared across the project; tearing it down has to be
        explicit and is currently triggered out-of-band (TTL eviction, user
        action). Removing only the worktree keeps the model consistent with
        the local-fs adapter.
        """
        self._by_id.pop(workspace.id, None)
        self._by_key.pop(workspace.key, None)
        sandbox_id = workspace.location.backend_workspace_id
        if not sandbox_id:
            return
        sandbox = self._get_sandbox(sandbox_id)
        if sandbox is None:
            return
        worktree_path = workspace.location.remote_path
        if not worktree_path:
            return
        bare_path = self._bare_path(workspace.request.repo.repo_name)
        # Best-effort: ask git to deregister the worktree, then rm -rf so we
        # don't leak orphaned `.git/worktrees/<name>` admin entries either way.
        sandbox.process.exec(
            f"git -C {shlex.quote(bare_path)} worktree remove --force "
            f"{shlex.quote(worktree_path)} 2>/dev/null || true"
        )
        sandbox.process.exec(
            f"rm -rf {shlex.quote(worktree_path)}"
        )

    async def mount_for_runtime(self, workspace: Workspace, *, writable: bool) -> Mount:
        if workspace.location.remote_path is None:
            raise RuntimeUnavailable("Daytona workspace has no remote path")
        return Mount(
            source=workspace.location.remote_path,
            target=workspace.location.remote_path,
            writable=writable,
        )

    async def is_alive(self, workspace: Workspace) -> bool:
        """Probe the Daytona SDK to see whether the backing sandbox still exists.

        Goes off-box (one SDK call), but stays read-only and avoids
        starting any process inside the sandbox — the probe must stay
        cheap because ``ProjectSandbox.health_check`` runs it on every
        conversation message.

        Treats "not found" as ``False`` (sandbox was archived / deleted
        out-of-band — the recovery path will mint a new one). Treats
        transient SDK errors as ``False`` too: callers will then call
        ``ensure()`` which goes through the same recovery code, so
        a one-off network blip doesn't kill the conversation, it just
        adds one extra SDK round-trip.
        """
        sandbox_id = workspace.location.backend_workspace_id
        if not sandbox_id:
            return False
        try:
            sandbox = await asyncio.to_thread(self._lookup_sandbox_by_id, sandbox_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "daytona: is_alive probe failed for sandbox %s — treating as dead "
                "so the caller will recover. %s: %s",
                sandbox_id,
                type(exc).__name__,
                exc,
            )
            return False
        return sandbox is not None

    def sandbox_for_workspace(self, workspace_id: str) -> Any | None:
        workspace = self._by_id.get(workspace_id)
        if workspace is None:
            return None
        sandbox_id = workspace.location.backend_workspace_id
        return self._get_sandbox(sandbox_id) if sandbox_id else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _lookup_existing(self, key: str) -> Workspace | None:
        existing_id = self._by_key.get(key)
        if not existing_id:
            return None
        existing = self._by_id.get(existing_id)
        if existing is None:
            return None
        existing.last_used_at = utc_now()
        existing.updated_at = utc_now()
        return existing

    def _create_workspace_sync(
        self, request: WorkspaceRequest, project_key: tuple[str, str]
    ) -> Workspace:
        # Reject git-ref injection at the boundary. Keeps `git fetch
        # origin -- <ref>` and `worktree add -b <branch>` from being
        # smuggled into shell commands. Mirrors the local adapter's
        # validation (P4 brings the two paths to parity).
        _validate_ref(request.base_ref)
        if request.branch_name:
            _validate_ref(request.branch_name)

        sandbox = self._ensure_sandbox(project_key)
        sandbox_id = str(getattr(sandbox, "id"))
        bare_path = self._bare_path(request.repo.repo_name)
        self._ensure_bare_repo(sandbox, bare_path, request)
        branch = request.branch_name or _default_branch_name(request)
        worktree_path = self._worktree_path(request, branch)
        self._ensure_worktree(
            sandbox,
            bare_path=bare_path,
            worktree_path=worktree_path,
            branch=branch,
            base_ref=request.base_ref,
            mode=request.mode,
            create_branch=request.create_branch,
        )
        return Workspace(
            id=new_id("ws"),
            key=request.key(),
            repo_cache_id=f"daytona:{sandbox_id}:{request.repo.repo_name}",
            request=request,
            location=WorkspaceLocation(
                kind=WorkspaceStorageKind.DAYTONA_SANDBOX,
                remote_path=worktree_path,
                backend_workspace_id=sandbox_id,
            ),
            backend_kind=self.kind,
            state=WorkspaceState.READY,
            metadata={"branch": branch},
            capabilities=Capabilities.from_mode(request.mode),
        )

    def _ensure_sandbox(self, project_key: tuple[str, str]) -> Any:
        sandbox_id = self._project_sandbox_ids.get(project_key)
        if sandbox_id:
            cached = self._sandboxes.get(sandbox_id)
            if cached is not None:
                return cached
            recovered_by_id = self._lookup_sandbox_by_id(sandbox_id)
            if recovered_by_id is not None:
                self._sandboxes[sandbox_id] = recovered_by_id
                return recovered_by_id
            # We had an id but the SDK can't find it — drop it.
            self._project_sandbox_ids.pop(project_key, None)

        # Worker restarts wipe `_project_sandbox_ids`. The Daytona-side
        # sandbox (with the agent's bare clone and committed worktrees)
        # persists. Adopt it instead of orphaning it under a fresh sandbox.
        sandbox = self._recover_project_sandbox(project_key)
        if sandbox is None:
            sandbox = self._create_sandbox(project_key)
        sandbox_id = str(getattr(sandbox, "id"))
        self._sandboxes[sandbox_id] = sandbox
        self._project_sandbox_ids[project_key] = sandbox_id
        return sandbox

    def _recover_project_sandbox(self, project_key: tuple[str, str]) -> Any | None:
        """Find an existing potpie-managed sandbox for ``(user, project)``.

        The labels we set at creation time (``managed-by=potpie``,
        ``potpie-user``, ``potpie-project``) are exactly the lookup key. If
        Daytona returns a sandbox in a stopped state we start it; archived
        or destroyed sandboxes are skipped — their volume is gone.
        """
        user_id, project_id = project_key
        target_labels = {
            "managed-by": "potpie",
            "potpie-user": user_id,
            "potpie-project": project_id,
        }
        list_method = getattr(self.client, "list", None)
        if not callable(list_method):
            return None
        try:
            candidates = list_method(labels=target_labels)
        except TypeError:
            # Older SDKs may not support a `labels` kwarg; pull everything
            # and filter client-side.
            try:
                candidates = list_method()
            except Exception as exc:
                logger.debug("daytona list during recovery failed: %s", exc)
                return None
        except Exception as exc:
            logger.debug(
                "daytona list(labels=...) during recovery failed: %s", exc
            )
            return None

        # Newer Daytona SDKs return a `PaginatedSandboxes` pydantic model with
        # the actual sandboxes on ``.items``; older versions / our test fakes
        # may return a plain list. Unwrap defensively so neither shape silently
        # becomes "no candidates" → fresh-sandbox creation.
        items: list[Any] = []
        if isinstance(candidates, list):
            items = candidates
        else:
            raw_items = getattr(candidates, "items", None)
            if isinstance(raw_items, list):
                items = raw_items
        matched = [
            s
            for s in items
            if all(
                (getattr(s, "labels", None) or {}).get(k) == v
                for k, v in target_labels.items()
            )
        ]
        if not matched:
            return None
        # Multiple matches shouldn't happen but pick the most recently created
        # as a sane fallback.
        matched.sort(
            key=lambda s: str(getattr(s, "created_at", "") or ""), reverse=True
        )
        for sandbox in matched:
            state = getattr(sandbox, "state", "")
            state = getattr(state, "value", state)
            state_str = str(state).lower()
            if state_str in {"archived", "error", "destroyed"}:
                continue
            if state_str in {"stopped", "stopping"} and hasattr(sandbox, "start"):
                try:
                    sandbox.start()
                except Exception as exc:
                    logger.debug(
                        "daytona start during recovery failed: %s", exc
                    )
                    continue
            logger.info(
                "daytona: recovered sandbox %s for project %s/%s (state=%s)",
                getattr(sandbox, "id", "?"),
                user_id,
                project_id,
                state_str or "unknown",
            )
            return sandbox
        return None

    def _create_sandbox(self, project_key: tuple[str, str]) -> Any:
        user_id, project_id = project_key
        # Build the agent-sandbox snapshot on demand if it's missing. Without
        # this, the first request after a fresh Daytona stack fails with
        # "Snapshot ... not found" until the operator runs `make daytona-up`.
        self._ensure_snapshot()
        labels = {
            "managed-by": "potpie",
            "component": "sandbox-core",
            "potpie-user": user_id,
            "potpie-project": project_id,
        }
        if self.snapshot:
            try:
                from daytona import CreateSandboxFromSnapshotParams

                # The dashboard-visible name. Daytona enforces uniqueness
                # within an org, so the user/project segments encode the
                # natural recovery key (label-based recovery still does
                # the actual matching, this is purely for ops UX).
                name = self._sandbox_name(user_id, project_id)
                kwargs: dict[str, Any] = dict(
                    snapshot=self.snapshot,
                    language="python",
                    labels=labels,
                    name=name,
                    auto_stop_interval=self.auto_stop_minutes,
                    auto_archive_interval=self.auto_archive_minutes,
                )
                if self.auto_delete_minutes is not None:
                    kwargs["auto_delete_interval"] = self.auto_delete_minutes
                if self.network_block_all:
                    kwargs["network_block_all"] = True
                elif self.network_allow_list:
                    kwargs["network_allow_list"] = self.network_allow_list
                if self.use_volume_for_bare:
                    # Volume mount is sticky to the sandbox at creation
                    # time — Daytona does not let us add it later. So a
                    # sandbox created without the mount stays without
                    # the mount; a flag flip needs a sandbox recreate.
                    mount = self._build_bare_volume_mount(user_id, project_id)
                    if mount is not None:
                        kwargs["volumes"] = [mount]
                params = CreateSandboxFromSnapshotParams(**kwargs)
                return self.client.create(params)
            except TypeError:
                # Fall back to default creation if the installed SDK uses a
                # different parameter shape (older SDKs missed `name` /
                # `network_*`; the bare ``create()`` is the universal floor).
                return self.client.create()
        return self.client.create()

    def _sandbox_name(self, user_id: str, project_id: str) -> str:
        """Build a dashboard-visible sandbox name.

        Daytona enforces a per-org uniqueness constraint on names and
        truncates length, so we hash long ids down to 8 chars. Two
        sandboxes for the same (user, project) will collide on the
        name — that's intended: when one is destroyed, the next call
        either recovers it via labels or creates a new one and the
        name slot becomes free.
        """
        prefix = self.sandbox_name_prefix
        return f"{prefix}-{_safe_segment(user_id)[:8]}-{_safe_segment(project_id)[:8]}"

    def _volume_name_for(self, user_id: str) -> str:
        """Per-user volume name. Stable across the user's project list.

        Sharing one volume across a user's projects keeps us under
        Daytona's 100-volume-per-org cap as the project count grows;
        per-project isolation comes from the ``subpath`` mount option.
        """
        return f"{self.volume_name_prefix}-{_safe_segment(user_id)}"

    def _build_bare_volume_mount(
        self, user_id: str, project_id: str
    ) -> Any | None:
        """Resolve the user's volume and build the per-project mount spec.

        Returns ``None`` when the SDK can't expose ``VolumeMount``
        (older / partial installs) — the caller then creates the
        sandbox without a mount, which falls back to local-fs bare
        clones. Doesn't raise: a volume hiccup must not block sandbox
        creation; the bare clone path will just re-clone into local fs
        on this run.
        """
        try:
            from daytona.common.volume import VolumeMount
        except ImportError:
            logger.warning(
                "daytona: VolumeMount class not available — falling back "
                "to local-fs bare clone for this sandbox"
            )
            return None
        volume_id = self._ensure_volume(user_id)
        if volume_id is None:
            return None
        # Subpath isolates the per-project tree inside the shared user
        # volume so the sandbox sees only its own ``.bare/`` dir.
        return VolumeMount(
            volume_id=volume_id,
            mount_path=self.volume_mount_path,
            subpath=_safe_segment(project_id),
        )

    def _ensure_volume(self, user_id: str) -> str | None:
        """Get-or-create the user's bare-cache volume; return its id.

        Returns ``None`` on transient failures so the caller falls back
        to local-fs cloning rather than failing the whole sandbox
        creation. Caches the resolved id process-wide because volumes
        are durable and the SDK round-trip is non-trivial.
        """
        cached = self._user_volume_ids.get(user_id)
        if cached is not None:
            return cached
        with self._volume_lock:
            cached = self._user_volume_ids.get(user_id)
            if cached is not None:
                return cached
            name = self._volume_name_for(user_id)
            volume_service = getattr(self.client, "volume", None)
            if volume_service is None:
                logger.warning(
                    "daytona: client has no volume service — falling back "
                    "to local-fs bare clone for user %s",
                    user_id,
                )
                return None
            try:
                volume = volume_service.get(name, create=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "daytona: get-or-create volume %s failed (%s: %s) — "
                    "falling back to local-fs bare clone",
                    name,
                    type(exc).__name__,
                    exc,
                )
                return None
            volume_id = str(getattr(volume, "id", "") or "")
            if not volume_id:
                return None
            self._user_volume_ids[user_id] = volume_id
            logger.info(
                "daytona: bare-cache volume %s ready (id=%s) for user %s",
                name,
                volume_id,
                user_id,
            )
            return volume_id

    def _ensure_snapshot(self) -> None:
        """Make sure the agent-sandbox snapshot is ACTIVE in Daytona.

        Called once per process on the first sandbox creation. The snapshot
        has to be in ``active`` state — Daytona rejects sandbox creation from
        a snapshot in ``pending``/``building``/``pulling``. So:

        * not found → build it from the bundled Dockerfile
        * present + active → done
        * present + transient state → wait until active (or timeout)
        * present + error/build_failed → raise

        ``_snapshot_ensured`` only latches on success so a timeout retry
        re-checks (the build may have finished out-of-band).
        """
        if self._snapshot_ensured or not self.snapshot:
            return
        with self._snapshot_lock:
            if self._snapshot_ensured:
                return
            try:
                existing = self.client.snapshot.get(self.snapshot)
            except Exception as exc:
                if _is_not_found(exc):
                    existing = None
                else:
                    # Auth, network, etc. — let the caller see it on the next
                    # SDK call rather than masking it here.
                    self._snapshot_ensured = True
                    return

            if existing is not None:
                state = _state_value(existing)
                if state == "active":
                    self._snapshot_ensured = True
                    return
                if state in {"error", "build_failed"}:
                    raise RuntimeUnavailable(
                        f"daytona snapshot {self.snapshot} is in state "
                        f"{state!r} — delete it via the Daytona dashboard "
                        "and retry"
                    )
                logger.info(
                    "daytona snapshot %s present in state %r — waiting for active",
                    self.snapshot,
                    state,
                )
                self._wait_for_active(state)
                return

            if self.snapshot_dockerfile is None or not self.snapshot_dockerfile.exists():
                logger.warning(
                    "daytona snapshot %s missing and no dockerfile available "
                    "for auto-build (set DAYTONA_SNAPSHOT_DOCKERFILE)",
                    self.snapshot,
                )
                self._snapshot_ensured = True
                return

            logger.info(
                "daytona snapshot %s missing — building from %s "
                "(timeout=%.0fs, heartbeat=%.0fs)",
                self.snapshot,
                self.snapshot_dockerfile,
                self.snapshot_build_timeout_s,
                self.snapshot_heartbeat_s,
            )
            self._build_snapshot()

    def _build_snapshot(self) -> None:
        """Submit the dockerfile and block until snapshot is built or fails."""
        started_at = time.monotonic()
        with self._heartbeat_running("building"):
            try:
                from daytona import CreateSnapshotParams
                from daytona.common.image import Image
                from daytona.common.sandbox import Resources

                image = Image.from_dockerfile(self.snapshot_dockerfile)
                # Resources are baked into the snapshot — Daytona forbids
                # overriding them at sandbox creation time when spawning
                # from a snapshot. Set them here so the snapshot ships
                # right-sized for medium repos.
                snapshot_kwargs: dict[str, Any] = dict(
                    name=self.snapshot, image=image
                )
                if (
                    self.snapshot_cpu is not None
                    or self.snapshot_memory_gb is not None
                    or self.snapshot_disk_gb is not None
                ):
                    snapshot_kwargs["resources"] = Resources(
                        cpu=self.snapshot_cpu,
                        memory=self.snapshot_memory_gb,
                        disk=self.snapshot_disk_gb,
                    )
                self.client.snapshot.create(
                    CreateSnapshotParams(**snapshot_kwargs),
                    on_logs=lambda chunk: logger.info("daytona snapshot: %s", chunk),
                    timeout=self.snapshot_build_timeout_s,
                )
                logger.info(
                    "daytona snapshot %s ready (took %.0fs)",
                    self.snapshot,
                    time.monotonic() - started_at,
                )
                self._snapshot_ensured = True
            except Exception as exc:
                if _is_timeout(exc):
                    raise RuntimeUnavailable(
                        f"daytona snapshot {self.snapshot} build timed out "
                        f"after {self.snapshot_build_timeout_s:.0f}s — "
                        "check Daytona runner logs"
                    ) from exc
                logger.error(
                    "daytona snapshot %s build failed: %s", self.snapshot, exc
                )
                self._snapshot_ensured = True

    def _wait_for_active(self, initial_state: str) -> None:
        """Poll Daytona until the snapshot reaches active (or timeout/fail).

        Used when an earlier session kicked off the build (snapshot exists but
        is still PENDING/BUILDING/PULLING). We can't call ``snapshot.create``
        again — Daytona would conflict — so we just wait.
        """
        started_at = time.monotonic()
        deadline = started_at + self.snapshot_build_timeout_s
        with self._heartbeat_running(f"waiting (initial state: {initial_state})"):
            while True:
                if time.monotonic() > deadline:
                    raise RuntimeUnavailable(
                        f"daytona snapshot {self.snapshot} did not reach "
                        f"active within {self.snapshot_build_timeout_s:.0f}s "
                        "— check Daytona runner logs"
                    )
                try:
                    snap = self.client.snapshot.get(self.snapshot)
                except Exception as exc:
                    raise RuntimeUnavailable(
                        f"daytona snapshot {self.snapshot} lookup failed "
                        f"while waiting for active: {exc}"
                    ) from exc
                state = _state_value(snap)
                if state == "active":
                    logger.info(
                        "daytona snapshot %s is active (waited %.0fs)",
                        self.snapshot,
                        time.monotonic() - started_at,
                    )
                    self._snapshot_ensured = True
                    return
                if state in {"error", "build_failed"}:
                    raise RuntimeUnavailable(
                        f"daytona snapshot {self.snapshot} entered state "
                        f"{state!r} while waiting — delete it via the Daytona "
                        "dashboard and retry"
                    )
                time.sleep(2)

    @contextlib.contextmanager
    def _heartbeat_running(self, label: str):
        """Run a block while a daemon thread emits a heartbeat log.

        Heartbeats fill the silence the SDK leaves between state transitions
        — without them a snapshot stuck in PENDING looks indistinguishable
        from a hung worker.
        """
        stop = threading.Event()
        started_at = time.monotonic()

        def beat() -> None:
            while not stop.wait(self.snapshot_heartbeat_s):
                elapsed = int(time.monotonic() - started_at)
                logger.info(
                    "daytona snapshot %s still %s (waited %ds)",
                    self.snapshot,
                    label,
                    elapsed,
                )

        thread = threading.Thread(
            target=beat, daemon=True, name="daytona-snapshot-heartbeat"
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=1)

    def _ensure_bare_repo(
        self, sandbox: Any, bare_path: str, request: WorkspaceRequest
    ) -> None:
        """Make sure ``bare_path`` holds a bare clone of the repo.

        Bare cloning is what makes the rest of this provider safe: a bare
        repo has no working tree, so no branch is "checked out" anywhere,
        and ``git worktree add`` can target any branch without conflicts.
        """
        sandbox_id = str(getattr(sandbox, "id"))
        bare_key = (sandbox_id, request.repo.repo_name)
        if bare_key in self._bare_repos:
            self._fetch_ref(sandbox, bare_path, request.base_ref)
            return
        # Sandbox may have been recovered with the clone already on disk —
        # Daytona persists the volume across restarts. Bare repos hold HEAD
        # at the root, not inside a `.git/` directory.
        probe = sandbox.process.exec(
            f"test -f {shlex.quote(bare_path + '/HEAD')}"
        )
        if int(getattr(probe, "exit_code", 0)) == 0:
            self._bare_repos.add(bare_key)
            self._fetch_ref(sandbox, bare_path, request.base_ref)
            return

        repo_url = request.repo.repo_url or _default_github_url(
            request.repo.repo_name
        )
        clone_url = _authenticated_url(repo_url, request.auth_token)
        self._clone_bare(sandbox, clone_url, bare_path, request.base_ref)
        self._bare_repos.add(bare_key)

    def _fetch_ref(self, sandbox: Any, bare_path: str, base_ref: str) -> None:
        # Best-effort refresh of the requested ref. Transient network failures
        # don't fail the request — what's already in the bare repo is usually
        # enough for the worktree step.
        sandbox.process.exec(
            f"git -C {shlex.quote(bare_path)} fetch origin -- "
            f"{shlex.quote(base_ref)} 2>/dev/null || true"
        )

    def _clone_bare(
        self, sandbox: Any, clone_url: str, bare_path: str, base_ref: str
    ) -> None:
        """Bare-clone the repo into ``bare_path`` then fetch ``base_ref``.

        The Daytona SDK's ``git.clone`` doesn't expose ``--bare``, so this
        path is shell-only. Auth is embedded in the URL (same approach as
        ``LocalGitWorkspaceProvider``); errors are sanitized before they
        escape so the token doesn't leak into logs.
        """
        parent = "/".join(bare_path.split("/")[:-1])
        self._exec_or_raise(
            sandbox, f"mkdir -p {shlex.quote(parent)}", cwd=None, operation="mkdir"
        )
        result = sandbox.process.exec(
            f"git clone --bare --filter=blob:none -- "
            f"{shlex.quote(clone_url)} {shlex.quote(bare_path)}"
        )
        if int(getattr(result, "exit_code", 0)) != 0:
            message = _sanitize_git_error(_payload(result))
            if (
                "authentication" in message.lower()
                or "permission denied" in message.lower()
            ):
                raise RepoAuthFailed(message)
            raise RepoCacheUnavailable(f"git clone --bare failed: {message}")
        # `--filter=blob:none` defers blob downloads; an explicit fetch primes
        # the requested ref so `worktree add` can resolve it without surprise.
        fetch = sandbox.process.exec(
            f"git -C {shlex.quote(bare_path)} fetch origin -- {shlex.quote(base_ref)}"
        )
        if int(getattr(fetch, "exit_code", 0)) != 0:
            raise RepoCacheUnavailable(
                f"git fetch failed: {_sanitize_git_error(_payload(fetch))}"
            )
        # Verify — the slim Daytona snapshot's toolbox can return success on
        # a 204 body that is hard to introspect, so prove the repo materialised.
        status = sandbox.process.exec(
            f"test -f {shlex.quote(bare_path + '/HEAD')}"
        )
        if int(getattr(status, "exit_code", 0)) != 0:
            raise RepoCacheUnavailable(
                f"Daytona git clone did not create a bare repository at {bare_path}"
            )

    def _ensure_worktree(
        self,
        sandbox: Any,
        *,
        bare_path: str,
        worktree_path: str,
        branch: str,
        base_ref: str,
        mode: WorkspaceMode,
        create_branch: bool,
    ) -> None:
        # Trust an existing worktree at the path. We key on `(repo, branch)`,
        # so the only way to re-enter is to ask for the same branch again.
        # `-e` (exists) handles both real git's gitdir-pointer file and the
        # test fake's directory marker.
        probe = sandbox.process.exec(
            f"test -e {shlex.quote(worktree_path + '/.git')}"
        )
        if int(getattr(probe, "exit_code", 0)) == 0:
            return

        # A previous attempt may have left a half-baked directory or a stale
        # admin entry under `.git/worktrees/`. Prune both before retrying so
        # `worktree add` doesn't bail with "already exists".
        sandbox.process.exec(
            f"git -C {shlex.quote(bare_path)} worktree prune 2>/dev/null || true"
        )
        sandbox.process.exec(f"rm -rf {shlex.quote(worktree_path)}")

        parent = str(PurePosixPath(worktree_path).parent)
        self._exec_or_raise(
            sandbox,
            f"mkdir -p {shlex.quote(parent)}",
            cwd=None,
            operation="mkdir worktrees",
        )
        if mode is WorkspaceMode.ANALYSIS and not create_branch:
            cmd = (
                f"git -C {shlex.quote(bare_path)} worktree add --detach "
                f"-- {shlex.quote(worktree_path)} {shlex.quote(base_ref)}"
            )
            result = sandbox.process.exec(cmd)
        else:
            # `-b` creates a fresh branch from base_ref. Re-running on the
            # same scope (same conversation_id/task_id) hits an existing
            # branch — fall through to plain `worktree add` so we reuse it
            # and preserve any commits the agent made on a prior run. Using
            # `-B` instead would reset the branch back to base_ref and throw
            # those commits away.
            cmd = (
                f"git -C {shlex.quote(bare_path)} worktree add -b "
                f"{shlex.quote(branch)} -- {shlex.quote(worktree_path)} "
                f"{shlex.quote(base_ref)}"
            )
            result = sandbox.process.exec(cmd)
            if (
                int(getattr(result, "exit_code", 0)) != 0
                and "already exists" in _payload(result).lower()
            ):
                cmd = (
                    f"git -C {shlex.quote(bare_path)} worktree add -- "
                    f"{shlex.quote(worktree_path)} {shlex.quote(branch)}"
                )
                result = sandbox.process.exec(cmd)
        if int(getattr(result, "exit_code", 0)) != 0:
            payload = _payload(result).strip()
            raise RepoCacheUnavailable(
                f"git worktree add failed: {payload or '(no output)'}"
            )

    def _bare_path(self, repo_name: str) -> str:
        # With volumes on, the bare clone lives inside the per-user
        # volume mounted at ``volume_mount_path``. The mount uses a
        # per-project ``subpath`` so two projects in the same volume
        # don't collide; the sandbox sees only this project's tree.
        # Repo name is unused here on purpose: one sandbox is one
        # project is one repo, and decoupling the path from the repo
        # name lets a repo rename in Potpie not invalidate the cached
        # clone (the volume content is what matters).
        if self.use_volume_for_bare:
            return f"{self.volume_mount_path}/.bare"
        return f"{self.workspace_root}/{_safe_segment(repo_name)}/.bare"

    def _worktree_path(self, request: WorkspaceRequest, branch: str) -> str:
        """Encode user + scope + branch in the worktree path.

        Two conversations on the same branch must NOT share a worktree.
        Without user/scope encoding, the bare path layout used to be
        ``.../worktrees/<branch>`` — fine for analysis, dangerous for
        EDIT/TASK because two ``conversation_id`` values that produced
        the same agent branch name (or hashed to similar UUIDs in tests)
        would collide on a single Daytona worktree. Mirrors the local
        adapter's path scheme.
        """
        safe_user = _safe_segment(request.user_id)
        scope = request.conversation_id or request.task_id or request.base_ref
        safe_scope = _safe_segment(scope)
        safe_branch = _safe_segment(branch)
        return (
            f"{self.workspace_root}/{_safe_segment(request.repo.repo_name)}"
            f"/worktrees/{safe_user}_{safe_scope}_{safe_branch}"
        )

    def _lookup_sandbox_by_id(self, sandbox_id: str) -> Any | None:
        """Resolve a Daytona sandbox by id via the SDK.

        Returns ``None`` when the SDK confirms the sandbox does not exist.
        Other errors (auth, network) are re-raised so callers can tell
        "intentionally gone" apart from "transient outage" — the recovery
        path should only trigger on the former.
        """
        for attr in ("get", "get_sandbox"):
            candidate = getattr(self.client, attr, None)
            if callable(candidate):
                try:
                    return candidate(sandbox_id)
                except Exception as exc:
                    if _is_not_found(exc):
                        return None
                    raise
        return None

    def recover_dead_sandbox(
        self,
        dead_sandbox_id: str,
        *,
        user_id: str | None,
        project_id: str | None,
    ) -> Any | None:
        """Replace a dead sandbox id with a live one for ``(user, project)``.

        Called from the runtime provider when a previously persisted sandbox
        id stops resolving (the user destroyed it, the TTL expired, etc.).
        Drops the dead caches and either adopts an existing potpie-managed
        sandbox via labels or creates a fresh one. Returns the live SDK
        sandbox object, or ``None`` if neither user_id nor project_id is
        available (so we can't ensure a project-scoped sandbox safely).
        """
        # Drop the dead id from the in-memory caches so subsequent lookups
        # don't keep returning the stale handle.
        self._sandboxes.pop(dead_sandbox_id, None)
        # Reverse-lookup the project key in case the caller didn't pass
        # one (e.g. older runtimes whose spec.labels predate the user_id /
        # project_id stamping).
        cached_pk: tuple[str, str] | None = None
        for pk, sid in list(self._project_sandbox_ids.items()):
            if sid == dead_sandbox_id:
                self._project_sandbox_ids.pop(pk, None)
                cached_pk = pk
        if user_id and project_id:
            project_key: tuple[str, str] = (user_id, project_id)
        elif cached_pk is not None:
            project_key = cached_pk
        else:
            logger.warning(
                "daytona: cannot recover sandbox %s — runtime spec did "
                "not carry (user_id, project_id) labels",
                dead_sandbox_id,
            )
            return None
        try:
            sandbox = self._ensure_sandbox(project_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "daytona: recover_dead_sandbox failed for %s/%s: %s",
                project_key[0],
                project_key[1],
                exc,
            )
            return None
        # Update any cached workspace whose location pointed at the dead id
        # so a follow-up `delete_workspace` / `sandbox_for_workspace` lookup
        # uses the new id without going back through `_get_sandbox`.
        # ``WorkspaceLocation`` is frozen (it lives in the domain layer and
        # mutating it via `dataclasses.replace` keeps the immutability
        # invariant intact).
        from dataclasses import replace as _dc_replace

        new_id = str(getattr(sandbox, "id"))
        for ws in self._by_id.values():
            if ws.location.backend_workspace_id == dead_sandbox_id:
                ws.location = _dc_replace(ws.location, backend_workspace_id=new_id)
                ws.updated_at = utc_now()
        logger.info(
            "daytona: recovered dead sandbox %s -> %s for %s/%s",
            dead_sandbox_id,
            new_id,
            project_key[0],
            project_key[1],
        )
        return sandbox

    @staticmethod
    def _exec_or_raise(
        sandbox: Any, command: str, *, cwd: str | None, operation: str
    ) -> None:
        response = sandbox.process.exec(command, cwd=cwd)
        exit_code = int(getattr(response, "exit_code", 0))
        if exit_code != 0:
            result = getattr(response, "result", "") or getattr(
                getattr(response, "artifacts", None), "stdout", ""
            )
            raise RepoCacheUnavailable(f"{operation} failed: {result}")

    def _get_sandbox(self, sandbox_id: str | None) -> Any | None:
        if sandbox_id is None:
            return None
        if sandbox_id in self._sandboxes:
            return self._sandboxes[sandbox_id]
        recovered = self._lookup_sandbox_by_id(sandbox_id)
        if recovered is not None:
            self._sandboxes[sandbox_id] = recovered
        return recovered


class DaytonaRuntimeProvider:
    kind = "daytona"
    capabilities = RuntimeCapabilities(
        preview_url=True,
        interactive_session=True,
    )

    def __init__(self, workspace_provider: DaytonaWorkspaceProvider) -> None:
        self._workspace_provider = workspace_provider
        self._runtimes: dict[str, Runtime] = {}

    async def create(self, workspace_id: str, spec: RuntimeSpec) -> Runtime:
        sandbox = self._workspace_provider.sandbox_for_workspace(workspace_id)
        if sandbox is None:
            backend_workspace_id = spec.labels.get("workspace_backend_id")
            sandbox = self._workspace_provider._get_sandbox(backend_workspace_id)
        if sandbox is None:
            raise RuntimeUnavailable(f"No Daytona sandbox for workspace {workspace_id}")
        sandbox_id = str(getattr(sandbox, "id"))
        runtime = Runtime(
            id=new_id("rt"),
            workspace_id=workspace_id,
            backend_kind=self.kind,
            backend_runtime_id=sandbox_id,
            spec=spec,
            state=RuntimeState.RUNNING,
        )
        self._runtimes[runtime.id] = runtime
        return runtime

    async def get(self, runtime_id: str) -> Runtime | None:
        return self._runtimes.get(runtime_id)

    async def start(self, runtime: Runtime) -> Runtime:
        self._runtimes.setdefault(runtime.id, runtime)
        sandbox = self._sandbox(runtime)
        if hasattr(sandbox, "start"):
            sandbox.start()
        runtime.state = RuntimeState.RUNNING
        runtime.last_started_at = utc_now()
        runtime.updated_at = utc_now()
        return runtime

    async def stop(self, runtime: Runtime) -> Runtime:
        self._runtimes.setdefault(runtime.id, runtime)
        # Don't actually stop the Daytona sandbox here — it's shared across
        # other workspaces in the same project. Just mark this runtime
        # logically stopped.
        runtime.state = RuntimeState.STOPPED
        runtime.updated_at = utc_now()
        return runtime

    async def destroy(self, runtime: Runtime) -> None:
        self._runtimes.pop(runtime.id, None)
        runtime.state = RuntimeState.DELETED

    async def exec(self, runtime: Runtime, request: ExecRequest) -> ExecResult:
        sandbox = self._sandbox_with_recovery(runtime)
        command = _command_string(request)
        cwd = request.cwd or runtime.spec.workdir
        try:
            response = sandbox.process.exec(
                command,
                cwd=cwd,
                env=dict(request.env) or None,
                timeout=request.timeout_s,
            )
        except Exception as exc:
            # The SDK can return a stale handle from `client.get(id)` even
            # after Daytona destroyed the sandbox — we only learn about it
            # when we actually try to exec. Recover via labels and retry
            # exactly once so the agent's tool call doesn't bubble a
            # confusing "Sandbox with ID … not found" error.
            if not _is_not_found(exc):
                raise
            recovered = self._recover_runtime(runtime, dead_sandbox_id=str(getattr(sandbox, "id", runtime.backend_runtime_id)))
            if recovered is None:
                raise
            response = recovered.process.exec(
                command,
                cwd=cwd,
                env=dict(request.env) or None,
                timeout=request.timeout_s,
            )
        stdout = _response_stdout(response)
        exit_code = int(getattr(response, "exit_code", 0))
        if request.max_output_bytes and len(stdout) > request.max_output_bytes:
            return ExecResult(
                exit_code=exit_code,
                stdout=stdout[: request.max_output_bytes],
                truncated=True,
            )
        return ExecResult(exit_code=exit_code, stdout=stdout)

    async def exec_stream(
        self, runtime: Runtime, request: ExecRequest
    ) -> AsyncIterator[ExecChunk]:
        result = await self.exec(runtime, request)
        if result.stdout:
            yield ExecChunk(stream="stdout", data=result.stdout)
        if result.stderr:
            yield ExecChunk(stream="stderr", data=result.stderr)

    # ------------------------------------------------------------------
    # Native filesystem ops (duck-typed; SandboxService prefers these
    # over `exec` when the runtime exposes them).
    #
    # The Daytona SDK's ``process.exec`` has no stdin parameter, so the
    # generic ``cat > path`` exec path silently drops content on Daytona.
    # ``sandbox.fs.upload_file`` is the documented write API and goes
    # through the same toolbox channel as exec — round-trip cost is
    # comparable. Read / list go through ``download_file`` / ``list_files``
    # for symmetry: typed responses, no shell-injection surface.
    # ------------------------------------------------------------------
    async def read_bytes(self, runtime: Runtime, path: str) -> bytes:
        sandbox = self._sandbox_with_recovery(runtime)
        return await asyncio.to_thread(sandbox.fs.download_file, path)

    async def write_bytes(
        self, runtime: Runtime, path: str, content: bytes
    ) -> None:
        sandbox = self._sandbox_with_recovery(runtime)
        # `upload_file` overwrites by default; parent must exist. Mkdir is
        # cheap on Daytona (toolbox-side `mkdir -p`) and keeps the call
        # site free of race-y "create then write" branches.
        parent = str(PurePosixPath(path).parent)
        if parent and parent not in (".", "/"):
            await asyncio.to_thread(sandbox.fs.create_folder, parent, "755")
        await asyncio.to_thread(sandbox.fs.upload_file, content, path)

    async def list_dir_native(
        self, runtime: Runtime, path: str
    ) -> list[tuple[str, bool, int | None]]:
        """Return ``(name, is_dir, size)`` triples. Empty list on missing dir.

        Returning a primitive triple keeps the runtime port free of
        domain types — SandboxClient adapts to ``FileEntry``. The Daytona
        toolbox raises a ``FileNotFoundError``-equivalent on missing
        directories; we surface that as an empty list to mirror the
        ``ls -1Ap`` exec path's "no listing on missing dir" semantics.
        """
        sandbox = self._sandbox_with_recovery(runtime)
        try:
            entries = await asyncio.to_thread(sandbox.fs.list_files, path)
        except Exception as exc:
            if _is_not_found(exc):
                return []
            raise
        out: list[tuple[str, bool, int | None]] = []
        for entry in entries:
            name = getattr(entry, "name", None) or ""
            is_dir = bool(getattr(entry, "is_dir", False))
            size_attr = getattr(entry, "size", None)
            size = int(size_attr) if size_attr is not None else None
            out.append((name, is_dir, size))
        return out

    def _require(self, runtime_id: str) -> Runtime:
        runtime = self._runtimes.get(runtime_id)
        if runtime is None:
            raise RuntimeNotFound(runtime_id)
        return runtime

    def _sandbox(self, runtime: Runtime) -> Any:
        sandbox = self._workspace_provider._get_sandbox(runtime.backend_runtime_id)
        if sandbox is None:
            raise RuntimeUnavailable(
                f"Daytona sandbox not found: {runtime.backend_runtime_id}"
            )
        return sandbox

    def _sandbox_with_recovery(self, runtime: Runtime) -> Any:
        """Resolve the runtime's sandbox, recovering once if it's gone.

        Mirrors :meth:`_sandbox` but, when the persisted backend id no
        longer resolves (the SDK confirms 404), tries one round of
        label-based recovery before giving up. Mutates
        ``runtime.backend_runtime_id`` to the new sandbox so the
        subsequent :func:`SandboxService._exec_unlocked` ``save_runtime``
        call persists the live id and follow-up calls don't re-pay the
        recovery cost.
        """
        try:
            return self._sandbox(runtime)
        except RuntimeUnavailable:
            dead = runtime.backend_runtime_id
            if not dead:
                raise
            recovered = self._recover_runtime(runtime, dead_sandbox_id=dead)
            if recovered is None:
                raise
            return recovered

    def _recover_runtime(
        self, runtime: Runtime, *, dead_sandbox_id: str
    ) -> Any | None:
        """Replace a dead Daytona sandbox with a fresh / adopted one.

        Pulls ``user_id`` / ``project_id`` from
        ``runtime.spec.labels`` (stamped by
        :meth:`SandboxService._build_runtime_spec`) and asks the workspace
        provider to recover. Updates ``runtime.backend_runtime_id`` in
        place so the service's downstream ``save_runtime`` persists the
        new id; without that we'd recover on every single exec call.
        """
        labels = runtime.spec.labels or {}
        sandbox = self._workspace_provider.recover_dead_sandbox(
            dead_sandbox_id,
            user_id=labels.get("user_id"),
            project_id=labels.get("project_id"),
        )
        if sandbox is None:
            return None
        new_id = str(getattr(sandbox, "id"))
        runtime.backend_runtime_id = new_id
        runtime.updated_at = utc_now()
        return sandbox


def _default_daytona_client() -> Any:
    try:
        from daytona import Daytona
    except ImportError as exc:
        raise RuntimeUnavailable(
            "Daytona SDK is not installed. Install potpie-sandbox[daytona]."
        ) from exc
    return Daytona()


def _env_int(name: str, default: int) -> int:
    """Parse an integer env var, falling back to ``default`` on missing/invalid.

    Used for TTL knobs where a misconfigured env value (e.g. "30m"
    instead of an int minute count) shouldn't crash the provider —
    we log and use the default so the deploy stays up.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning(
            "daytona: env var %s=%r is not an int; falling back to %d",
            name,
            raw,
            default,
        )
        return default


def _command_string(request: ExecRequest) -> str:
    if request.shell:
        return request.cmd[0] if len(request.cmd) == 1 else " ".join(
            shlex.quote(p) for p in request.cmd
        )
    return " ".join(shlex.quote(p) for p in request.cmd)


def _response_stdout(response: Any) -> bytes:
    value = getattr(response, "result", None)
    if value is None:
        artifacts = getattr(response, "artifacts", None)
        value = getattr(artifacts, "stdout", "") if artifacts else ""
    if isinstance(value, bytes):
        return value
    return str(value or "").encode()


def _authenticated_url(repo_url: str, auth_token: str | None) -> str:
    """Embed a token directly in the URL via ``x-access-token:<token>@host``.

    The shell-based bare clone has no separate auth knob, so the credential
    has to live in the URL. Same approach as ``LocalGitWorkspaceProvider``.
    """
    if not auth_token:
        return repo_url
    parsed = urlparse(repo_url)
    if not parsed.scheme or not parsed.netloc:
        return repo_url
    host = parsed.hostname or parsed.netloc
    if parsed.port:
        host = f"{host}:{parsed.port}"
    netloc = f"x-access-token:{quote(auth_token, safe='')}@{host}"
    return parsed._replace(netloc=netloc).geturl()


def _sanitize_git_error(message: str) -> str:
    """Mask the embedded token before letting an error message escape."""
    return message.replace("x-access-token:", "x-access-token:***")


def _payload(response: Any) -> str:
    """Coerce a Daytona toolbox response body to a string for diagnostics."""
    value = getattr(response, "result", "") or ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _safe_segment(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in value)


def _validate_ref(ref: str) -> None:
    """Reject git-ref values that would unsafely interpolate into shell.

    The Daytona adapter uses ``sandbox.process.exec`` with shell-style
    command strings; an attacker-controlled branch with newlines or
    ``..`` could break out of the intended ref. Mirrors the local
    adapter's ``validate_ref`` so both backends share the same threat
    model.
    """
    if not ref:
        raise ValueError("git ref cannot be empty")
    if ".." in ref or "\n" in ref or "\r" in ref:
        raise ValueError(f"git ref contains unsafe characters: {ref!r}")


def _default_github_url(repo_name: str) -> str:
    return f"https://github.com/{repo_name}.git"


def _is_not_found(exc: Exception) -> bool:
    """Match a Daytona "snapshot not found" error.

    The SDK exposes a typed ``DaytonaNotFoundError`` but it imports lazily;
    fall back to message-sniffing if the typed class isn't reachable.
    """
    try:
        from daytona.common.errors import DaytonaNotFoundError

        if isinstance(exc, DaytonaNotFoundError):
            return True
    except ImportError:
        pass
    message = str(exc).lower()
    return "not found" in message or "404" in message


def _is_timeout(exc: Exception) -> bool:
    """Match the SDK's `DaytonaTimeoutError` raised by `@with_timeout()`."""
    try:
        from daytona.common.errors import DaytonaTimeoutError

        if isinstance(exc, DaytonaTimeoutError):
            return True
    except ImportError:
        pass
    return "exceeded timeout" in str(exc).lower()


def _state_value(snap: Any) -> str:
    """Normalize a Snapshot.state field to a lowercase string.

    `state` is a `SnapshotState(str, Enum)` in the SDK, so `.value` exists,
    but be defensive in case the API returns a bare string.
    """
    state = getattr(snap, "state", None)
    if state is None:
        return ""
    val = getattr(state, "value", None)
    return str(val if val is not None else state).lower()


def _default_branch_name(request: WorkspaceRequest) -> str:
    """Pick a branch name when the caller didn't pass one.

    Never returns ``base_ref`` for write-capable modes — the agent must commit
    onto its own branch, not whatever the user pointed at. Mirrors
    ``LocalGitWorkspaceProvider._default_branch_name``.
    """
    if request.mode is WorkspaceMode.ANALYSIS:
        # Used purely as metadata; the worktree itself is detached.
        return request.base_ref
    if request.mode is WorkspaceMode.TASK and request.task_id:
        return f"agent/task-{request.task_id.replace('/', '-')}"
    if request.conversation_id:
        return f"agent/edits-{request.conversation_id.replace('/', '-')}"
    return f"agent/workspace-{new_id('branch')}"
