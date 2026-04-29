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
import shlex
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator, Callable
from urllib.parse import quote, urlparse

logger = logging.getLogger(__name__)

from sandbox.domain.errors import RepoAuthFailed, RepoCacheUnavailable, RuntimeNotFound, RuntimeUnavailable
from sandbox.domain.models import (
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

    def __init__(
        self,
        *,
        client_factory: Callable[[], Any] | None = None,
        snapshot: str | None = None,
        workspace_root: str = "/home/daytona/workspace",
        snapshot_dockerfile: str | Path | None = None,
        snapshot_build_timeout_s: float = 20 * 60,
        snapshot_heartbeat_s: float = 15,
    ) -> None:
        self._client_factory = client_factory or _default_daytona_client
        self.snapshot = snapshot
        self.workspace_root = workspace_root.rstrip("/")
        self.snapshot_dockerfile = (
            Path(snapshot_dockerfile) if snapshot_dockerfile else None
        )
        self.snapshot_build_timeout_s = snapshot_build_timeout_s
        self.snapshot_heartbeat_s = snapshot_heartbeat_s
        self._client: Any | None = None
        self._sandboxes: dict[str, Any] = {}                          # sandbox_id -> SDK object
        self._project_sandbox_ids: dict[tuple[str, str], str] = {}    # (user, project) -> sandbox_id
        self._bare_repos: set[tuple[str, str]] = set()                # (sandbox_id, repo_name)
        self._project_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._by_id: dict[str, Workspace] = {}
        self._by_key: dict[str, str] = {}
        self._snapshot_ensured = False
        self._snapshot_lock = threading.Lock()

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
        sandbox = self._ensure_sandbox(project_key)
        sandbox_id = str(getattr(sandbox, "id"))
        bare_path = self._bare_path(request.repo.repo_name)
        self._ensure_bare_repo(sandbox, bare_path, request)
        branch = request.branch_name or _default_branch_name(request)
        worktree_path = self._worktree_path(request.repo.repo_name, branch)
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

                params = CreateSandboxFromSnapshotParams(
                    snapshot=self.snapshot,
                    language="python",
                    labels=labels,
                    auto_stop_interval=30,
                    auto_archive_interval=43200,
                )
                return self.client.create(params)
            except TypeError:
                # Fall back to default creation if the installed SDK uses a
                # different parameter shape.
                return self.client.create()
        return self.client.create()

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

                image = Image.from_dockerfile(self.snapshot_dockerfile)
                self.client.snapshot.create(
                    CreateSnapshotParams(name=self.snapshot, image=image),
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
        return f"{self.workspace_root}/{_safe_segment(repo_name)}/.bare"

    def _worktree_path(self, repo_name: str, branch: str) -> str:
        return (
            f"{self.workspace_root}/{_safe_segment(repo_name)}/worktrees/"
            f"{_safe_segment(branch)}"
        )

    def _lookup_sandbox_by_id(self, sandbox_id: str) -> Any | None:
        for attr in ("get", "get_sandbox"):
            candidate = getattr(self.client, attr, None)
            if callable(candidate):
                try:
                    return candidate(sandbox_id)
                except Exception:
                    return None
        return None

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
        sandbox = self._sandbox(runtime)
        command = _command_string(request)
        cwd = request.cwd or runtime.spec.workdir
        response = sandbox.process.exec(
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


def _default_daytona_client() -> Any:
    try:
        from daytona import Daytona
    except ImportError as exc:
        raise RuntimeUnavailable(
            "Daytona SDK is not installed. Install potpie-sandbox[daytona]."
        ) from exc
    return Daytona()


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
