"""Application service that coordinates workspace and runtime providers."""

from __future__ import annotations

from sandbox.domain.errors import (
    GitPlatformNotConfigured,
    RuntimeNotFound,
    WorkspaceNotFound,
)
from sandbox.domain.models import (
    Author,
    ExecRequest,
    ExecResult,
    Mount,
    NetworkMode,
    PullRequest,
    PullRequestComment,
    PullRequestCommentResult,
    PullRequestRequest,
    RemoteAuth,
    RepoCache,
    RepoCacheRequest,
    RepoIdentity,
    Runtime,
    RuntimeRequest,
    RuntimeSpec,
    RuntimeState,
    Workspace,
    WorkspaceRequest,
    WorkspaceState,
    utc_now,
)
from sandbox.domain.ports.git_platform import GitPlatformProvider
from sandbox.domain.ports.identity import BotIdentityProvider, RemoteAuthProvider
from sandbox.domain.ports.locks import LockManager
from sandbox.domain.ports.repos import RepoCacheProvider
from sandbox.domain.ports.runtimes import RuntimeProvider
from sandbox.domain.ports.stores import SandboxStore
from sandbox.domain.ports.workspaces import WorkspaceProvider


class SandboxService:
    """Coordinate durable repo caches, workspaces, and disposable runtimes.

    The service intentionally does not know how each layer is stored. Local
    `.repos`, Docker volumes, and Daytona sandboxes are all backend
    implementations behind ports. The service only owns mapping, locks,
    and lifecycle policy.
    """

    def __init__(
        self,
        *,
        workspace_provider: WorkspaceProvider,
        runtime_provider: RuntimeProvider,
        store: SandboxStore,
        locks: LockManager,
        repo_cache_provider: RepoCacheProvider | None = None,
        git_platform_provider: GitPlatformProvider | None = None,
        bot_identity_provider: BotIdentityProvider | None = None,
        remote_auth_provider: RemoteAuthProvider | None = None,
        runtime_workdir: str = "/work",
    ) -> None:
        self._workspace_provider = workspace_provider
        self._runtime_provider = runtime_provider
        self._store = store
        self._locks = locks
        self._repo_cache_provider = repo_cache_provider
        self._git_platform_provider = git_platform_provider
        self._bot_identity_provider = bot_identity_provider
        self._remote_auth_provider = remote_auth_provider
        self._runtime_workdir = runtime_workdir

    async def bot_identity_for(
        self, repo: RepoIdentity, *, user_id: str | None = None
    ) -> Author | None:
        """Public accessor used by `SandboxClient.commit` to default the
        author when the caller doesn't pass one. Returns ``None`` when
        no provider is wired so the client can fall back to git defaults.
        """
        if self._bot_identity_provider is None:
            return None
        return await self._bot_identity_provider.identity_for_repo(
            repo=repo, user_id=user_id
        )

    async def remote_auth_for(
        self, repo: RepoIdentity, *, user_id: str | None = None
    ) -> RemoteAuth | None:
        """Public accessor used by `SandboxClient.push` to inject auth
        per-call. Returns ``None`` to push anonymously (works for public
        repos, fails for private)."""
        if self._remote_auth_provider is None:
            return None
        return await self._remote_auth_provider.auth_for_remote(
            repo=repo, user_id=user_id
        )

    async def ensure_repo_cache(self, request: RepoCacheRequest) -> RepoCache:
        """Materialize the bare repo for `request.repo` and persist it.

        Idempotent on `RepoCacheRequest.key()`: concurrent callers share
        the same lock; a hit in the store short-circuits the provider
        call. Use this from parsing (P5 in the roadmap) so the cache row
        exists before the first agent call — subsequent
        ``get_or_create_workspace`` calls reuse the same on-disk bare
        repo without any extra clone.
        """
        if self._repo_cache_provider is None:
            raise RuntimeError(
                "SandboxService.ensure_repo_cache requires a "
                "repo_cache_provider; pass one to __init__"
            )
        key = request.key()
        async with self._locks.lock(f"repo_cache:{key}"):
            existing = await self._store.find_repo_cache_by_key(key)
            if existing is not None and existing.backend_kind == self._repo_cache_provider.kind:
                existing.last_used_at = utc_now()
                existing.updated_at = utc_now()
                await self._store.save_repo_cache(existing)
                return existing
            # Stale or missing — refresh through the provider. Drop a
            # stale entry from a different backend so the store doesn't
            # carry orphaned rows.
            if existing is not None:
                await self._store.delete_repo_cache(existing.id)

            cache = await self._repo_cache_provider.ensure_cache(request)
            await self._store.save_repo_cache(cache)
            return cache

    async def get_or_create_workspace(self, request: WorkspaceRequest) -> Workspace:
        key = request.key()
        async with self._locks.lock(f"workspace:{key}"):
            existing = await self._store.find_workspace_by_key(key)
            if existing is not None:
                if existing.backend_kind == self._workspace_provider.kind:
                    existing.last_used_at = utc_now()
                    existing.updated_at = utc_now()
                    await self._store.save_workspace(existing)
                    return existing
                # Stale entry from a different provider (e.g. user switched
                # SANDBOX_WORKSPACE_PROVIDER between runs, or the bridge
                # implementation changed its `kind`). The persisted
                # `WorkspaceLocation` is incompatible with the current
                # provider's `mount_for_runtime`, so we drop it and let the
                # current provider create a fresh one. Any orphan runtime
                # rows go with it so they don't dangle in the store.
                stale_runtime = await self._store.find_runtime_by_workspace(existing.id)
                if stale_runtime is not None:
                    await self._store.delete_runtime(stale_runtime.id)
                await self._store.delete_workspace(existing.id)

            workspace = await self._workspace_provider.get_or_create_workspace(request)
            await self._store.save_workspace(workspace)
            return workspace

    async def get_workspace(self, workspace_id: str) -> Workspace:
        workspace = await self._store.get_workspace(workspace_id)
        if workspace is None:
            workspace = await self._workspace_provider.get_workspace(workspace_id)
        if workspace is None:
            raise WorkspaceNotFound(f"Workspace not found: {workspace_id}")
        return workspace

    async def is_workspace_alive(self, workspace_id: str) -> bool:
        """Cheap liveness probe — used by the project-sandbox health check.

        Returns ``False`` when the workspace is unknown to the store
        (already destroyed) or when the provider's own ``is_alive``
        check fails. Doesn't raise: callers route the false result
        into a re-create rather than handling exceptions inline.
        """
        workspace = await self._store.get_workspace(workspace_id)
        if workspace is None:
            workspace = await self._workspace_provider.get_workspace(workspace_id)
        if workspace is None:
            return False
        if workspace.state is WorkspaceState.DELETED:
            return False
        return await self._workspace_provider.is_alive(workspace)

    async def acquire_session(self, request: WorkspaceRequest) -> Workspace:
        """Run the doc's Edit-Flow steps 2 and 3 in one call.

        1. Ensure the parent `RepoCache` exists and is persisted (when a
           `RepoCacheProvider` is wired). Skipped silently for backends
           that don't expose the cache port (e.g. Daytona today, until
           P4 promotes it).
        2. Get-or-create the workspace. For `EDIT`/`TASK` modes the
           workspace adapter forks a unique branch off `request.base_ref`.

        Higher-level than `get_or_create_workspace`: this is what
        agent-harness and parsing callers should use; it removes the
        burden of remembering to provision the cache first.
        """
        if self._repo_cache_provider is not None:
            await self.ensure_repo_cache(
                RepoCacheRequest(
                    repo=request.repo,
                    base_ref=request.base_ref,
                    user_id=request.user_id,
                    auth_token=request.auth_token,
                )
            )
        return await self.get_or_create_workspace(request)

    async def create_pull_request(
        self, request: PullRequestRequest
    ) -> PullRequest:
        """Open a PR via the configured ``GitPlatformProvider``.

        Workspace-side: the caller is expected to have already pushed
        ``request.head_branch`` to origin (via :meth:`SandboxClient.push`).
        This service method is purely the platform-side step. Raises
        :class:`GitPlatformNotConfigured` if no provider was wired.
        """
        if self._git_platform_provider is None:
            raise GitPlatformNotConfigured(
                "SandboxService.create_pull_request requires a "
                "git_platform_provider; pass one to __init__"
            )
        return await self._git_platform_provider.create_pull_request(request)

    async def comment_on_pull_request(
        self, request: PullRequestComment
    ) -> PullRequestCommentResult:
        """Post a comment on an existing PR via the configured platform.

        Same precondition as :meth:`create_pull_request`: the
        ``GitPlatformProvider`` has to be wired or this raises
        :class:`GitPlatformNotConfigured`. Inline (``path`` + ``line``)
        and top-level shapes are both supported; the validation lives
        in the adapter so each platform can refuse the shapes it
        doesn't natively support.
        """
        if self._git_platform_provider is None:
            raise GitPlatformNotConfigured(
                "SandboxService.comment_on_pull_request requires a "
                "git_platform_provider; pass one to __init__"
            )
        return await self._git_platform_provider.comment_on_pull_request(request)

    async def release_session(
        self, workspace_id: str, *, destroy_runtime: bool = False
    ) -> None:
        """Wind down an active session without deleting the workspace.

        Default: hibernate the runtime. The worktree (and its branch)
        survive so the next acquire_session reattaches in milliseconds.
        Pass ``destroy_runtime=True`` to free the runtime entirely; the
        workspace itself still survives — call ``destroy_workspace`` to
        actually drop it.
        """
        runtime = await self._store.find_runtime_by_workspace(workspace_id)
        if runtime is None or runtime.state is RuntimeState.DELETED:
            return
        if destroy_runtime:
            await self.destroy_runtime(runtime.id)
            return
        if runtime.state is not RuntimeState.STOPPED:
            await self.hibernate_runtime(runtime.id)

    async def get_or_create_runtime(self, request: RuntimeRequest) -> Runtime:
        workspace = await self.get_workspace(request.workspace_id)
        existing = await self._store.find_runtime_by_workspace(
            workspace.id, self._runtime_provider.kind
        )
        if existing is not None and existing.state is not RuntimeState.DELETED:
            if existing.state is RuntimeState.STOPPED:
                existing = await self._runtime_provider.start(existing)
            existing.last_used_at = utc_now()
            existing.updated_at = utc_now()
            await self._store.save_runtime(existing)
            return existing

        mount = await self._workspace_provider.mount_for_runtime(
            workspace, writable=request.writable
        )
        identity = await self.bot_identity_for(
            workspace.request.repo, user_id=workspace.request.user_id
        )
        spec = self._build_runtime_spec(request, workspace, mount, identity)
        runtime = await self._runtime_provider.create(workspace.id, spec)
        await self._store.save_runtime(runtime)
        return runtime

    async def exec(self, workspace_id: str, request: ExecRequest) -> ExecResult:
        lock_key = f"workspace-command:{workspace_id}"
        if request.command_kind.mutates_workspace:
            async with self._locks.lock(lock_key):
                return await self._exec_unlocked(workspace_id, request)
        return await self._exec_unlocked(workspace_id, request)

    # ------------------------------------------------------------------
    # File ops dispatched to the runtime's native fs when available.
    #
    # Backends that ship typed fs APIs (Daytona's ``sandbox.fs``) expose
    # ``read_bytes`` / ``write_bytes`` / ``list_dir_native`` on the runtime
    # provider; we prefer those because the generic exec path can't pipe
    # stdin to ``sandbox.process.exec`` (the SDK has no stdin parameter,
    # so ``cat > file`` writes silently fail). For backends that don't
    # expose them, callers should fall back to the exec path themselves —
    # we don't synthesise one here because the right encoding (base64
    # vs printf vs heredoc) depends on the backend's quoting rules.
    # ------------------------------------------------------------------
    async def fs_read_file(self, workspace_id: str, path: str) -> bytes | None:
        runtime, native = await self._native_fs("read_bytes", workspace_id)
        if native is None:
            return None
        return await native(runtime, path)

    async def fs_write_file(
        self, workspace_id: str, path: str, content: bytes
    ) -> bool:
        """Write via runtime native fs if exposed; return False otherwise.

        Returning a flag (rather than raising) lets the caller branch on
        ``False`` and use its own write path — needed because
        ``SandboxClient.write_file`` already has a local-fs fast path and
        a generic exec fallback for backends that lack native fs.
        """
        async with self._locks.lock(f"workspace-command:{workspace_id}"):
            runtime, native = await self._native_fs("write_bytes", workspace_id)
            if native is None:
                return False
            await native(runtime, path, content)
            workspace = await self.get_workspace(workspace_id)
            workspace.dirty = True
            workspace.last_used_at = utc_now()
            workspace.updated_at = utc_now()
            await self._store.save_workspace(workspace)
            return True

    async def fs_list_dir(
        self, workspace_id: str, path: str
    ) -> list[tuple[str, bool, int | None]] | None:
        runtime, native = await self._native_fs("list_dir_native", workspace_id)
        if native is None:
            return None
        return await native(runtime, path)

    async def _native_fs(self, attr: str, workspace_id: str):
        """Resolve ``runtime`` plus a runtime-provider method by name, if any.

        Caller passes the name (``"read_bytes"`` / ``"write_bytes"`` /
        ``"list_dir_native"``); we return ``(runtime, callable)`` or
        ``(runtime, None)`` when the method isn't exposed. Runtime
        creation and start mirror :meth:`_exec_unlocked` so a cold-start
        write doesn't bypass runtime provisioning.
        """
        method = getattr(self._runtime_provider, attr, None)
        if method is None:
            return None, None
        runtime = await self._store.find_runtime_by_workspace(
            workspace_id, self._runtime_provider.kind
        )
        if runtime is None:
            runtime = await self.get_or_create_runtime(RuntimeRequest(workspace_id))
        if runtime.state is RuntimeState.STOPPED:
            runtime = await self._runtime_provider.start(runtime)
        if runtime.state is RuntimeState.DELETED:
            raise RuntimeNotFound(f"Runtime was deleted: {runtime.id}")
        return runtime, method

    async def hibernate_runtime(self, runtime_id: str) -> None:
        runtime = await self._store.get_runtime(runtime_id)
        if runtime is None:
            provider_runtime = await self._runtime_provider.get(runtime_id)
            if provider_runtime is None:
                raise RuntimeNotFound(f"Runtime not found: {runtime_id}")
            runtime = provider_runtime
        runtime = await self._runtime_provider.stop(runtime)
        runtime.updated_at = utc_now()
        await self._store.save_runtime(runtime)

    async def destroy_runtime(self, runtime_id: str) -> None:
        runtime = await self._store.get_runtime(runtime_id)
        if runtime is None:
            provider_runtime = await self._runtime_provider.get(runtime_id)
            if provider_runtime is None:
                await self._store.delete_runtime(runtime_id)
                return
            runtime = provider_runtime
        await self._runtime_provider.destroy(runtime)
        await self._store.delete_runtime(runtime_id)

    async def destroy_workspace(
        self, workspace_id: str, *, destroy_runtime: bool = True
    ) -> None:
        async with self._locks.lock(f"workspace:{workspace_id}"):
            workspace = await self.get_workspace(workspace_id)
            if destroy_runtime:
                runtime = await self._store.find_runtime_by_workspace(workspace_id)
                if runtime is not None:
                    await self.destroy_runtime(runtime.id)
            await self._workspace_provider.delete_workspace(workspace)
            await self._store.delete_workspace(workspace_id)

    async def _exec_unlocked(
        self, workspace_id: str, request: ExecRequest
    ) -> ExecResult:
        runtime = await self._store.find_runtime_by_workspace(
            workspace_id, self._runtime_provider.kind
        )
        if runtime is None:
            runtime = await self.get_or_create_runtime(RuntimeRequest(workspace_id))
        if runtime.state is RuntimeState.STOPPED:
            runtime = await self._runtime_provider.start(runtime)
        if runtime.state is RuntimeState.DELETED:
            raise RuntimeNotFound(f"Runtime was deleted: {runtime.id}")

        result = await self._runtime_provider.exec(runtime, request)
        runtime.last_used_at = utc_now()
        runtime.updated_at = utc_now()
        await self._store.save_runtime(runtime)

        workspace = await self.get_workspace(workspace_id)
        workspace.last_used_at = utc_now()
        workspace.updated_at = utc_now()
        if request.command_kind.mutates_workspace and result.exit_code == 0:
            workspace.dirty = True
        await self._store.save_workspace(workspace)
        return result

    def _build_runtime_spec(
        self,
        request: RuntimeRequest,
        workspace: Workspace,
        mount: Mount,
        identity: Author | None = None,
    ) -> RuntimeSpec:
        workdir = mount.target or self._runtime_workdir
        network = request.network
        if not isinstance(network, NetworkMode):
            network = NetworkMode(str(network))
        labels = {
            "workspace_id": request.workspace_id,
            "workspace_backend_kind": workspace.backend_kind,
        }
        if workspace.location.backend_workspace_id:
            labels["workspace_backend_id"] = workspace.location.backend_workspace_id
        # Stash the project key on the runtime so the Daytona adapter can
        # recover when its persisted sandbox id is gone (e.g. the user
        # deleted the sandbox out-of-band, or it expired). Without these
        # labels the runtime has no way to map its dead backend_runtime_id
        # back to a (user, project) tuple for label-based lookup.
        if workspace.request.user_id:
            labels["user_id"] = workspace.request.user_id
        if workspace.request.project_id:
            labels["project_id"] = workspace.request.project_id
        env = dict(request.env)
        # Stamp the bot identity into the runtime env so every git
        # commit run inside this runtime — through ``client.commit`` or
        # raw ``sandbox_shell git commit`` — is attributed to the same
        # author. The caller can still override per-exec via
        # ``ExecRequest.env`` (per-exec env wins in
        # ``LocalSubprocessRuntimeProvider._build_env``).
        if identity is not None:
            env.setdefault("GIT_AUTHOR_NAME", identity.name)
            env.setdefault("GIT_AUTHOR_EMAIL", identity.email)
            env.setdefault("GIT_COMMITTER_NAME", identity.name)
            env.setdefault("GIT_COMMITTER_EMAIL", identity.email)
        return RuntimeSpec(
            image=request.image,
            workdir=workdir,
            mounts=(mount,),
            env=env,
            resources=request.resources,
            network=network,
            labels=labels,
        )
