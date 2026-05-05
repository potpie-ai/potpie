"""End-to-end integration tests for the Daytona sandbox backend.

These talk to a real Daytona stack and create real sandboxes; they're
gated behind ``DAYTONA_API_URL`` + ``DAYTONA_API_KEY`` so the suite
remains green in environments without the stack.

The tests use the public ``octocat/Hello-World`` repo so no GitHub
token is needed. Every test that creates a sandbox tears it down in a
``finally`` block so repeated runs don't leak resources.

Coverage:

* Full ``acquire_session`` -> exec -> commit -> destroy flow
* One-sandbox-per-(user,project): two branches share, two projects don't
* Conversation isolation: writes in conv-A invisible from conv-B
* Worker-restart recovery via Daytona labels (P4 invariant)
* Capability gating (ANALYSIS read-only, EDIT writable+isolated)
* PR refusal on read-only handle
* Ref-validation rejects ``..`` / newlines before they hit shell exec
* ``acquire_session`` skips ``ensure_repo_cache`` silently when no
  ``RepoCacheProvider`` is wired (Daytona today)
"""

from __future__ import annotations

import os
import secrets
from typing import Any, AsyncIterator

import pytest

from sandbox import (
    Capabilities,
    CommandKind,
    SandboxClient,
    SandboxOpError,
    WorkspaceMode,
)


REPO_NAME = "octocat/Hello-World"
REPO_URL = "https://github.com/octocat/Hello-World.git"
BASE_REF = "master"


pytestmark = pytest.mark.skipif(
    not (os.getenv("DAYTONA_API_URL") and os.getenv("DAYTONA_API_KEY")),
    reason="DAYTONA_API_URL and DAYTONA_API_KEY must be set",
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
async def _safe_destroy_workspace(client: SandboxClient, handle) -> None:
    """Best-effort: never let teardown failure mask a real assertion."""
    try:
        await client.destroy_workspace(handle)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: destroy_workspace failed: {exc!r}")


async def _delete_daytona_sandbox(client: SandboxClient, sandbox_id: str) -> None:
    """Drop the Daytona sandbox itself.

    ``destroy_workspace`` only drops the worktree; the parent sandbox
    survives because it's shared across the project. Tests that
    actually want to free the sandbox (so the next test starts clean)
    have to call this explicitly.
    """
    workspace_provider = client.container.workspace_provider
    sandbox = workspace_provider._sandboxes.get(sandbox_id)  # type: ignore[attr-defined]
    if sandbox is None:
        try:
            sandbox = workspace_provider.client.get(sandbox_id)
        except Exception:
            sandbox = None
    if sandbox is not None and hasattr(sandbox, "delete"):
        try:
            sandbox.delete()
        except Exception as exc:  # noqa: BLE001
            print(f"warning: sandbox.delete failed: {exc!r}")


def _unique_project() -> str:
    """Per-test project_id so tests don't share Daytona sandboxes."""
    return f"p-{secrets.token_hex(4)}"


# ----------------------------------------------------------------------
# Workspace lifecycle
# ----------------------------------------------------------------------
class TestDaytonaLifecycle:
    """Doc invariants for the Daytona backend (see
    ``app/src/sandbox/sandbox/adapters/outbound/daytona/provider.py``):

    - One sandbox per ``(user_id, project_id)``.
    - Branches inside that sandbox are git worktrees; the bare clone
      is shared so a second branch costs no extra Daytona container.
    - Worktree path includes ``user``, ``scope``, ``branch`` so two
      conversation_ids on the same branch don't collide.
    """

    async def test_acquire_session_creates_remote_worktree(
        self, daytona_client: SandboxClient
    ) -> None:
        project_id = _unique_project()
        handle = await daytona_client.acquire_session(
            user_id="u-lifecycle",
            project_id=project_id,
            repo=REPO_NAME,
            repo_url=REPO_URL,
            branch="agent/edits-life",
            base_ref=BASE_REF,
            create_branch=True,
            mode=WorkspaceMode.EDIT,
            conversation_id="life",
        )
        try:
            assert handle.backend_kind == "daytona"
            # Daytona has no host filesystem path — it's all remote.
            assert handle.local_path is None
            assert handle.remote_path is not None
            assert "/worktrees/" in handle.remote_path
            assert handle.branch == "agent/edits-life"

            # Verify the workspace is wired up by reading the README.
            cat = await daytona_client.exec(
                handle, ["cat", "README"], command_kind=CommandKind.READ
            )
            assert cat.exit_code == 0
            assert cat.stdout, "Hello-World fixture has a non-empty README"
        finally:
            sandbox_id = handle.workspace_id
            ws = await daytona_client.container.service.get_workspace(
                handle.workspace_id
            )
            await _safe_destroy_workspace(daytona_client, handle)
            # Free the underlying sandbox too so we don't leak.
            await _delete_daytona_sandbox(
                daytona_client, ws.location.backend_workspace_id
            )

    async def test_writes_persist_within_session(
        self, daytona_client: SandboxClient
    ) -> None:
        project_id = _unique_project()
        handle = await daytona_client.acquire_session(
            user_id="u-write", project_id=project_id,
            repo=REPO_NAME, repo_url=REPO_URL,
            branch="agent/edits-write", base_ref=BASE_REF, create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="write",
        )
        try:
            marker = f"potpie-{secrets.token_hex(2)}"
            write = await daytona_client.exec(
                handle,
                ["sh", "-c", f"printf {marker} > marker.txt"],
                command_kind=CommandKind.WRITE,
            )
            assert write.exit_code == 0

            verify = await daytona_client.exec(
                handle, ["cat", "marker.txt"], command_kind=CommandKind.READ
            )
            assert verify.exit_code == 0
            assert verify.stdout.strip() == marker.encode()

            # And the workspace's dirty flag flipped.
            ws = await daytona_client.container.service.get_workspace(
                handle.workspace_id
            )
            assert ws.dirty is True
        finally:
            ws = await daytona_client.container.service.get_workspace(
                handle.workspace_id
            )
            sandbox_id = ws.location.backend_workspace_id
            await _safe_destroy_workspace(daytona_client, handle)
            await _delete_daytona_sandbox(daytona_client, sandbox_id)

    async def test_acquire_session_is_idempotent(
        self, daytona_client: SandboxClient
    ) -> None:
        """Two calls with the same key must return the same workspace AND
        the same sandbox id. This is the per-process equivalent of the
        worker-restart test below."""
        project_id = _unique_project()
        kwargs = dict(
            user_id="u-idem", project_id=project_id,
            repo=REPO_NAME, repo_url=REPO_URL,
            branch="agent/edits-idem", base_ref=BASE_REF, create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="idem",
        )
        h1 = await daytona_client.acquire_session(**kwargs)
        try:
            h2 = await daytona_client.acquire_session(**kwargs)
            assert h1.workspace_id == h2.workspace_id
            assert h1.remote_path == h2.remote_path
        finally:
            ws = await daytona_client.container.service.get_workspace(
                h1.workspace_id
            )
            sandbox_id = ws.location.backend_workspace_id
            await _safe_destroy_workspace(daytona_client, h1)
            await _delete_daytona_sandbox(daytona_client, sandbox_id)


# ----------------------------------------------------------------------
# Sandbox sharing within a project
# ----------------------------------------------------------------------
class TestProjectSharing:
    async def test_two_branches_share_one_sandbox(
        self, daytona_client: SandboxClient
    ) -> None:
        """The Phase 3 invariant: two branches on the same project map
        to the same Daytona sandbox id and different worktree paths.

        This is the cost-control claim — without it, every branch would
        spin up a fresh container."""
        project_id = _unique_project()
        h_a = await daytona_client.acquire_session(
            user_id="u-share", project_id=project_id,
            repo=REPO_NAME, repo_url=REPO_URL,
            branch="agent/edits-a", base_ref=BASE_REF, create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="share-a",
        )
        h_b = await daytona_client.acquire_session(
            user_id="u-share", project_id=project_id,
            repo=REPO_NAME, repo_url=REPO_URL,
            branch="agent/edits-b", base_ref=BASE_REF, create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="share-b",
        )
        sandbox_id: str | None = None
        try:
            ws_a = await daytona_client.container.service.get_workspace(
                h_a.workspace_id
            )
            ws_b = await daytona_client.container.service.get_workspace(
                h_b.workspace_id
            )
            sandbox_id = ws_a.location.backend_workspace_id
            assert ws_a.location.backend_workspace_id == ws_b.location.backend_workspace_id
            assert ws_a.location.remote_path != ws_b.location.remote_path
            assert h_a.workspace_id != h_b.workspace_id
        finally:
            await _safe_destroy_workspace(daytona_client, h_a)
            await _safe_destroy_workspace(daytona_client, h_b)
            if sandbox_id:
                await _delete_daytona_sandbox(daytona_client, sandbox_id)

    async def test_two_projects_get_distinct_sandboxes(
        self, daytona_client: SandboxClient
    ) -> None:
        project_a = _unique_project()
        project_b = _unique_project()
        h_a = await daytona_client.acquire_session(
            user_id="u-projects", project_id=project_a,
            repo=REPO_NAME, repo_url=REPO_URL,
            branch="agent/edits-pa", base_ref=BASE_REF, create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="proj-a",
        )
        h_b = await daytona_client.acquire_session(
            user_id="u-projects", project_id=project_b,
            repo=REPO_NAME, repo_url=REPO_URL,
            branch="agent/edits-pb", base_ref=BASE_REF, create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="proj-b",
        )
        sandbox_a = sandbox_b = None
        try:
            ws_a = await daytona_client.container.service.get_workspace(
                h_a.workspace_id
            )
            ws_b = await daytona_client.container.service.get_workspace(
                h_b.workspace_id
            )
            sandbox_a = ws_a.location.backend_workspace_id
            sandbox_b = ws_b.location.backend_workspace_id
            assert sandbox_a != sandbox_b, (
                "different projects must NOT share a Daytona sandbox"
            )
        finally:
            await _safe_destroy_workspace(daytona_client, h_a)
            await _safe_destroy_workspace(daytona_client, h_b)
            if sandbox_a:
                await _delete_daytona_sandbox(daytona_client, sandbox_a)
            if sandbox_b:
                await _delete_daytona_sandbox(daytona_client, sandbox_b)


# ----------------------------------------------------------------------
# Conversation isolation (the P4 invariant)
# ----------------------------------------------------------------------
class TestDaytonaConversationIsolation:
    async def test_writes_in_one_conversation_invisible_in_another(
        self, daytona_client: SandboxClient
    ) -> None:
        """Two conversations on the same project share a sandbox but
        must NOT share files — they live on different branches inside
        different worktrees.

        Without this, an agent edit in conversation A would surface in
        conversation B's `git status`, which is a security leak between
        independent agent sessions.
        """
        project_id = _unique_project()
        h_a = await daytona_client.acquire_session(
            user_id="u-iso", project_id=project_id,
            repo=REPO_NAME, repo_url=REPO_URL,
            branch="agent/edits-conv-a", base_ref=BASE_REF, create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="conv-a",
        )
        h_b = await daytona_client.acquire_session(
            user_id="u-iso", project_id=project_id,
            repo=REPO_NAME, repo_url=REPO_URL,
            branch="agent/edits-conv-b", base_ref=BASE_REF, create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="conv-b",
        )
        sandbox_id = None
        try:
            # Write a unique marker only in conv-A.
            marker = f"only-in-a-{secrets.token_hex(2)}"
            write = await daytona_client.exec(
                h_a,
                ["sh", "-c", f"printf {marker} > ONLY_IN_A.txt"],
                command_kind=CommandKind.WRITE,
            )
            assert write.exit_code == 0

            # conv-B's worktree must not see the file.
            check = await daytona_client.exec(
                h_b, ["ls", "ONLY_IN_A.txt"], command_kind=CommandKind.READ
            )
            assert check.exit_code != 0, (
                "conv-B should not see conv-A's file — branch isolation broken"
            )

            ws_a = await daytona_client.container.service.get_workspace(
                h_a.workspace_id
            )
            sandbox_id = ws_a.location.backend_workspace_id
        finally:
            await _safe_destroy_workspace(daytona_client, h_a)
            await _safe_destroy_workspace(daytona_client, h_b)
            if sandbox_id:
                await _delete_daytona_sandbox(daytona_client, sandbox_id)


# ----------------------------------------------------------------------
# Worker restart recovery (the doc bug-bash test)
# ----------------------------------------------------------------------
class TestWorkerRestartRecovery:
    async def test_fresh_provider_adopts_existing_sandbox(
        self, metadata_path
    ) -> None:
        """Reproduces the historical bug: worker A creates a sandbox and
        commits work; worker B starts up cold and must adopt the same
        sandbox via Daytona labels rather than spin up a fresh one
        (which would orphan worker A's commits).

        See ``DaytonaWorkspaceProvider._recover_project_sandbox``.
        """
        from daytona import Daytona, DaytonaConfig

        from sandbox import SandboxClient
        from sandbox.adapters.outbound.daytona.provider import (
            DaytonaRuntimeProvider,
            DaytonaWorkspaceProvider,
        )
        from sandbox.adapters.outbound.file.json_store import JsonSandboxStore
        from sandbox.adapters.outbound.memory.eviction import NoOpEvictionPolicy
        from sandbox.adapters.outbound.memory.locks import InMemoryLockManager
        from sandbox.application.services.sandbox_service import SandboxService
        from sandbox.bootstrap.container import SandboxContainer

        config = DaytonaConfig(
            api_url=os.environ["DAYTONA_API_URL"],
            api_key=os.environ["DAYTONA_API_KEY"],
        )
        # Single SDK client shared across the two providers — the
        # restart simulation is "fresh in-memory state, same Daytona".
        sdk = Daytona(config)

        def _build() -> SandboxClient:
            wp = DaytonaWorkspaceProvider(
                client_factory=lambda: sdk,
                snapshot=os.getenv("DAYTONA_SNAPSHOT") or "potpie/agent-sandbox:0.1.0",
                workspace_root=os.getenv("DAYTONA_WORKSPACE_ROOT", "/home/agent/work"),
            )
            rp = DaytonaRuntimeProvider(wp)
            store = JsonSandboxStore(metadata_path)
            locks = InMemoryLockManager()
            svc = SandboxService(
                workspace_provider=wp, runtime_provider=rp,
                store=store, locks=locks,
                repo_cache_provider=None,
            )
            container = SandboxContainer(
                workspace_provider=wp, runtime_provider=rp,
                store=store, locks=locks, service=svc,
                eviction=NoOpEvictionPolicy(), repo_cache_provider=None,
            )
            return SandboxClient.from_container(container)

        project_id = _unique_project()
        conv = f"recovery-{secrets.token_hex(2)}"
        kwargs = dict(
            user_id="u-recovery", project_id=project_id,
            repo=REPO_NAME, repo_url=REPO_URL,
            branch=f"agent/edits-{conv}", base_ref=BASE_REF, create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id=conv,
        )

        marker = f"persist-{secrets.token_hex(2)}"
        client_a = _build()
        h_a = await client_a.acquire_session(**kwargs)
        sandbox_id = (
            await client_a.container.service.get_workspace(h_a.workspace_id)
        ).location.backend_workspace_id

        try:
            write = await client_a.exec(
                h_a,
                ["sh", "-c", f"printf {marker} > marker.txt"],
                command_kind=CommandKind.WRITE,
            )
            assert write.exit_code == 0

            # Worker B comes up cold. The metadata.json from A is reused
            # so the workspace row is still in the store — but the live
            # provider state is fresh, so the *sandbox* lookup goes
            # through ``_recover_project_sandbox``.
            client_b = _build()
            h_b = await client_b.acquire_session(**kwargs)
            ws_b = await client_b.container.service.get_workspace(h_b.workspace_id)
            assert ws_b.location.backend_workspace_id == sandbox_id, (
                f"recovery failed: got fresh sandbox "
                f"{ws_b.location.backend_workspace_id} instead of "
                f"adopting {sandbox_id}"
            )

            # And the marker file written by A is visible to B.
            read = await client_b.exec(
                h_b, ["cat", "marker.txt"], command_kind=CommandKind.READ
            )
            assert read.stdout.strip() == marker.encode()
        finally:
            await _safe_destroy_workspace(client_a, h_a)
            if sandbox_id:
                await _delete_daytona_sandbox(client_a, sandbox_id)


# ----------------------------------------------------------------------
# Capability gating
# ----------------------------------------------------------------------
class TestDaytonaCapabilities:
    async def test_analysis_handle_is_readonly(
        self, daytona_client: SandboxClient
    ) -> None:
        project_id = _unique_project()
        analysis = await daytona_client.acquire_session(
            user_id="u-cap", project_id=project_id,
            repo=REPO_NAME, repo_url=REPO_URL,
            branch=BASE_REF, base_ref=BASE_REF,
            mode=WorkspaceMode.ANALYSIS,
        )
        sandbox_id = None
        try:
            assert analysis.capabilities == Capabilities(
                writable=False, isolated=False, persistent=True
            )
            # PR refused on read-only handle.
            with pytest.raises(SandboxOpError, match="writable"):
                await daytona_client.create_pull_request(
                    analysis,
                    repo=REPO_NAME,
                    title="should not happen",
                    body="...",
                    base_branch=BASE_REF,
                )
            ws = await daytona_client.container.service.get_workspace(
                analysis.workspace_id
            )
            sandbox_id = ws.location.backend_workspace_id
        finally:
            await _safe_destroy_workspace(daytona_client, analysis)
            if sandbox_id:
                await _delete_daytona_sandbox(daytona_client, sandbox_id)

    async def test_edit_handle_is_writable_and_isolated(
        self, daytona_client: SandboxClient
    ) -> None:
        project_id = _unique_project()
        handle = await daytona_client.acquire_session(
            user_id="u-cap-edit", project_id=project_id,
            repo=REPO_NAME, repo_url=REPO_URL,
            branch="agent/edits-cap", base_ref=BASE_REF, create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="cap",
        )
        sandbox_id = None
        try:
            assert handle.capabilities == Capabilities(
                writable=True, isolated=True, persistent=True
            )
            ws = await daytona_client.container.service.get_workspace(
                handle.workspace_id
            )
            sandbox_id = ws.location.backend_workspace_id
        finally:
            await _safe_destroy_workspace(daytona_client, handle)
            if sandbox_id:
                await _delete_daytona_sandbox(daytona_client, sandbox_id)


# ----------------------------------------------------------------------
# Ref injection
# ----------------------------------------------------------------------
class TestRefValidation:
    """The Daytona adapter shells ``base_ref`` and ``branch_name`` into
    ``sandbox.process.exec`` strings. Both are validated up front to
    block injection (newlines, ``..``)."""

    @pytest.mark.parametrize("bad_ref", ["main\nrm -rf /", "main..origin"])
    async def test_bad_base_ref_rejected_at_boundary(
        self, daytona_client: SandboxClient, bad_ref: str
    ) -> None:
        with pytest.raises(ValueError, match="unsafe"):
            await daytona_client.acquire_session(
                user_id="u-ref", project_id=_unique_project(),
                repo=REPO_NAME, repo_url=REPO_URL,
                branch="agent/safe", base_ref=bad_ref, create_branch=True,
                mode=WorkspaceMode.EDIT, conversation_id="ref-bad-base",
            )

    @pytest.mark.parametrize("bad_branch", ["evil\necho x", "back/../escape"])
    async def test_bad_branch_name_rejected_at_boundary(
        self, daytona_client: SandboxClient, bad_branch: str
    ) -> None:
        with pytest.raises(ValueError, match="unsafe"):
            await daytona_client.acquire_session(
                user_id="u-ref", project_id=_unique_project(),
                repo=REPO_NAME, repo_url=REPO_URL,
                branch=bad_branch, base_ref=BASE_REF, create_branch=True,
                mode=WorkspaceMode.EDIT, conversation_id="ref-bad-branch",
            )


# ----------------------------------------------------------------------
# acquire_session without RepoCacheProvider
# ----------------------------------------------------------------------
class TestRepoCacheBehavior:
    async def test_acquire_session_skips_ensure_cache_when_unwired(
        self, daytona_client: SandboxClient
    ) -> None:
        """Daytona doesn't expose a separate ``RepoCacheProvider``
        today (P4 in the roadmap), so the bootstrap leaves it ``None``.
        ``acquire_session`` must NOT raise ``RuntimeError`` in that case
        — it has to silently skip the cache step and go straight to
        workspace creation. The local provider's behavior is the
        opposite (raises) so we pin the silent-skip explicitly here."""
        # We expect this to succeed despite no RepoCacheProvider being
        # wired. If a future change accidentally raises here, this test
        # catches it.
        project_id = _unique_project()
        handle = await daytona_client.acquire_session(
            user_id="u-cache", project_id=project_id,
            repo=REPO_NAME, repo_url=REPO_URL,
            branch="agent/edits-cache-skip", base_ref=BASE_REF, create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="cache-skip",
        )
        sandbox_id = None
        try:
            ws = await daytona_client.container.service.get_workspace(
                handle.workspace_id
            )
            sandbox_id = ws.location.backend_workspace_id
            # And ensure_repo_cache via the public API DOES raise — pin
            # that complementary behavior so a future P4 implementation
            # is forced to update both call sites.
            with pytest.raises(RuntimeError, match="repo_cache_provider"):
                await daytona_client.ensure_repo_cache(
                    repo=REPO_NAME, base_ref=BASE_REF, repo_url=REPO_URL
                )
        finally:
            await _safe_destroy_workspace(daytona_client, handle)
            if sandbox_id:
                await _delete_daytona_sandbox(daytona_client, sandbox_id)
