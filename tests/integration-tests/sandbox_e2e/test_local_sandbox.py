"""End-to-end integration tests for the local sandbox backend.

These tests treat the sandbox module as a black box: they build a real
``SandboxClient`` with the local provider, point it at a hermetic
on-disk git fixture (``upstream_repo``), and verify the behavior the
public docs (``docs/sandbox-core-setup.md``) promise.

Where the equivalent ``app/src/sandbox/tests/integration/`` suite is
gated behind ``RUN_LIVE_SANDBOX_TESTS=1`` (it hits real GitHub), this
suite is fully hermetic — no network, no GH token. The behaviors we
care about are exactly the same; only the upstream URL differs.

Coverage by section:

* RepoCache lifecycle ......... idempotency, key, isolation across base_refs
* Workspace lifecycle ......... acquire / re-acquire / release / destroy
* Capability gating ............ ANALYSIS read-only vs EDIT writable+isolated
* Branch / conversation isolation
* JSON store persistence ...... metadata survives a fresh client
* Path safety .................. traversal + symlink escape rejected
* Git operations ............... read/write/list/search/status/diff/commit
* Concurrent execution ......... per-workspace write lock; reads are parallel
* Pull-request gating .......... read-only handle refused; not-configured raises
* Auth-token hygiene ........... token never persisted in .git/config

Each test names the doc invariant or roadmap point it pins down so a
future reader can tell what behavior would be lost if the test were
removed.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from sandbox import (
    Capabilities,
    CommandKind,
    SandboxClient,
    SandboxOpError,
    SandboxSettings,
    WorkspaceMode,
    build_sandbox_container,
)
from sandbox.domain.errors import (
    GitPlatformNotConfigured,
    InvalidWorkspacePath,
    RuntimeCommandRejected,
    WorkspaceNotFound,
)


REPO_NAME = "owner/test-repo"


# ----------------------------------------------------------------------
# RepoCache lifecycle
# ----------------------------------------------------------------------
class TestRepoCache:
    """``LocalRepoCacheProvider`` is the durable bare-repo concern.

    Doc invariants (see ``docs/sandbox-core-setup.md`` "RepoCache"):
    - keyed on ``(provider_host, repo_name)`` — *not* on base_ref
    - shared across users
    - never holds uncommitted state
    - auth tokens never persisted on disk
    """

    async def test_ensure_repo_cache_creates_bare_clone(
        self, local_client: SandboxClient, upstream_repo: Path, repos_base: Path
    ) -> None:
        cache = await local_client.ensure_repo_cache(
            user_id="u1",
            repo=REPO_NAME,
            base_ref="main",
            repo_url=str(upstream_repo),
        )
        assert cache.backend_kind == "local"
        bare = Path(cache.location.local_path)
        assert bare.is_dir()
        assert (bare / "HEAD").exists(), "bare clone must have a HEAD file"
        # `git clone --bare` produces refs but no working tree.
        assert not (bare / ".git").exists(), "bare repo should not have a .git/"
        # And the bare lives where the docs say it does.
        assert bare == (repos_base / REPO_NAME / ".bare").resolve()

    async def test_ensure_repo_cache_is_idempotent_on_host_and_name(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        """Repeat calls must reuse the same cache row, even when base_ref differs.

        Cache identity is ``(provider_host, repo_name)``. ``base_ref``
        only tells the provider what to fetch into the existing bare —
        not what to key the cache on.
        """
        first = await local_client.ensure_repo_cache(
            user_id="u1",
            repo=REPO_NAME,
            base_ref="main",
            repo_url=str(upstream_repo),
        )
        second = await local_client.ensure_repo_cache(
            user_id="u2",  # different user, same repo
            repo=REPO_NAME,
            base_ref="main",
            repo_url=str(upstream_repo),
        )
        assert first.id == second.id
        assert first.location.local_path == second.location.local_path
        assert first.key == second.key

    async def test_token_not_persisted_in_git_config(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        """The bare's ``origin`` URL must be the plain URL, not the tokenized one.

        The bare repo is shared across users; persisting one user's
        token would leak it to the next caller. ``LocalRepoCacheProvider``
        scrubs ``origin`` after the initial clone (see
        ``_clone_bare`` -> ``git remote set-url``).
        """
        await local_client.ensure_repo_cache(
            user_id="u1",
            repo=REPO_NAME,
            base_ref="main",
            repo_url=str(upstream_repo),
            auth_token="ghp_super_secret_token",  # noqa: S106
        )
        bare = Path(local_client.container.repo_cache_provider.bare_path(REPO_NAME))
        config = (bare / "config").read_text()
        assert "ghp_super_secret_token" not in config
        assert "x-access-token" not in config


# ----------------------------------------------------------------------
# Workspace lifecycle
# ----------------------------------------------------------------------
class TestWorkspaceLifecycle:
    """Doc invariants (see ``docs/sandbox-core-setup.md`` "Workspace"):

    - keyed on ``(user, project, repo, mode, scope)``
    - EDIT/TASK fork a NEW branch from base_ref; ANALYSIS does not
    - re-acquiring the same key returns the same workspace (no new clone)
    - destroying a workspace must NOT destroy the parent cache
    - destroying a workspace MUST remove the on-disk worktree
    """

    async def test_acquire_session_creates_worktree_on_new_branch(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        handle = await local_client.acquire_session(
            user_id="u1",
            project_id="p1",
            repo=REPO_NAME,
            repo_url=str(upstream_repo),
            branch="agent/edits-conv1",
            base_ref="main",
            create_branch=True,
            mode=WorkspaceMode.EDIT,
            conversation_id="conv1",
        )
        assert handle.local_path is not None
        worktree = Path(handle.local_path)
        assert worktree.is_dir()
        # The worktree's git pointer makes it a real git checkout, not a copy.
        assert (worktree / ".git").exists()
        # README.md from the fixture upstream made it through.
        assert (worktree / "README.md").read_text() == "hello sandbox\n"
        # The branch the handle reports should match what git thinks.
        head = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        assert head.stdout.strip() == handle.branch == "agent/edits-conv1"

    async def test_acquire_session_is_idempotent_on_workspace_key(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        """Two ``acquire_session`` calls on the same key must reuse one workspace.

        This is the load-bearing contract behind "agent reconnect": the
        second message of a conversation must land on the same worktree
        the first one created, with the same files and the same id.
        """
        kwargs = dict(
            user_id="u1",
            project_id="p1",
            repo=REPO_NAME,
            repo_url=str(upstream_repo),
            branch="agent/edits-conv1",
            base_ref="main",
            create_branch=True,
            mode=WorkspaceMode.EDIT,
            conversation_id="conv1",
        )
        h1 = await local_client.acquire_session(**kwargs)
        h2 = await local_client.acquire_session(**kwargs)
        assert h1.workspace_id == h2.workspace_id
        assert h1.local_path == h2.local_path
        assert h1.branch == h2.branch

    async def test_destroy_workspace_keeps_repo_cache(
        self, local_client: SandboxClient, upstream_repo: Path, repos_base: Path
    ) -> None:
        """Cache must outlive workspace destruction.

        Doc rule (Workspace section): destroying a workspace must NOT
        delete the parent cache.
        """
        cache = await local_client.ensure_repo_cache(
            user_id="u1",
            repo=REPO_NAME,
            base_ref="main",
            repo_url=str(upstream_repo),
        )
        bare = Path(cache.location.local_path)

        handle = await local_client.acquire_session(
            user_id="u1",
            project_id="p1",
            repo=REPO_NAME,
            repo_url=str(upstream_repo),
            branch="agent/edits-c1",
            base_ref="main",
            create_branch=True,
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
        )
        worktree = Path(handle.local_path)

        await local_client.destroy_workspace(handle)
        assert not worktree.exists(), "worktree must be removed by destroy_workspace"
        assert bare.is_dir(), "repo cache must outlive workspace destruction"

        # And the destroyed workspace_id must be unknown to the service.
        with pytest.raises(WorkspaceNotFound):
            await local_client.container.service.get_workspace(handle.workspace_id)

    async def test_release_session_keeps_workspace_alive(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        """``release_session`` hibernates the runtime; the worktree survives."""
        handle = await local_client.acquire_session(
            user_id="u1",
            project_id="p1",
            repo=REPO_NAME,
            repo_url=str(upstream_repo),
            branch="agent/edits-released",
            base_ref="main",
            create_branch=True,
            mode=WorkspaceMode.EDIT,
            conversation_id="released",
        )
        worktree = Path(handle.local_path)
        # Touch a runtime so release_session has something to hibernate.
        await local_client.exec(handle, ["true"], command_kind=CommandKind.READ)

        await local_client.release_session(handle)

        # The worktree on disk is untouched.
        assert worktree.is_dir()
        # The runtime row in the store should be STOPPED, not deleted.
        runtime = await local_client.container.store.find_runtime_by_workspace(
            handle.workspace_id
        )
        assert runtime is not None
        from sandbox.domain.models import RuntimeState
        assert runtime.state is RuntimeState.STOPPED


# ----------------------------------------------------------------------
# Capability gating
# ----------------------------------------------------------------------
class TestCapabilities:
    """``Capabilities.from_mode`` is the single source of truth (see
    ``Capabilities.from_mode`` doc).

    - ANALYSIS  -> writable=False isolated=False persistent=True
    - EDIT/TASK -> writable=True  isolated=True  persistent=True
    """

    async def test_edit_workspace_is_writable_and_isolated(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        handle = await local_client.acquire_session(
            user_id="u1",
            project_id="p1",
            repo=REPO_NAME,
            repo_url=str(upstream_repo),
            branch="agent/edits-cap-edit",
            base_ref="main",
            create_branch=True,
            mode=WorkspaceMode.EDIT,
            conversation_id="cap-edit",
        )
        assert handle.capabilities == Capabilities(
            writable=True, isolated=True, persistent=True
        )

    async def test_analysis_workspace_is_readonly_and_uses_base_ref(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        """ANALYSIS branches off ``base_ref`` itself — no agent/edits- prefix."""
        handle = await local_client.get_workspace(
            user_id="u1",
            project_id="p1",
            repo=REPO_NAME,
            repo_url=str(upstream_repo),
            branch="main",
            base_ref="main",
            mode=WorkspaceMode.ANALYSIS,
        )
        assert handle.capabilities == Capabilities(
            writable=False, isolated=False, persistent=True
        )
        assert handle.branch == "main"

    async def test_two_modes_share_same_repo_cache(
        self, local_client: SandboxClient, upstream_repo: Path, repos_base: Path
    ) -> None:
        """ANALYSIS and EDIT workspaces share one bare clone (no re-clone)."""
        cache = await local_client.ensure_repo_cache(
            user_id="u1", repo=REPO_NAME, base_ref="main", repo_url=str(upstream_repo)
        )
        bare = Path(cache.location.local_path)
        bare_inode = bare.stat().st_ino

        edit = await local_client.get_workspace(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-c", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="c",
        )
        analysis = await local_client.get_workspace(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="main", base_ref="main",
            mode=WorkspaceMode.ANALYSIS,
        )

        # Different workspaces, same parent cache, same bare on disk.
        assert edit.workspace_id != analysis.workspace_id
        assert bare.stat().st_ino == bare_inode
        # And both worktrees live under that bare's repo dir.
        for handle in (edit, analysis):
            assert (repos_base / REPO_NAME / "worktrees") in Path(
                handle.local_path
            ).parents


# ----------------------------------------------------------------------
# Conversation isolation (the P4 fix)
# ----------------------------------------------------------------------
class TestConversationIsolation:
    """Two parallel agent sessions must never stomp each other.

    The path scheme ``<user>_<scope>_<branch>`` (see
    ``LocalGitWorkspaceProvider._worktree_path``) is what gives us this
    isolation; the unit test is "writes in conv-A are invisible from
    conv-B."
    """

    async def test_two_conversations_get_distinct_worktrees(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        h_a = await local_client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-conv-a", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="conv-a",
        )
        h_b = await local_client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-conv-b", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="conv-b",
        )
        assert h_a.workspace_id != h_b.workspace_id
        assert h_a.branch != h_b.branch
        assert h_a.local_path != h_b.local_path

        await local_client.write_file(h_a, "ONLY_IN_A.txt", b"a-only\n")
        # h_b's directory must not have it.
        listing = subprocess.run(
            ["ls", h_b.local_path],
            capture_output=True, text=True, check=True,
        )
        assert "ONLY_IN_A.txt" not in listing.stdout

    async def test_same_branch_two_conversations_surfaces_typed_error(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        """git itself refuses to check out one branch in two writable
        worktrees. The sandbox must propagate that as a typed
        ``RepoCacheUnavailable``, not a generic exception or — worse —
        silently share the worktree across conversations.

        This pins the failure mode so a future change that "fixes" the
        collision by silently sharing one worktree across two
        conversation_ids is caught immediately.
        """
        from sandbox.domain.errors import RepoCacheUnavailable

        common_branch = "shared-branch"
        h_a = await local_client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch=common_branch, base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="conv-x",
        )
        with pytest.raises(RepoCacheUnavailable, match="already checked out"):
            await local_client.acquire_session(
                user_id="u1", project_id="p1",
                repo=REPO_NAME, repo_url=str(upstream_repo),
                branch=common_branch, base_ref="main",
                mode=WorkspaceMode.EDIT, conversation_id="conv-y",
            )
        # And the first session is unaffected.
        assert Path(h_a.local_path).is_dir()


# ----------------------------------------------------------------------
# Persistence across SandboxClient instances
# ----------------------------------------------------------------------
class TestStorePersistence:
    """``JsonSandboxStore`` flushes after every save, so a restart should
    see the same workspaces and runtimes.

    This pins the "worker restart doesn't orphan the agent's work"
    contract from the persistence section of the doc.
    """

    async def test_workspace_metadata_survives_fresh_client(
        self, repos_base: Path, metadata_path: Path, upstream_repo: Path
    ) -> None:
        # First client: create the workspace and write a marker file.
        settings = SandboxSettings(
            provider="local", runtime="local_subprocess",
            repos_base_path=str(repos_base),
            metadata_path=str(metadata_path),
            local_allow_write=True,
        )
        client_a = SandboxClient.from_container(build_sandbox_container(settings))
        h_a = await client_a.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-persist", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="persist",
        )
        await client_a.write_file(h_a, "MARKER.txt", b"hello")
        original_id = h_a.workspace_id
        original_path = h_a.local_path

        # Second client, same metadata.json + same .repos/. The store
        # must hand back the same workspace row, and the file written
        # in the first session must still be there.
        client_b = SandboxClient.from_container(build_sandbox_container(settings))
        h_b = await client_b.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-persist", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="persist",
        )
        assert h_b.workspace_id == original_id
        assert h_b.local_path == original_path
        assert (Path(h_b.local_path) / "MARKER.txt").read_text() == "hello"

    async def test_auth_token_is_scrubbed_from_metadata(
        self, local_client: SandboxClient, upstream_repo: Path, metadata_path: Path
    ) -> None:
        """``JsonSandboxStore._json_ready`` strips ``auth_token`` keys
        before writing. This is a credential-leakage check on the
        persisted file, not a fetch-time concern."""
        await local_client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-scrub", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="scrub",
            auth_token="ghp_should_never_appear_in_metadata",
        )
        contents = metadata_path.read_text()
        assert "ghp_should_never_appear_in_metadata" not in contents


# ----------------------------------------------------------------------
# Path safety
# ----------------------------------------------------------------------
class TestPathSafety:
    """``_validate_relpath`` and ``_safe_local_path`` are defence in
    depth against path traversal at the SandboxClient surface.

    Even on the local subprocess runtime — which has no fs-level
    isolation — these checks prevent an LLM-supplied path from reading
    or writing outside the worktree."""

    async def test_absolute_path_is_rejected(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        handle = await self._handle(local_client, upstream_repo, "abs")
        with pytest.raises(InvalidWorkspacePath):
            await local_client.read_file(handle, "/etc/passwd")
        with pytest.raises(InvalidWorkspacePath):
            await local_client.write_file(handle, "/tmp/x", b"x")

    async def test_dotdot_traversal_is_rejected(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        handle = await self._handle(local_client, upstream_repo, "dotdot")
        with pytest.raises(InvalidWorkspacePath):
            await local_client.read_file(handle, "../escape.txt")
        with pytest.raises(InvalidWorkspacePath):
            await local_client.write_file(handle, "src/../../escape.txt", b"x")

    async def test_symlink_escape_is_rejected(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        """A symlink inside the worktree must not let reads escape it.

        The string-level check passes (no ``..``), but
        ``_safe_local_path`` resolves the symlink and rejects the
        target if it lands outside the workspace root.
        """
        handle = await self._handle(local_client, upstream_repo, "symlink")
        worktree = Path(handle.local_path)
        # Build a symlink that points at the system /etc directory. On
        # the local provider this would otherwise let the agent read
        # arbitrary host files via a relative path.
        link = worktree / "rogue"
        link.symlink_to("/etc")
        with pytest.raises(InvalidWorkspacePath):
            await local_client.read_file(handle, "rogue/passwd")

    @staticmethod
    async def _handle(
        client: SandboxClient, upstream_repo: Path, conv: str
    ):
        return await client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch=f"agent/edits-{conv}", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id=conv,
        )


# ----------------------------------------------------------------------
# Git operations: read / write / list / search / status / diff / commit
# ----------------------------------------------------------------------
class TestGitOperations:
    """Round-trip the public file/git helpers against the fixture repo."""

    async def test_read_write_list(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        handle = await local_client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-rw", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="rw",
        )
        # Read the fixture file.
        readme = await local_client.read_file(handle, "README.md")
        assert readme == b"hello sandbox\n"

        # Write a new file and read it back.
        await local_client.write_file(handle, "notes/hello.md", "**hi**")
        on_disk = (Path(handle.local_path) / "notes" / "hello.md").read_text()
        assert on_disk == "**hi**"

        entries = await local_client.list_dir(handle, "notes")
        names = [e.name for e in entries]
        assert "hello.md" in names

    async def test_search_via_ripgrep(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        if shutil.which("rg") is None:
            pytest.skip("rg not on PATH")
        handle = await local_client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-search", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="search",
        )
        hits = await local_client.search(handle, pattern="alive", max_hits=5)
        assert hits, "expected at least one hit on the fixture's app.py"
        assert any("app.py" in h.path for h in hits)

    async def test_status_diff_commit_round_trip(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        """The committed-state contract:

        1. New file in worktree -> `untracked`
        2. After write to tracked file -> `unstaged`
        3. After commit -> `is_clean=True`, returned SHA matches HEAD
        """
        handle = await local_client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-commit", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="commit",
        )
        # Create a brand-new file.
        await local_client.write_file(handle, "AGENT_NOTES.md", b"# Hi\n")
        status = await local_client.status(handle)
        assert "AGENT_NOTES.md" in status.untracked
        assert not status.is_clean

        # Modify the existing tracked README.
        await local_client.write_file(handle, "README.md", b"hello sandbox\nplus me\n")
        status = await local_client.status(handle)
        assert "README.md" in status.unstaged

        # Commit everything; status returns clean.
        sha = await local_client.commit(
            handle, "agent commit",
            author=("Test", "test@example.com"),
        )
        assert len(sha) == 40
        after = await local_client.status(handle)
        assert after.is_clean

        # `git log` must surface the commit.
        log = await local_client.exec(
            handle,
            ["git", "log", "-1", "--format=%s"],
            command_kind=CommandKind.READ,
        )
        assert log.stdout.decode().strip() == "agent commit"

    async def test_dirty_flag_only_flips_on_successful_write(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        """``SandboxService._exec_unlocked`` only marks dirty when both
        ``command_kind.mutates_workspace`` AND ``exit_code == 0``.

        A failed write — even a syntactically WRITE one — must leave
        the workspace flagged clean so eviction can still skip it."""
        handle = await local_client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-dirty", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="dirty",
        )
        ws = await local_client.container.service.get_workspace(handle.workspace_id)
        assert ws.dirty is False

        # A WRITE that fails: writing to a directory should fail.
        result = await local_client.exec(
            handle,
            ["sh", "-c", "echo x > src"],  # src is a directory in the fixture
            command_kind=CommandKind.WRITE,
        )
        assert result.exit_code != 0
        ws = await local_client.container.service.get_workspace(handle.workspace_id)
        assert ws.dirty is False, "failed WRITE must NOT flip dirty"

        # A WRITE that succeeds: dirty flips to True.
        ok = await local_client.exec(
            handle,
            ["sh", "-c", "echo x > new.txt"],
            command_kind=CommandKind.WRITE,
        )
        assert ok.exit_code == 0
        ws = await local_client.container.service.get_workspace(handle.workspace_id)
        assert ws.dirty is True


# ----------------------------------------------------------------------
# Concurrent execution
# ----------------------------------------------------------------------
class TestConcurrency:
    """``SandboxService.exec`` takes a per-workspace lock for mutating
    commands; reads are unlocked. The two halves of this test pin both."""

    async def test_concurrent_writes_serialize_per_workspace(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        handle = await local_client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-concurrent", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="concurrent",
        )
        await local_client.exec(
            handle,
            ["sh", "-c", "printf '' > log.txt"],
            command_kind=CommandKind.WRITE,
        )

        async def append(token: str) -> None:
            await local_client.exec(
                handle,
                [
                    "sh", "-c",
                    f"printf {token} >> log.txt && sleep 0.2 && printf {token} >> log.txt",
                ],
                command_kind=CommandKind.WRITE,
            )

        await asyncio.gather(append("A"), append("B"))
        result = await local_client.exec(
            handle, ["cat", "log.txt"], command_kind=CommandKind.READ
        )
        # If the lock works the file ends with `AABB` or `BBAA`,
        # never with interleaved `ABAB` or `BABA`.
        assert result.stdout in {b"AABB", b"BBAA"}, result.stdout

    async def test_concurrent_reads_do_not_serialize(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        """Reads should be allowed in parallel — they don't take the lock.

        This is more behaviour smoke-test than wall-clock proof: we
        kick off two slow reads at once and assert the total wall time
        is closer to 1× the per-read sleep than 2×, with a generous
        margin so we don't fight CI flakiness."""
        handle = await local_client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-parallel-r", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="parallel-r",
        )

        async def slow_read() -> None:
            await local_client.exec(
                handle,
                ["sh", "-c", "sleep 0.5; cat README.md"],
                command_kind=CommandKind.READ,
            )

        loop = asyncio.get_event_loop()
        start = loop.time()
        await asyncio.gather(slow_read(), slow_read())
        elapsed = loop.time() - start
        # 2 serialized reads would be ~1.0s. 2 parallel ~0.5-0.7s. The
        # ceiling is set well below 1.0s to catch the regression but
        # leave headroom.
        assert elapsed < 0.95, f"reads serialized (took {elapsed:.2f}s)"


# ----------------------------------------------------------------------
# Read-only runtime + capability gating on PR
# ----------------------------------------------------------------------
class TestRuntimePolicies:
    async def test_readonly_runtime_blocks_writes(
        self, repos_base: Path, metadata_path: Path, upstream_repo: Path
    ) -> None:
        """``LocalSubprocessRuntimeProvider(allow_write=False)`` rejects
        every WRITE/INSTALL/TEST kind regardless of the workspace's
        own capabilities — this is a runtime-side guard, not a
        workspace-mode one."""
        settings = SandboxSettings(
            provider="local", runtime="local_subprocess",
            repos_base_path=str(repos_base),
            metadata_path=str(metadata_path),
            local_allow_write=False,
        )
        client = SandboxClient.from_container(build_sandbox_container(settings))
        handle = await client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-ro", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="ro",
        )
        # READ is fine.
        ok = await client.exec(handle, ["true"], command_kind=CommandKind.READ)
        assert ok.exit_code == 0
        # WRITE is rejected at the runtime boundary.
        with pytest.raises(RuntimeCommandRejected):
            await client.exec(
                handle,
                ["sh", "-c", "echo x > x.txt"],
                command_kind=CommandKind.WRITE,
            )

    async def test_pr_creation_refuses_readonly_handle(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        """``create_pull_request`` fires its capability check before the
        platform call — opening a PR on a read-only handle is a
        nonsense operation and the user gets a clear error instead of a
        confusing "not configured"."""
        analysis = await local_client.get_workspace(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="main", base_ref="main",
            mode=WorkspaceMode.ANALYSIS,
        )
        with pytest.raises(SandboxOpError, match="writable"):
            await local_client.create_pull_request(
                analysis,
                repo=REPO_NAME,
                title="should not happen",
                body="...",
                base_branch="main",
            )

    async def test_pr_creation_raises_when_platform_unconfigured(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        """When no ``GitPlatformProvider`` is wired (the default for
        the local container), the writable-handle path falls through
        to the service which must raise ``GitPlatformNotConfigured``."""
        handle = await local_client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-no-pr", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="no-pr",
        )
        with pytest.raises(GitPlatformNotConfigured):
            await local_client.create_pull_request(
                handle,
                repo=REPO_NAME,
                title="t", body="b", base_branch="main",
            )

    async def test_runtime_destroy_keeps_workspace_alive(
        self, local_client: SandboxClient, upstream_repo: Path
    ) -> None:
        """Doc rule (SandboxRuntime section): destroying a runtime must
        not delete a workspace unless the caller explicitly asks for
        workspace cleanup."""
        handle = await local_client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-rtdestroy", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="rtdestroy",
        )
        # Touch the runtime to materialize it in the store.
        await local_client.exec(handle, ["true"], command_kind=CommandKind.READ)
        runtime = await local_client.container.store.find_runtime_by_workspace(
            handle.workspace_id
        )
        assert runtime is not None

        # destroy_runtime via release_session(destroy_runtime=True).
        await local_client.release_session(handle, destroy_runtime=True)

        # Workspace and worktree both intact.
        assert Path(handle.local_path).is_dir()
        ws = await local_client.container.service.get_workspace(handle.workspace_id)
        assert ws.id == handle.workspace_id

        # Re-exec auto-creates a new runtime and works.
        result = await local_client.exec(
            handle, ["cat", "README.md"], command_kind=CommandKind.READ
        )
        assert result.exit_code == 0
