"""Live end-to-end test of the sandbox core against a real GitHub repo.

This is the canonical "does it actually work?" test. It clones
``https://github.com/potpie-ai/potpie-ui`` (a small public Next.js repo)
into a real `.repos/` cache, opens an EDIT workspace on a fresh branch,
and walks the full agent flow:

    ensure_repo_cache → get_workspace → read_file → write_file → exec
    → status → diff → commit → release_session → re-acquire → verify
    → destroy_workspace

It also runs an ANALYSIS workspace against the same cache to confirm
the read-only capability gating and the cache-sharing claim from the
roadmap.

The test is opt-in (skipped by default) because it makes outbound
network calls. Set ``RUN_LIVE_SANDBOX_TESTS=1`` to enable. CI / dev
runs it on demand; the rest of the suite stays hermetic.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import AsyncIterator

import pytest

from sandbox import (
    Capabilities,
    CommandKind,
    SandboxClient,
    SandboxOpError,
    WorkspaceMode,
    build_sandbox_container,
)
from sandbox.bootstrap.settings import SandboxSettings


REPO_NAME = "potpie-ai/potpie-ui"
REPO_URL = "https://github.com/potpie-ai/potpie-ui.git"
BASE_BRANCH = "main"


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_SANDBOX_TESTS") != "1",
    reason="Set RUN_LIVE_SANDBOX_TESTS=1 to run live network-bound sandbox tests",
)


def _git_available() -> bool:
    return shutil.which("git") is not None


def _internet_reachable(host: str = "github.com") -> bool:
    """Cheap pre-flight to skip cleanly when offline."""
    import socket

    try:
        socket.create_connection((host, 443), timeout=5).close()
        return True
    except OSError:
        return False


@pytest.fixture
def sandbox_root(tmp_path: Path) -> Path:
    return tmp_path / ".repos"


@pytest.fixture
def metadata_path(tmp_path: Path) -> Path:
    return tmp_path / "metadata.json"


@pytest.fixture
async def client(
    sandbox_root: Path, metadata_path: Path
) -> AsyncIterator[SandboxClient]:
    """Build a real `SandboxClient` rooted at a tmp `.repos/` dir.

    Forces the canonical local provider regardless of env so the test
    actually exercises the new code path.
    """
    if not _git_available():
        pytest.skip("git not on PATH")
    if not _internet_reachable():
        pytest.skip("github.com unreachable from this host")
    settings = SandboxSettings(
        provider="local",
        runtime="local_subprocess",
        repos_base_path=str(sandbox_root),
        metadata_path=str(metadata_path),
        local_allow_write=True,
    )
    container = build_sandbox_container(settings)
    yield SandboxClient.from_container(container)


# ----------------------------------------------------------------------
# Live flow
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_full_agent_flow_against_potpie_ui(
    client: SandboxClient, sandbox_root: Path
) -> None:
    """End-to-end: cache → workspace → edit → commit → re-acquire → verify."""
    user_id = "live-test-user"
    project_id = "live-test-project"
    conversation_id = "live-conv-1"

    # ---- 1. Provision the repo cache (parsing's READY hook) ----------
    cache = await client.ensure_repo_cache(
        user_id=user_id,
        repo=REPO_NAME,
        base_ref=BASE_BRANCH,
        repo_url=REPO_URL,
    )
    assert cache.backend_kind == "local"
    assert cache.location.local_path is not None
    bare = Path(cache.location.local_path)
    assert bare.is_dir()
    assert (bare / "HEAD").exists(), "bare clone should have a HEAD file"

    # Calling again is idempotent — same id, same on-disk path.
    cache_again = await client.ensure_repo_cache(
        user_id=user_id,
        repo=REPO_NAME,
        base_ref=BASE_BRANCH,
        repo_url=REPO_URL,
    )
    assert cache_again.id == cache.id
    assert cache_again.location.local_path == cache.location.local_path

    # ---- 2. Acquire an EDIT workspace on a new conversation branch ---
    # Caller is responsible for picking a unique workspace branch (the
    # API contract — auto-derive happens only inside the adapter when
    # ``branch_name`` is left unset, which the client doesn't currently
    # expose). Mirror the convention used elsewhere in the harness.
    work_branch = f"agent/edits-{conversation_id}"
    handle = await client.acquire_session(
        user_id=user_id,
        project_id=project_id,
        repo=REPO_NAME,
        repo_url=REPO_URL,
        branch=work_branch,
        base_ref=BASE_BRANCH,
        create_branch=True,
        mode=WorkspaceMode.EDIT,
        conversation_id=conversation_id,
    )
    assert handle.local_path is not None
    worktree = Path(handle.local_path)
    assert worktree.is_dir()
    assert handle.capabilities.writable is True
    assert handle.capabilities.isolated is True
    assert handle.branch == work_branch, handle.branch

    # README.md is the canonical file to read first — every agent run
    # in the harness reads it to orient itself.
    readme = await client.read_file(handle, "README.md")
    assert b"potpie" in readme.lower(), "README should mention potpie"

    # ---- 3. Search via ripgrep over the worktree ---------------------
    # rg is preinstalled in the agent-sandbox image but not always on
    # the host; gate this leg of the test on rg availability.
    if shutil.which("rg") is not None:
        hits = await client.search(handle, pattern="potpie", max_hits=5)
        assert hits, "expected at least one ripgrep hit for 'potpie'"

    # ---- 4. Write a new file inside the worktree ---------------------
    new_file_rel = "AGENT_NOTES.md"
    body = "# Agent notes\n\nWritten by live integration test.\n"
    await client.write_file(handle, new_file_rel, body)
    on_disk = (worktree / new_file_rel).read_text()
    assert on_disk == body, "write_file must land bytes on the host fs"

    # ---- 5. Run a shell command (read-kind) --------------------------
    ls = await client.exec(handle, ["ls", "-1"], command_kind=CommandKind.READ)
    assert ls.exit_code == 0
    listing = ls.stdout.decode()
    assert "AGENT_NOTES.md" in listing, listing
    # Make sure we're on the right branch from the runtime's perspective.
    branch_check = await client.exec(
        handle, ["git", "rev-parse", "--abbrev-ref", "HEAD"], command_kind=CommandKind.READ
    )
    assert branch_check.exit_code == 0
    assert branch_check.stdout.decode().strip() == handle.branch

    # ---- 6. Mutating exec — append to the new file ------------------
    append = await client.exec(
        handle,
        ["sh", "-c", "printf '\\nappended\\n' >> AGENT_NOTES.md"],
        command_kind=CommandKind.WRITE,
    )
    assert append.exit_code == 0
    assert (worktree / new_file_rel).read_text().endswith("appended\n")

    # ---- 7. Git status — should show new + dirty --------------------
    status = await client.status(handle)
    assert status.branch == handle.branch
    assert not status.is_clean
    # New file is `??` (untracked) — and there should be no staged or
    # unstaged tracked changes for this brand-new file.
    assert new_file_rel in status.untracked

    # ---- 8. Diff before commit (working tree, against HEAD) ---------
    # `git diff` without --cached only shows tracked changes; for a
    # purely new file we expect an empty diff. We just want to confirm
    # the call succeeds.
    diff_text = await client.diff(handle)
    assert isinstance(diff_text, str)

    # ---- 9. Commit and read SHA back --------------------------------
    sha = await client.commit(
        handle,
        "Add agent notes",
        author=("Sandbox Test", "sandbox-test@example.com"),
    )
    assert len(sha) == 40, sha
    after_status = await client.status(handle)
    assert after_status.is_clean

    # ---- 10. Release the session (hibernate runtime, keep worktree) -
    await client.release_session(handle)

    # ---- 11. Re-acquire the same workspace; expect the SAME path ----
    handle_again = await client.acquire_session(
        user_id=user_id,
        project_id=project_id,
        repo=REPO_NAME,
        repo_url=REPO_URL,
        branch=work_branch,
        base_ref=BASE_BRANCH,
        create_branch=True,
        mode=WorkspaceMode.EDIT,
        conversation_id=conversation_id,
    )
    assert handle_again.workspace_id == handle.workspace_id, (
        "second acquire_session for the same conversation must return the same id"
    )
    assert handle_again.local_path == handle.local_path
    persisted = (worktree / new_file_rel).read_text()
    assert persisted.startswith("# Agent notes"), (
        "files written in the previous turn must survive a release/re-acquire cycle"
    )
    log_call = await client.exec(
        handle_again,
        ["git", "log", "-1", "--format=%s"],
        command_kind=CommandKind.READ,
    )
    assert log_call.exit_code == 0
    assert log_call.stdout.decode().strip() == "Add agent notes"

    # ---- 12. Tear down the workspace; cache must survive -----------
    await client.destroy_workspace(handle_again)
    assert not worktree.exists(), "worktree should be removed by destroy_workspace"
    assert bare.is_dir(), "repo cache must outlive workspace destruction"
    # And ensure_repo_cache must continue to work (no re-clone needed —
    # the bare on disk is reused).
    cache_after = await client.ensure_repo_cache(
        user_id=user_id,
        repo=REPO_NAME,
        base_ref=BASE_BRANCH,
        repo_url=REPO_URL,
    )
    assert cache_after.location.local_path == str(bare.resolve())


@pytest.mark.asyncio
async def test_analysis_workspace_is_readonly_and_shares_cache(
    client: SandboxClient, sandbox_root: Path
) -> None:
    """ANALYSIS workspaces are read-only and reuse the same cache as EDIT."""
    user_id = "live-test-user-2"
    project_id = "live-test-project-2"

    # Warm the cache once.
    cache = await client.ensure_repo_cache(
        user_id=user_id,
        repo=REPO_NAME,
        base_ref=BASE_BRANCH,
        repo_url=REPO_URL,
    )
    assert cache.location.local_path is not None
    bare = Path(cache.location.local_path)
    assert bare.is_dir()

    analysis = await client.get_workspace(
        user_id=user_id,
        project_id=project_id,
        repo=REPO_NAME,
        repo_url=REPO_URL,
        branch=BASE_BRANCH,
        base_ref=BASE_BRANCH,
        mode=WorkspaceMode.ANALYSIS,
    )
    assert analysis.capabilities == Capabilities(
        writable=False, isolated=False, persistent=True
    )
    # ANALYSIS branches off base_ref, no agent/edits- prefix.
    assert analysis.branch == BASE_BRANCH

    # Reading is fine.
    readme = await client.read_file(analysis, "README.md")
    assert readme

    # The analysis worktree path lives under the same .repos/<repo>/worktrees
    # tree — i.e. it's sharing the bare repo with the EDIT workspace.
    assert analysis.local_path is not None
    a_path = Path(analysis.local_path).resolve()
    assert sandbox_root.resolve() in a_path.parents

    # No re-clone happened — the bare's mtime should not have moved
    # forward in any meaningful way; the easier assertion is that the
    # cache row is the same id we got from ensure_repo_cache above.
    cache_again = await client.ensure_repo_cache(
        user_id=user_id,
        repo=REPO_NAME,
        base_ref=BASE_BRANCH,
        repo_url=REPO_URL,
    )
    assert cache_again.id == cache.id


@pytest.mark.asyncio
async def test_pr_tool_refuses_readonly_handle(
    client: SandboxClient,
) -> None:
    """`create_pull_request` must refuse a workspace acquired in ANALYSIS mode.

    We don't actually open a PR (that would need GitHub auth and create a real
    PR); we just verify the capability check fires before any platform call.
    """
    user_id = "live-test-user-3"
    project_id = "live-test-project-3"
    await client.ensure_repo_cache(
        user_id=user_id,
        repo=REPO_NAME,
        base_ref=BASE_BRANCH,
        repo_url=REPO_URL,
    )
    analysis = await client.get_workspace(
        user_id=user_id,
        project_id=project_id,
        repo=REPO_NAME,
        repo_url=REPO_URL,
        branch=BASE_BRANCH,
        base_ref=BASE_BRANCH,
        mode=WorkspaceMode.ANALYSIS,
    )
    with pytest.raises(SandboxOpError, match="writable"):
        await client.create_pull_request(
            analysis,
            repo=REPO_NAME,
            title="should not happen",
            body="...",
            base_branch=BASE_BRANCH,
        )


@pytest.mark.asyncio
async def test_two_conversations_get_isolated_branches(
    client: SandboxClient,
) -> None:
    """Two conversations on the same repo get independent worktrees and branches.

    This is the load-bearing claim of EDIT mode: parallel agent sessions
    must not stomp each other.
    """
    user_id = "live-test-user-4"
    project_id = "live-test-project-4"
    await client.ensure_repo_cache(
        user_id=user_id,
        repo=REPO_NAME,
        base_ref=BASE_BRANCH,
        repo_url=REPO_URL,
    )
    h_a = await client.get_workspace(
        user_id=user_id,
        project_id=project_id,
        repo=REPO_NAME,
        repo_url=REPO_URL,
        branch="agent/edits-conv-a",
        base_ref=BASE_BRANCH,
        create_branch=True,
        mode=WorkspaceMode.EDIT,
        conversation_id="conv-a",
    )
    h_b = await client.get_workspace(
        user_id=user_id,
        project_id=project_id,
        repo=REPO_NAME,
        repo_url=REPO_URL,
        branch="agent/edits-conv-b",
        base_ref=BASE_BRANCH,
        create_branch=True,
        mode=WorkspaceMode.EDIT,
        conversation_id="conv-b",
    )
    assert h_a.workspace_id != h_b.workspace_id
    assert h_a.branch != h_b.branch
    assert h_a.local_path != h_b.local_path
    assert h_b.local_path is not None
    # Writes in one workspace must not leak into the other.
    await client.write_file(h_a, "ONLY_IN_A.txt", b"a-only\n")
    listing_b = subprocess.run(
        ["ls"], cwd=h_b.local_path, capture_output=True, text=True, check=False
    )
    assert "ONLY_IN_A.txt" not in listing_b.stdout
