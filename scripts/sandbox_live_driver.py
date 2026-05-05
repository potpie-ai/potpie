"""Live, end-to-end driver for the consolidated sandbox tool surface.

Exercises the same code path the agent harness uses, against a real
clone of ``https://github.com/potpie-ai/potpie-ui``:

  1. Build a SandboxClient pointed at a tmp .repos/ root.
  2. Provision a repo cache (the parsing-side step).
  3. Acquire an EDIT workspace on a per-conversation branch.
  4. Wire up ``create_sandbox_tools(client=..., handle=...)`` — the
     explicit form an agent would receive.
  5. Drive the four tools as if a model called them:
       sandbox_text_editor view → str_replace → view
       sandbox_shell ls / git log
       sandbox_search "potpie"
       sandbox_git status / diff / commit
  6. Release + re-acquire to confirm changes survive the lap.
  7. Tear down.

Run with:

    python scripts/sandbox_live_driver.py

Skips cleanly if git isn't on PATH or github.com is unreachable.
"""
from __future__ import annotations

import asyncio
import shutil
import socket
import sys
import tempfile
from pathlib import Path

# Ensure the in-tree sandbox package wins over any installed copy.
ROOT = Path(__file__).resolve().parents[1]
SANDBOX_SRC = ROOT / "app" / "src" / "sandbox"
if str(SANDBOX_SRC) not in sys.path:
    sys.path.insert(0, str(SANDBOX_SRC))

from sandbox import (  # noqa: E402
    SandboxClient,
    WorkspaceMode,
    build_sandbox_container,
)
from sandbox.bootstrap.settings import SandboxSettings  # noqa: E402

# Import after sandbox is on the path; the in-tree app module wraps it.
sys.path.insert(0, str(ROOT))
from app.modules.intelligence.tools.sandbox.tools import create_sandbox_tools  # noqa: E402

REPO_NAME = "potpie-ai/potpie-ui"
REPO_URL = "https://github.com/potpie-ai/potpie-ui.git"
BASE_BRANCH = "main"
USER = "live-driver"
PROJECT = "live-driver-project"
CONV = "drv-conv-1"


def _online() -> bool:
    try:
        socket.create_connection(("github.com", 443), timeout=5).close()
        return True
    except OSError:
        return False


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


async def _drive(tmp_root: Path) -> None:
    settings = SandboxSettings(
        provider="local",
        runtime="local_subprocess",
        repos_base_path=str(tmp_root / ".repos"),
        metadata_path=str(tmp_root / "metadata.json"),
        local_allow_write=True,
    )
    container = build_sandbox_container(settings)
    client = SandboxClient.from_container(container)

    _section("ensure_repo_cache (parsing READY hook)")
    cache = await client.ensure_repo_cache(
        user_id=USER,
        repo=REPO_NAME,
        base_ref=BASE_BRANCH,
        repo_url=REPO_URL,
    )
    print(f"  cache id  = {cache.id}")
    print(f"  bare path = {cache.location.local_path}")

    _section("acquire_session (EDIT workspace on per-conv branch)")
    work_branch = f"agent/edits-{CONV}"
    handle = await client.acquire_session(
        user_id=USER,
        project_id=PROJECT,
        repo=REPO_NAME,
        repo_url=REPO_URL,
        branch=work_branch,
        base_ref=BASE_BRANCH,
        create_branch=True,
        mode=WorkspaceMode.EDIT,
        conversation_id=CONV,
    )
    print(f"  workspace_id = {handle.workspace_id}")
    print(f"  branch       = {handle.branch}")
    print(f"  local_path   = {handle.local_path}")
    print(f"  capabilities = {handle.capabilities}")

    _section("create_sandbox_tools (explicit form, capability-gated)")
    tools = create_sandbox_tools(
        client=client,
        handle=handle,
        pr_repo_name=REPO_NAME,
        pr_repo_url=REPO_URL,
    )
    by_name = {t.name: t for t in tools}
    print(f"  exposed: {sorted(by_name)}")

    _section("sandbox_text_editor view README.md")
    res = await by_name["sandbox_text_editor"].func(
        command="view", path="README.md", view_range=[1, 8]
    )
    if res.get("success"):
        print(f"  total_lines = {res['total_lines']}")
        first = res["content"].splitlines()[:3]
        print(f"  preview     = {first!r}")
    else:
        print(f"  FAIL: {res}")

    _section("sandbox_text_editor create AGENT_NOTES.md")
    body = "# Agent notes\n\nWritten by the live driver.\n"
    res = await by_name["sandbox_text_editor"].func(
        command="create", path="AGENT_NOTES.md", file_text=body
    )
    print(f"  success={res.get('success')} bytes={res.get('bytes')}")

    _section("sandbox_text_editor str_replace AGENT_NOTES.md")
    res = await by_name["sandbox_text_editor"].func(
        command="str_replace",
        path="AGENT_NOTES.md",
        old_str="live driver",
        new_str="LIVE driver run",
    )
    print(f"  success={res.get('success')}")

    _section("sandbox_text_editor view AGENT_NOTES.md (round-trip)")
    res = await by_name["sandbox_text_editor"].func(
        command="view", path="AGENT_NOTES.md"
    )
    print(f"  content = {res.get('content')!r}")

    _section("sandbox_search ripgrep 'potpie'")
    res = await by_name["sandbox_search"].func(pattern="potpie", max_hits=3)
    print(f"  success={res.get('success')} hits={len(res.get('hits', []))}")
    for hit in res.get("hits", [])[:3]:
        print(f"    {hit['path']}:{hit['line']}  {hit['snippet'][:80]!r}")

    _section("sandbox_shell echo + ls")
    res = await by_name["sandbox_shell"].func(command="ls -1 | head -5")
    print(
        f"  exit={res.get('exit_code')} "
        f"stdout={res.get('stdout','').strip().splitlines()[:5]}"
    )

    _section("sandbox_git status (after edits)")
    res = await by_name["sandbox_git"].func(command="status")
    print(
        "  branch={branch} clean={is_clean} "
        "staged={staged} unstaged={unstaged} untracked={untracked}".format(
            **{
                k: res.get(k)
                for k in ("branch", "is_clean", "staged", "unstaged", "untracked")
            }
        )
    )

    _section("sandbox_git commit")
    res = await by_name["sandbox_git"].func(
        command="commit",
        message="Add agent notes (live driver)",
    )
    print(f"  success={res.get('success')} sha={res.get('commit')}")

    _section("sandbox_git log")
    res = await by_name["sandbox_git"].func(command="log", limit=3)
    for entry in res.get("commits", [])[:3]:
        print(f"  {entry.get('sha', '')[:8]}  {entry.get('subject', '')}")

    _section("release_session + re-acquire (changes must persist)")
    await client.release_session(handle)
    handle2 = await client.acquire_session(
        user_id=USER,
        project_id=PROJECT,
        repo=REPO_NAME,
        repo_url=REPO_URL,
        branch=work_branch,
        base_ref=BASE_BRANCH,
        create_branch=True,
        mode=WorkspaceMode.EDIT,
        conversation_id=CONV,
    )
    same = handle2.workspace_id == handle.workspace_id
    persisted = (Path(handle2.local_path) / "AGENT_NOTES.md").read_text()
    print(f"  same workspace = {same}")
    print(f"  AGENT_NOTES still present, head = {persisted.splitlines()[0]!r}")

    _section("destroy_workspace")
    await client.destroy_workspace(handle2)
    gone = not Path(handle.local_path).exists()
    print(f"  worktree removed = {gone}")
    print(f"  bare cache kept  = {Path(cache.location.local_path).is_dir()}")


async def main() -> int:
    if shutil.which("git") is None:
        print("git not on PATH; skipping live driver")
        return 0
    if not _online():
        print("github.com unreachable; skipping live driver")
        return 0
    with tempfile.TemporaryDirectory(prefix="sandbox-driver-") as tmp:
        await _drive(Path(tmp))
    print("\nLIVE DRIVER OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
