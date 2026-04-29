"""Unit tests for the consolidated sandbox-backed agent tools.

The four tools — ``sandbox_text_editor``, ``sandbox_shell``,
``sandbox_search``, ``sandbox_git`` — dispatch through a stub
``SandboxClient`` so the wiring layer is tested without exercising real
git, the filesystem, or the network. ``ProjectService`` is patched too
since the real call hits the DB.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from app.modules.intelligence.tools.sandbox import (
    client as sandbox_client_mod,
    tool_functions as fn,
)
from app.modules.intelligence.tools.sandbox.context import set_run_context
from app.modules.intelligence.tools.sandbox.tools import create_sandbox_tools
from sandbox import ExecResult, FileEntry, GitStatus, Hit, WorkspaceHandle


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
@dataclass
class _Call:
    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    args: tuple[Any, ...] = field(default_factory=tuple)


class FakeSandboxClient:
    """In-memory stand-in for :class:`SandboxClient`."""

    def __init__(self) -> None:
        self.calls: list[_Call] = []
        self.handle = WorkspaceHandle(
            workspace_id="ws_test",
            branch="agent/edits-conv1",
            backend_kind="local",
            local_path="/tmp/fake/work",
            remote_path=None,
        )
        # Per-test canned responses
        self.read_file_result: bytes = b"hello world\n"
        self.list_dir_result: list[FileEntry] = [
            FileEntry(name="src", is_dir=True, size=None),
            FileEntry(name="README.md", is_dir=False, size=12),
        ]
        self.search_result: list[Hit] = [
            Hit(path="src/app.py", line=3, snippet="alive")
        ]
        self.exec_result = ExecResult(exit_code=0, stdout=b"out\n", stderr=b"")
        self.status_result = GitStatus(
            branch="agent/edits-conv1",
            is_clean=False,
            staged=(),
            unstaged=("README.md",),
            untracked=(),
        )
        self.diff_result = "@@ -1 +1 @@\n-hello\n+world\n"
        self.commit_sha = "a" * 40

    async def get_workspace(self, **kwargs: Any) -> WorkspaceHandle:
        self.calls.append(_Call("get_workspace", kwargs=kwargs))
        return self.handle

    async def read_file(
        self, handle: WorkspaceHandle, path: str, *, max_bytes: int | None = None
    ) -> bytes:
        self.calls.append(_Call("read_file", kwargs={"path": path}))
        return self.read_file_result

    async def write_file(
        self, handle: WorkspaceHandle, path: str, content: bytes | str
    ) -> None:
        self.calls.append(_Call("write_file", kwargs={"path": path, "content": content}))

    async def list_dir(
        self, handle: WorkspaceHandle, path: str = "."
    ) -> list[FileEntry]:
        self.calls.append(_Call("list_dir", kwargs={"path": path}))
        return self.list_dir_result

    async def search(
        self, handle: WorkspaceHandle, pattern: str, **kwargs: Any
    ) -> list[Hit]:
        self.calls.append(_Call("search", args=(pattern,), kwargs=kwargs))
        return self.search_result

    async def exec(
        self, handle: WorkspaceHandle, cmd: list[str], **kwargs: Any
    ) -> ExecResult:
        self.calls.append(_Call("exec", args=(tuple(cmd),), kwargs=kwargs))
        return self.exec_result

    async def status(self, handle: WorkspaceHandle) -> GitStatus:
        self.calls.append(_Call("status"))
        return self.status_result

    async def diff(
        self,
        handle: WorkspaceHandle,
        *,
        base_ref: str | None = None,
        paths: list[str] | None = None,
    ) -> str:
        self.calls.append(_Call("diff", kwargs={"base_ref": base_ref, "paths": paths}))
        return self.diff_result

    async def commit(
        self,
        handle: WorkspaceHandle,
        message: str,
        *,
        paths: list[str] | None = None,
        author: tuple[str, str] | None = None,
    ) -> str:
        self.calls.append(
            _Call("commit", args=(message,), kwargs={"paths": paths, "author": author})
        )
        return self.commit_sha

    async def push(
        self,
        handle: WorkspaceHandle,
        *,
        remote: str = "origin",
        set_upstream: bool = True,
        force: bool = False,
    ) -> None:
        self.calls.append(
            _Call("push", kwargs={"remote": remote, "set_upstream": set_upstream, "force": force})
        )


_PROJECT_DETAILS = {
    "project_name": "owner/repo",
    "branch_name": "main",
    "user_id": "u1",
    "repo_path": "https://github.com/owner/repo.git",
}


@pytest.fixture
def fake_client() -> Iterator[FakeSandboxClient]:
    client = FakeSandboxClient()
    sandbox_client_mod.set_sandbox_client(client)  # type: ignore[arg-type]
    set_run_context(user_id="u1", conversation_id="conv1", branch="main")
    # Force the text_editor _is_dir probe to consider local_path as a real
    # filesystem location; we patch it to always return False so the tests
    # land on the file-read path unless a test overrides it.
    with patch.object(fn, "_is_dir", side_effect=_fake_is_dir):
        yield client
    sandbox_client_mod.set_sandbox_client(None)


# Default: README.md is a file, src is a directory. Tests can monkey this.
_DIRS = {"src", "lib"}


async def _fake_is_dir(client: Any, handle: Any, path: str) -> bool:
    return path in _DIRS or path == "."


@pytest.fixture(autouse=True)
def stub_project_lookup() -> Iterator[None]:
    with patch.object(fn, "_project_details", return_value=_PROJECT_DETAILS):
        yield


# ----------------------------------------------------------------------
# Catalog wiring
# ----------------------------------------------------------------------
def test_create_sandbox_tools_registers_four_consolidated_tools() -> None:
    tools = create_sandbox_tools()
    names = {t.name for t in tools}
    assert names == {
        "sandbox_text_editor",
        "sandbox_shell",
        "sandbox_search",
        "sandbox_git",
    }


# ----------------------------------------------------------------------
# sandbox_text_editor — view / create / str_replace / insert
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_view_file_returns_content(fake_client: FakeSandboxClient) -> None:
    fake_client.read_file_result = b"line1\nline2\nline3\n"
    out = await fn.sandbox_text_editor_tool(
        project_id="p1", command="view", path="README.md"
    )
    assert out["success"] is True
    assert out["kind"] == "file"
    assert out["content"] == "line1\nline2\nline3\n"
    assert out["total_lines"] == 3
    assert out["bytes"] == len(b"line1\nline2\nline3\n")


@pytest.mark.asyncio
async def test_view_file_with_view_range(fake_client: FakeSandboxClient) -> None:
    fake_client.read_file_result = b"a\nb\nc\nd\ne\n"
    out = await fn.sandbox_text_editor_tool(
        project_id="p1", command="view", path="README.md", view_range=[2, 4]
    )
    assert out["content"] == "b\nc\nd\n"
    assert out["view_range"] == [2, 4]
    assert out["total_lines"] == 5


@pytest.mark.asyncio
async def test_view_directory(fake_client: FakeSandboxClient) -> None:
    out = await fn.sandbox_text_editor_tool(
        project_id="p1", command="view", path="src"
    )
    assert out["success"] is True
    assert out["kind"] == "directory"
    names = {e["name"] for e in out["entries"]}
    assert "README.md" in names
    assert "src" in names


@pytest.mark.asyncio
async def test_view_invalid_range_rejected(
    fake_client: FakeSandboxClient,
) -> None:
    out = await fn.sandbox_text_editor_tool(
        project_id="p1", command="view", path="README.md", view_range=[5, 2]
    )
    assert out["success"] is False
    assert "view_range" in out["error"]


@pytest.mark.asyncio
async def test_create_writes_new_file(fake_client: FakeSandboxClient) -> None:
    # Patch the existence probe so the file is treated as new.
    with patch("pathlib.Path.exists", return_value=False):
        out = await fn.sandbox_text_editor_tool(
            project_id="p1",
            command="create",
            path="src/new.py",
            file_text="print('hi')\n",
        )
    assert out["success"] is True
    assert out["lines"] == 1
    write_calls = [c for c in fake_client.calls if c.method == "write_file"]
    assert write_calls and write_calls[0].kwargs["path"] == "src/new.py"


@pytest.mark.asyncio
async def test_create_fails_when_file_exists(
    fake_client: FakeSandboxClient,
) -> None:
    with patch("pathlib.Path.exists", return_value=True):
        out = await fn.sandbox_text_editor_tool(
            project_id="p1",
            command="create",
            path="src/exists.py",
            file_text="x",
        )
    assert out["success"] is False
    assert "already exists" in out["error"]


@pytest.mark.asyncio
async def test_str_replace_unique_match(
    fake_client: FakeSandboxClient,
) -> None:
    fake_client.read_file_result = b"line a\nline b\nline c\n"
    out = await fn.sandbox_text_editor_tool(
        project_id="p1",
        command="str_replace",
        path="f.py",
        old_str="line b",
        new_str="line B!",
    )
    assert out["success"] is True
    assert out["lines_added"] == 1
    assert out["lines_removed"] == 1
    write = [c for c in fake_client.calls if c.method == "write_file"][0]
    assert write.kwargs["content"] == "line a\nline B!\nline c\n"


@pytest.mark.asyncio
async def test_str_replace_zero_matches(
    fake_client: FakeSandboxClient,
) -> None:
    fake_client.read_file_result = b"hello\nworld\n"
    out = await fn.sandbox_text_editor_tool(
        project_id="p1",
        command="str_replace",
        path="f.py",
        old_str="missing",
        new_str="x",
    )
    assert out["success"] is False
    assert "not found" in out["error"]


@pytest.mark.asyncio
async def test_str_replace_multiple_matches(
    fake_client: FakeSandboxClient,
) -> None:
    fake_client.read_file_result = b"abc abc abc"
    out = await fn.sandbox_text_editor_tool(
        project_id="p1",
        command="str_replace",
        path="f.py",
        old_str="abc",
        new_str="x",
    )
    assert out["success"] is False
    assert out["occurrences"] == 3


@pytest.mark.asyncio
async def test_insert_after_line(fake_client: FakeSandboxClient) -> None:
    fake_client.read_file_result = b"a\nb\nc\n"
    out = await fn.sandbox_text_editor_tool(
        project_id="p1",
        command="insert",
        path="f.py",
        insert_line=1,
        new_str="NEW",
    )
    assert out["success"] is True
    assert out["lines_added"] == 1
    write = [c for c in fake_client.calls if c.method == "write_file"][0]
    assert write.kwargs["content"] == "a\nNEW\nb\nc\n"


@pytest.mark.asyncio
async def test_insert_at_top(fake_client: FakeSandboxClient) -> None:
    fake_client.read_file_result = b"a\nb\n"
    out = await fn.sandbox_text_editor_tool(
        project_id="p1",
        command="insert",
        path="f.py",
        insert_line=0,
        new_str="HEADER",
    )
    assert out["success"] is True
    write = [c for c in fake_client.calls if c.method == "write_file"][0]
    assert write.kwargs["content"] == "HEADER\na\nb\n"


@pytest.mark.asyncio
async def test_insert_past_eof_rejected(
    fake_client: FakeSandboxClient,
) -> None:
    fake_client.read_file_result = b"a\nb\n"
    out = await fn.sandbox_text_editor_tool(
        project_id="p1",
        command="insert",
        path="f.py",
        insert_line=99,
        new_str="x",
    )
    assert out["success"] is False
    assert "past EOF" in out["error"]


@pytest.mark.asyncio
async def test_unknown_command_rejected(
    fake_client: FakeSandboxClient,
) -> None:
    out = await fn.sandbox_text_editor_tool(
        project_id="p1", command="frobnicate", path="x"
    )
    assert out["success"] is False
    assert "unknown command" in out["error"]


# ----------------------------------------------------------------------
# sandbox_shell
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_shell_returns_combined_envelope(
    fake_client: FakeSandboxClient,
) -> None:
    out = await fn.sandbox_shell_tool(project_id="p1", command="ls -la")
    assert out["success"] is True
    assert out["stdout"] == "out\n"
    assert out["exit_code"] == 0
    exec_calls = [c for c in fake_client.calls if c.method == "exec"]
    assert exec_calls[0].args == (("sh", "-c", "ls -la"),)


@pytest.mark.asyncio
async def test_shell_failure_marks_unsuccessful(
    fake_client: FakeSandboxClient,
) -> None:
    fake_client.exec_result = ExecResult(exit_code=2, stdout=b"", stderr=b"boom")
    out = await fn.sandbox_shell_tool(project_id="p1", command="false")
    assert out["success"] is False
    assert out["exit_code"] == 2
    assert out["stderr"] == "boom"


# ----------------------------------------------------------------------
# sandbox_search
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_search_emits_hits(fake_client: FakeSandboxClient) -> None:
    out = await fn.sandbox_search_tool(
        project_id="p1", pattern="alive", glob="**/*.py"
    )
    assert out["success"] is True
    assert out["hits"] == [{"path": "src/app.py", "line": 3, "snippet": "alive"}]
    search_calls = [c for c in fake_client.calls if c.method == "search"]
    assert search_calls[0].args == ("alive",)
    assert search_calls[0].kwargs["glob"] == "**/*.py"


# ----------------------------------------------------------------------
# sandbox_git — status / diff / log / commit / push
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_git_status(fake_client: FakeSandboxClient) -> None:
    out = await fn.sandbox_git_tool(project_id="p1", command="status")
    assert out["success"] is True
    assert out["unstaged"] == ["README.md"]
    assert out["is_clean"] is False


@pytest.mark.asyncio
async def test_git_diff_with_base_ref(fake_client: FakeSandboxClient) -> None:
    out = await fn.sandbox_git_tool(
        project_id="p1", command="diff", base_ref="main"
    )
    assert out["success"] is True
    assert "+world" in out["diff"]
    diff_calls = [c for c in fake_client.calls if c.method == "diff"]
    assert diff_calls[0].kwargs["base_ref"] == "main"


@pytest.mark.asyncio
async def test_git_log_parses_format(
    fake_client: FakeSandboxClient,
) -> None:
    fake_client.exec_result = ExecResult(
        exit_code=0,
        stdout=b"deadbee\tBot\tbot@x\t2026-01-01T00:00:00\tinitial\n"
        b"abcdef0\tDev\tdev@x\t2026-01-02T00:00:00\tsecond\n",
        stderr=b"",
    )
    out = await fn.sandbox_git_tool(project_id="p1", command="log", limit=2)
    assert out["success"] is True
    assert len(out["commits"]) == 2
    assert out["commits"][0]["sha"] == "deadbee"
    assert out["commits"][0]["subject"] == "initial"


@pytest.mark.asyncio
async def test_git_commit_returns_sha(fake_client: FakeSandboxClient) -> None:
    out = await fn.sandbox_git_tool(
        project_id="p1", command="commit", message="msg"
    )
    assert out["success"] is True
    assert out["commit"] == "a" * 40
    commit_calls = [c for c in fake_client.calls if c.method == "commit"]
    assert commit_calls[0].args == ("msg",)


@pytest.mark.asyncio
async def test_git_commit_requires_message(
    fake_client: FakeSandboxClient,
) -> None:
    out = await fn.sandbox_git_tool(project_id="p1", command="commit")
    assert out["success"] is False
    assert "message" in out["error"]


@pytest.mark.asyncio
async def test_git_push_passes_flags(fake_client: FakeSandboxClient) -> None:
    out = await fn.sandbox_git_tool(
        project_id="p1", command="push", set_upstream=False, force=True
    )
    assert out["success"] is True
    push = [c for c in fake_client.calls if c.method == "push"][0]
    assert push.kwargs == {
        "remote": "origin",
        "set_upstream": False,
        "force": True,
    }


@pytest.mark.asyncio
async def test_git_unknown_command(fake_client: FakeSandboxClient) -> None:
    out = await fn.sandbox_git_tool(project_id="p1", command="rebase")
    assert out["success"] is False
    assert "unknown command" in out["error"]


# ----------------------------------------------------------------------
# Workspace resolution
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resolve_workspace_pulls_branch_from_project(
    fake_client: FakeSandboxClient,
) -> None:
    """Branch comes from the project record + conversation_id contextvar.

    Guards against a regression where the LLM could pass a different
    project_id / conversation_id and bypass the server-side lookup.
    """
    await fn.sandbox_text_editor_tool(
        project_id="p1", command="view", path="README.md"
    )
    get_ws = [c for c in fake_client.calls if c.method == "get_workspace"][0]
    assert get_ws.kwargs["branch"] == "main"
    assert get_ws.kwargs["conversation_id"] == "conv1"
    assert get_ws.kwargs["user_id"] == "u1"
    assert get_ws.kwargs["project_id"] == "p1"
    assert get_ws.kwargs["repo"] == "owner/repo"


@pytest.mark.asyncio
async def test_missing_user_id_raises(fake_client: FakeSandboxClient) -> None:
    """If the execution flow forgot to seed user_id, tool fails clearly."""
    from app.modules.intelligence.tools.sandbox.context import _user_id_ctx

    _user_id_ctx.set(None)
    out = await fn.sandbox_text_editor_tool(
        project_id="p1", command="view", path="README.md"
    )
    assert out["success"] is False
    assert "user_id" in out["error"]
