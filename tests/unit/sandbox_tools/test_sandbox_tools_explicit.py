"""Explicit-handle form of `create_sandbox_tools`.

The legacy zero-arg factory (back-compat with the existing harness) is
covered by ``test_sandbox_tools.py``. This file exercises the new
explicit form where the harness pre-resolves a `WorkspaceHandle` and
passes it in. Tools dispatch through the handle directly — no
contextvars, no DB lookups.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from app.modules.intelligence.tools.sandbox.tools import create_sandbox_tools
from sandbox import (
    Capabilities,
    ExecResult,
    FileEntry,
    GitStatus,
    Hit,
    PullRequest,
    WorkspaceHandle,
)


# ----------------------------------------------------------------------
# Fakes (lightweight; the legacy test file's full FakeSandboxClient is
# overkill for the wiring-level checks we need here)
# ----------------------------------------------------------------------
@dataclass
class _Call:
    method: str
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeClient:
    """In-memory stand-in for `SandboxClient` covering the methods the
    explicit toolset funnels through."""

    def __init__(self) -> None:
        self.calls: list[_Call] = []
        self.read_file_result: bytes = b""
        self.exec_result = ExecResult(exit_code=0, stdout=b"", stderr=b"")
        self.search_result: list[Hit] = []
        self.status_result = GitStatus(
            branch="agent/edits-conv1",
            is_clean=True,
            staged=(),
            unstaged=(),
            untracked=(),
        )
        self.diff_result = ""
        self.commit_sha = "f" * 40

    async def read_file(self, handle: Any, path: str, *, max_bytes: int | None = None) -> bytes:
        self.calls.append(_Call("read_file", kwargs={"path": path}))
        return self.read_file_result

    async def write_file(self, handle: Any, path: str, content: bytes | str) -> None:
        self.calls.append(_Call("write_file", kwargs={"path": path}))

    async def list_dir(self, handle: Any, path: str = ".") -> list[FileEntry]:
        self.calls.append(_Call("list_dir", kwargs={"path": path}))
        return [FileEntry(name="README.md", is_dir=False, size=12)]

    async def search(self, handle: Any, pattern: str, **kwargs: Any) -> list[Hit]:
        self.calls.append(_Call("search", args=(pattern,), kwargs=kwargs))
        return self.search_result

    async def exec(self, handle: Any, cmd: list[str], **kwargs: Any) -> ExecResult:
        self.calls.append(_Call("exec", args=(tuple(cmd),), kwargs=kwargs))
        return self.exec_result

    async def status(self, handle: Any) -> GitStatus:
        self.calls.append(_Call("status"))
        return self.status_result

    async def diff(self, handle: Any, *, base_ref: str | None = None, paths: list[str] | None = None) -> str:
        self.calls.append(_Call("diff"))
        return self.diff_result

    async def commit(self, handle: Any, message: str, **kwargs: Any) -> str:
        self.calls.append(_Call("commit", args=(message,)))
        return self.commit_sha

    async def push(self, handle: Any, **kwargs: Any) -> None:
        self.calls.append(_Call("push", kwargs=kwargs))

    async def create_pull_request(
        self, handle: Any, **kwargs: Any
    ) -> PullRequest:
        self.calls.append(_Call("create_pull_request", kwargs=kwargs))
        return PullRequest(
            id=99,
            url="https://example.test/pulls/99",
            title=str(kwargs.get("title", "")),
            head_branch=str(kwargs.get("head_branch") or handle.branch),
            base_branch=str(kwargs.get("base_branch", "main")),
            backend_kind="fake",
        )


def _writable_handle(tmp_path: Path) -> WorkspaceHandle:
    return WorkspaceHandle(
        workspace_id="ws_w",
        branch="agent/edits-conv1",
        backend_kind="local",
        local_path=str(tmp_path),
        capabilities=Capabilities(writable=True, isolated=True, persistent=True),
    )


def _readonly_handle(tmp_path: Path) -> WorkspaceHandle:
    return WorkspaceHandle(
        workspace_id="ws_r",
        branch="main",
        backend_kind="local",
        local_path=str(tmp_path),
        capabilities=Capabilities(writable=False, isolated=False, persistent=True),
    )


# ----------------------------------------------------------------------
# Toolset shape
# ----------------------------------------------------------------------
def test_writable_handle_yields_full_toolset(tmp_path: Path) -> None:
    """Writable handle: search + text_editor + shell + git, with bound
    schemas (no project_id field)."""
    client = FakeClient()
    handle = _writable_handle(tmp_path)
    tools = create_sandbox_tools(client=client, handle=handle)
    names = {t.name for t in tools}
    assert names == {
        "sandbox_search",
        "sandbox_text_editor",
        "sandbox_shell",
        "sandbox_git",
    }
    # Bound schemas drop project_id.
    for tool in tools:
        assert "project_id" not in tool.args_schema.model_fields, (
            f"{tool.name}: bound form must omit project_id"
        )


def test_readonly_handle_drops_write_tools(tmp_path: Path) -> None:
    """Read-only handle: only sandbox_search is exposed (capability gate)."""
    client = FakeClient()
    handle = _readonly_handle(tmp_path)
    tools = create_sandbox_tools(client=client, handle=handle)
    names = {t.name for t in tools}
    assert names == {"sandbox_search"}


def test_enforce_capabilities_off_exposes_everything(tmp_path: Path) -> None:
    """When the caller opts out of gating, even read-only handles get
    the full toolset; the runtime is responsible for refusing writes."""
    client = FakeClient()
    handle = _readonly_handle(tmp_path)
    tools = create_sandbox_tools(
        client=client, handle=handle, enforce_capabilities=False
    )
    assert {t.name for t in tools} == {
        "sandbox_search",
        "sandbox_text_editor",
        "sandbox_shell",
        "sandbox_git",
    }


def test_create_sandbox_tools_rejects_partial_args() -> None:
    """Passing only one of (client, handle) is a programmer error."""
    with pytest.raises(ValueError, match="both"):
        create_sandbox_tools(client=FakeClient())  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Bound dispatch: tool functions go straight through to the fake client,
# no contextvars touched.
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_contextvars() -> Iterator[None]:
    """Sanity guard: bound tools must NOT depend on the run-context vars.

    We don't set any context here. The legacy form would explode; the
    bound form must succeed.
    """
    yield


@pytest.mark.asyncio
async def test_bound_search_dispatches_through_client(tmp_path: Path) -> None:
    client = FakeClient()
    client.search_result = [Hit(path="src/app.py", line=2, snippet="match")]
    handle = _writable_handle(tmp_path)
    tools = {
        t.name: t for t in create_sandbox_tools(client=client, handle=handle)
    }

    out = await tools["sandbox_search"].func(pattern="match")
    assert out["success"] is True
    assert out["pattern"] == "match"
    assert len(out["hits"]) == 1
    # Exactly one search call routed through the bound client.
    assert [c.method for c in client.calls] == ["search"]


@pytest.mark.asyncio
async def test_bound_shell_dispatches_through_client(tmp_path: Path) -> None:
    client = FakeClient()
    client.exec_result = ExecResult(exit_code=0, stdout=b"hello\n", stderr=b"")
    handle = _writable_handle(tmp_path)
    tools = {
        t.name: t for t in create_sandbox_tools(client=client, handle=handle)
    }

    out = await tools["sandbox_shell"].func(command="echo hello")
    assert out["success"] is True
    assert out["stdout"] == "hello\n"
    assert out["branch"] == "agent/edits-conv1"
    assert client.calls[0].method == "exec"


@pytest.mark.asyncio
async def test_bound_git_status_dispatches_through_client(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    handle = _writable_handle(tmp_path)
    tools = {
        t.name: t for t in create_sandbox_tools(client=client, handle=handle)
    }

    out = await tools["sandbox_git"].func(command="status")
    assert out["success"] is True
    assert out["branch"] == "agent/edits-conv1"
    assert client.calls[0].method == "status"


# ----------------------------------------------------------------------
# sandbox_pr (P7)
# ----------------------------------------------------------------------
def test_pr_tool_only_emitted_when_repo_name_passed(tmp_path: Path) -> None:
    """Without `pr_repo_name` the toolset doesn't include sandbox_pr —
    the workspace handle alone doesn't carry the repo identity."""
    client = FakeClient()
    handle = _writable_handle(tmp_path)
    no_pr = {
        t.name for t in create_sandbox_tools(client=client, handle=handle)
    }
    assert "sandbox_pr" not in no_pr

    with_pr = {
        t.name
        for t in create_sandbox_tools(
            client=client, handle=handle, pr_repo_name="owner/repo"
        )
    }
    assert "sandbox_pr" in with_pr


def test_pr_tool_dropped_for_readonly_handle(tmp_path: Path) -> None:
    """sandbox_pr is gated on writable, even when pr_repo_name is set."""
    client = FakeClient()
    handle = _readonly_handle(tmp_path)
    tools = {
        t.name
        for t in create_sandbox_tools(
            client=client, handle=handle, pr_repo_name="owner/repo"
        )
    }
    assert "sandbox_pr" not in tools


@pytest.mark.asyncio
async def test_pr_tool_dispatches_to_client(tmp_path: Path) -> None:
    client = FakeClient()
    handle = _writable_handle(tmp_path)
    tools = {
        t.name: t
        for t in create_sandbox_tools(
            client=client,
            handle=handle,
            pr_repo_name="owner/repo",
            pr_repo_url="https://github.com/owner/repo.git",
        )
    }
    out = await tools["sandbox_pr"].func(
        title="Add foo",
        body="Implements foo.",
        base_branch="main",
    )
    assert out["success"] is True
    assert out["pr_id"] == 99
    assert out["url"].endswith("/pulls/99")
    # Exactly one PR call dispatched.
    assert [c.method for c in client.calls if c.method == "create_pull_request"] == [
        "create_pull_request"
    ]
