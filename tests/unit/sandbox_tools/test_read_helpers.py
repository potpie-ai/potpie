"""Unit tests for ``read_helpers`` — the sandbox-tier shim used by the
legacy agent file-read tools.

These tests exercise the contract the read-tier callers (``fetch_file``,
``fetch_files_batch``, ``get_code_file_structure``, KG hydrators, etc.)
rely on:

  * ``acquire_analysis_workspace`` returns ``None`` on resolution failure
    so callers can fall through to GitHub.
  * ``read_file_via_sandbox`` slices by line range and returns ``None`` on
    any sandbox-side error (so callers fall through cleanly).
  * ``list_dir_via_sandbox`` produces the same nested tree shape that
    ``LocalRepoService`` / ``GithubService`` already emit.
  * ``read_files_batch_via_sandbox`` reuses one workspace handle for the
    whole batch and surfaces per-file errors without failing the batch.

We use the same in-memory ``FakeSandboxClient`` shape as
``test_sandbox_tools.py`` so the contract surface stays consistent.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from app.modules.intelligence.tools.sandbox import (
    client as sandbox_client_mod,
    read_helpers,
)
from app.modules.intelligence.tools.sandbox.context import set_run_context
from sandbox import FileEntry, WorkspaceHandle


@dataclass
class _Call:
    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeSandboxClient:
    """In-memory client used by the read-helper tier."""

    def __init__(self) -> None:
        self.calls: list[_Call] = []
        self.handle = WorkspaceHandle(
            workspace_id="ws_read",
            branch="main",
            backend_kind="local",
            local_path="/tmp/fake/work",
            remote_path=None,
        )
        self.files: dict[str, bytes] = {
            "README.md": b"line1\nline2\nline3\nline4\n",
            "src/app.py": b"print('hi')\n",
        }
        self.tree: dict[str, list[FileEntry]] = {
            ".": [
                FileEntry(name="src", is_dir=True, size=None),
                FileEntry(name="README.md", is_dir=False, size=24),
            ],
            "src": [
                FileEntry(name="app.py", is_dir=False, size=12),
            ],
        }

    async def get_workspace(self, **kwargs: Any) -> WorkspaceHandle:
        self.calls.append(_Call("get_workspace", kwargs=kwargs))
        return self.handle

    async def read_file(
        self, handle: WorkspaceHandle, path: str, *, max_bytes: int | None = None
    ) -> bytes:
        self.calls.append(_Call("read_file", kwargs={"path": path}))
        if path not in self.files:
            raise RuntimeError(f"missing: {path}")
        return self.files[path]

    async def list_dir(
        self, handle: WorkspaceHandle, path: str = "."
    ) -> list[FileEntry]:
        self.calls.append(_Call("list_dir", kwargs={"path": path}))
        if path not in self.tree:
            raise RuntimeError(f"no dir: {path}")
        return self.tree[path]


_SUMMARY = {
    "repo_name": "owner/repo",
    "base_branch": "main",
    "user_id": "u1",
    "repo_url": "https://github.com/owner/repo.git",
}


@pytest.fixture
def fake_client() -> Iterator[FakeSandboxClient]:
    client = FakeSandboxClient()
    sandbox_client_mod.set_sandbox_client(client)  # type: ignore[arg-type]
    set_run_context(user_id="u1", conversation_id="conv1", branch="main")
    # Patch the binding *imported into* read_helpers — that's what
    # acquire_analysis_workspace actually calls.
    with patch.object(
        read_helpers, "lookup_project_summary", return_value=_SUMMARY
    ):
        yield client
    sandbox_client_mod.set_sandbox_client(None)


# ----------------------------------------------------------------------
# acquire_analysis_workspace
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_acquire_workspace_returns_handle_on_happy_path(
    fake_client: FakeSandboxClient,
) -> None:
    handle = await read_helpers.acquire_analysis_workspace("p1")
    assert handle is fake_client.handle
    # The resolver must request ANALYSIS mode so we don't try to create a
    # per-conversation branch for a read-only consumer.
    [call] = [c for c in fake_client.calls if c.method == "get_workspace"]
    assert call.kwargs["branch"] == "main"


@pytest.mark.asyncio
async def test_acquire_workspace_returns_none_when_lookup_fails() -> None:
    set_run_context(user_id="u1", conversation_id="c1", branch="main")
    # Bind the sandbox client so resolve_workspace doesn't crash on a fresh
    # process — but the lookup raises so the helper must swallow.
    sandbox_client_mod.set_sandbox_client(FakeSandboxClient())  # type: ignore[arg-type]
    try:
        with patch.object(
            read_helpers,
            "lookup_project_summary",
            side_effect=ValueError("missing project"),
        ):
            result = await read_helpers.acquire_analysis_workspace("missing")
        assert result is None
    finally:
        sandbox_client_mod.set_sandbox_client(None)


@pytest.mark.asyncio
async def test_acquire_workspace_returns_none_when_repo_name_missing(
    fake_client: FakeSandboxClient,
) -> None:
    bad = {**_SUMMARY, "repo_name": ""}
    with patch.object(
        read_helpers, "lookup_project_summary", return_value=bad
    ):
        result = await read_helpers.acquire_analysis_workspace("p1")
    assert result is None


# ----------------------------------------------------------------------
# read_file_via_sandbox
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_read_file_returns_full_content(
    fake_client: FakeSandboxClient,
) -> None:
    text = await read_helpers.read_file_via_sandbox("p1", "README.md")
    assert text == "line1\nline2\nline3\nline4\n"


@pytest.mark.asyncio
async def test_read_file_slices_by_line_range(
    fake_client: FakeSandboxClient,
) -> None:
    text = await read_helpers.read_file_via_sandbox(
        "p1", "README.md", start_line=2, end_line=3
    )
    assert text == "line2\nline3\n"


@pytest.mark.asyncio
async def test_read_file_returns_none_on_read_error(
    fake_client: FakeSandboxClient,
) -> None:
    text = await read_helpers.read_file_via_sandbox("p1", "missing.txt")
    assert text is None


# ----------------------------------------------------------------------
# read_files_batch_via_sandbox
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_batch_read_acquires_workspace_once(
    fake_client: FakeSandboxClient,
) -> None:
    out = await read_helpers.read_files_batch_via_sandbox(
        "p1", ["README.md", "src/app.py"]
    )
    assert out is not None
    assert {entry["path"] for entry in out} == {"README.md", "src/app.py"}
    # Only one workspace acquisition for the entire batch.
    workspace_calls = [c for c in fake_client.calls if c.method == "get_workspace"]
    assert len(workspace_calls) == 1


@pytest.mark.asyncio
async def test_batch_read_surfaces_per_file_error(
    fake_client: FakeSandboxClient,
) -> None:
    out = await read_helpers.read_files_batch_via_sandbox(
        "p1", ["README.md", "missing.txt"]
    )
    assert out is not None
    by_path = {e["path"]: e for e in out}
    assert "content" in by_path["README.md"]
    assert "error" in by_path["missing.txt"]


# ----------------------------------------------------------------------
# list_dir_via_sandbox + format_dir_tree
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_dir_returns_nested_structure(
    fake_client: FakeSandboxClient,
) -> None:
    tree = await read_helpers.list_dir_via_sandbox("p1")
    assert tree is not None
    assert tree["type"] == "directory"
    children_by_name = {c["name"]: c for c in tree["children"]}
    assert children_by_name["src"]["type"] == "directory"
    assert children_by_name["README.md"]["type"] == "file"
    # Nested directory walked correctly.
    src_children = {c["name"]: c for c in children_by_name["src"]["children"]}
    assert src_children["app.py"]["type"] == "file"


def test_format_dir_tree_renders_indented_string() -> None:
    structure = {
        "name": "",
        "type": "directory",
        "children": [
            {
                "name": "src",
                "type": "directory",
                "children": [
                    {"name": "app.py", "type": "file"},
                ],
            },
            {"name": "README.md", "type": "file"},
        ],
    }
    rendered = read_helpers.format_dir_tree(structure)
    # Directories sort before files; nested children indent two spaces deeper.
    lines = rendered.splitlines()
    assert lines[0] == "  src"
    assert lines[1] == "    app.py"
    assert lines[2] == "  README.md"
