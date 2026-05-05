"""Sandbox-backed read helpers for legacy agent tools.

The pre-sandbox file-read tools (``fetch_file``, ``get_code_file_structure``,
KG-backed code readers, etc.) used to reach the codebase through one of two
paths:

  * VS Code extension tunnel — for users running Potpie locally with the
    extension active. Reads the user's IDE state (incl. unsaved buffers).
  * ``CodeProviderService`` — GitHub API or ``LocalRepoService`` (the
    ``.repos`` worktree managed by ``RepoManager``).

The second tier was the legacy ``RepoManager`` surface, which we are
phasing out. This module gives those tools a *third* tier — a sandbox
``WorkspaceMode.ANALYSIS`` view of the project — so they can read from
the same on-disk worktree the new ``sandbox_text_editor`` writes to,
without each tool having to know about the workspace lifecycle.

Order of preference in caller code:

  1. VS Code tunnel (unchanged — IDE state is irreplaceable).
  2. **Sandbox workspace (this module).**
  3. ``CodeProviderService`` GitHub fallback (last-resort safety net).

Every helper here is best-effort: it returns ``None`` on any failure
(missing contextvar, project lookup miss, workspace resolution failure,
read error) so callers can fall through cleanly.
"""

from __future__ import annotations

from typing import Any, List, Optional

from sandbox import WorkspaceMode

from app.modules.intelligence.tools.sandbox.client import (
    get_sandbox_client,
    lookup_project_summary,
    resolve_workspace,
)
from app.modules.intelligence.tools.sandbox.context import get_user_id
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


async def acquire_analysis_workspace(project_id: str) -> Optional[Any]:
    """Resolve an ``ANALYSIS`` workspace handle for ``project_id``.

    Returns ``None`` on any failure — caller is responsible for falling
    through to the legacy code path. We deliberately swallow exceptions
    here so a flaky sandbox never breaks file-read tools that have a
    GitHub fallback available.
    """
    try:
        summary = lookup_project_summary(project_id)
    except Exception:  # noqa: BLE001
        logger.debug(
            "[sandbox-read] project lookup failed for %s — falling through",
            project_id,
        )
        return None

    user_id = get_user_id() or summary.get("user_id")
    if not user_id:
        return None
    repo_name = summary.get("repo_name")
    if not repo_name:
        return None
    branch = summary.get("base_branch") or "main"
    repo_url = summary.get("repo_url") or None

    try:
        return await resolve_workspace(
            user_id=user_id,
            project_id=project_id,
            repo_name=repo_name,
            repo_url=repo_url,
            branch=branch,
            base_ref=branch,
            create_branch=False,
            mode=WorkspaceMode.ANALYSIS,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "[sandbox-read] resolve_workspace failed for %s: %s — falling through",
            project_id,
            exc,
        )
        return None


async def read_file_via_sandbox(
    project_id: str,
    file_path: str,
    *,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> Optional[str]:
    """Read ``file_path`` from the project's analysis workspace.

    ``start_line`` / ``end_line`` are 1-indexed (inclusive). Returns the
    sliced text on success, ``None`` on any failure so callers can fall
    through to GitHub / the tunnel.
    """
    handle = await acquire_analysis_workspace(project_id)
    if handle is None:
        return None
    client = get_sandbox_client()
    try:
        data = await client.read_file(handle, file_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "[sandbox-read] read_file %s failed for %s: %s",
            file_path,
            project_id,
            exc,
        )
        return None
    text = data.decode("utf-8", errors="replace")
    if start_line is None and end_line is None:
        return text
    lines = text.splitlines(keepends=True)
    start_idx = (start_line - 1) if start_line and start_line > 0 else 0
    end_idx = end_line if end_line and end_line > 0 else len(lines)
    return "".join(lines[start_idx:end_idx])


async def read_files_batch_via_sandbox(
    project_id: str,
    paths: List[str],
) -> Optional[List[dict]]:
    """Read multiple files from the analysis workspace in one workspace acquisition.

    Returns a list of ``{path, content}`` or ``{path, error}`` entries on
    success, ``None`` if the workspace itself can't be resolved (so the
    caller can fall through). Per-file errors don't fail the batch — they
    surface in the per-file ``error`` field.
    """
    handle = await acquire_analysis_workspace(project_id)
    if handle is None:
        return None
    client = get_sandbox_client()
    out: List[dict] = []
    for path in paths:
        try:
            data = await client.read_file(handle, path)
            text = data.decode("utf-8", errors="replace")
            out.append(
                {
                    "path": path,
                    "content": text,
                    "line_count": len(text.splitlines()),
                }
            )
        except Exception as exc:  # noqa: BLE001
            out.append({"path": path, "error": str(exc)})
    return out


async def list_dir_via_sandbox(
    project_id: str,
    path: Optional[str] = None,
) -> Optional[dict]:
    """List a directory recursively via the analysis workspace.

    Returns a nested ``{name, type: "directory"|"file", children: [...]}``
    structure matching the format already used by ``LocalRepoService`` and
    ``GithubService``. Returns ``None`` on workspace resolution failure.

    The recursion is bounded at ~2000 entries to mirror the legacy
    GitHub tree-API behaviour and keep the LLM payload small.
    """
    handle = await acquire_analysis_workspace(project_id)
    if handle is None:
        return None
    client = get_sandbox_client()
    rel = path or "."

    return await _walk_dir(client, handle, rel, depth_remaining=12, budget=[2000])


async def _walk_dir(
    client: Any,
    handle: Any,
    rel: str,
    *,
    depth_remaining: int,
    budget: list,
) -> Optional[dict]:
    """Recursive directory walker with a shared entry budget.

    ``budget`` is a single-element list so the recursion can decrement it
    in place without explicit return-value plumbing.
    """
    if budget[0] <= 0 or depth_remaining < 0:
        return None
    try:
        entries = await client.list_dir(handle, rel)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[sandbox-read] list_dir(%s) failed: %s", rel, exc)
        return None
    children: list = []
    name = rel.split("/")[-1] if rel and rel != "." else ""
    node = {"name": name, "type": "directory", "children": children}
    for entry in entries:
        budget[0] -= 1
        if budget[0] <= 0:
            break
        # Skip the obvious noise that explodes any walk.
        if entry.name in {".git", "node_modules", "__pycache__", ".venv", ".mypy_cache"}:
            continue
        child_rel = entry.name if rel in ("", ".") else f"{rel}/{entry.name}"
        if entry.is_dir:
            sub = await _walk_dir(
                client, handle, child_rel,
                depth_remaining=depth_remaining - 1,
                budget=budget,
            )
            if sub is not None:
                children.append(sub)
        else:
            children.append({"name": entry.name, "type": "file"})
    return node


def format_dir_tree(structure: dict) -> str:
    """Render the nested dir structure as the same indented string the
    legacy ``GetCodeFileStructureTool`` returns. Kept here so tools don't
    each duplicate the formatter.
    """
    if not structure:
        return ""

    def _fmt(node: dict, depth: int) -> List[str]:
        out: List[str] = []
        if depth > 0:
            out.append("  " * depth + (node.get("name") or ""))
        for child in sorted(
            node.get("children", []) or [],
            key=lambda x: (x.get("type") != "directory", x.get("name", "").lower()),
        ):
            if child.get("type") == "directory":
                out.extend(_fmt(child, depth + 1))
            else:
                out.append("  " * (depth + 1) + child.get("name", ""))
        return out

    return "\n".join(_fmt(structure, 0))


__all__ = [
    "acquire_analysis_workspace",
    "read_file_via_sandbox",
    "read_files_batch_via_sandbox",
    "list_dir_via_sandbox",
    "format_dir_tree",
]
