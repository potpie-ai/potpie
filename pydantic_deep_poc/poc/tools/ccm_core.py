"""In-memory Code Changes Manager operations on RunContext."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from poc.managers.run_context import RunContext


def _mark_unverified(run: RunContext) -> None:
    run.verification_passed = False
    run.verification_report = ""


def _validate_rel_path(rel: str) -> str | None:
    rel = rel.strip()
    if not rel:
        return "path is empty"
    if rel in {".", ":", "}", "{", "]", "["}:
        return f"invalid path: {rel!r}"
    p = Path(rel)
    if p.is_absolute():
        return "absolute paths are not allowed"
    if ".." in p.parts:
        return "parent-directory traversal is not allowed"
    return None


def _full(run: RunContext, rel: str) -> Path:
    return Path(run.project_root) / rel


def _read_original(run: RunContext, rel: str) -> str:
    p = _full(run, rel)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def add_file_to_changes(
    run: RunContext, rel_path: str, content: str, description: str = ""
) -> dict[str, Any]:
    if err := _validate_rel_path(rel_path):
        return {"ok": False, "error": err}
    orig = _read_original(run, rel_path)
    run.code_changes[rel_path] = {
        "original": orig,
        "content": content,
        "description": description,
    }
    _mark_unverified(run)
    return {"ok": True, "path": rel_path, "lines_changed": len(content.splitlines())}


def update_file_in_changes(run: RunContext, rel_path: str, content: str) -> dict[str, Any]:
    if err := _validate_rel_path(rel_path):
        return {"ok": False, "error": err}
    if rel_path not in run.code_changes:
        orig = _read_original(run, rel_path)
        run.code_changes[rel_path] = {"original": orig, "content": content, "description": ""}
    else:
        run.code_changes[rel_path]["content"] = content
    _mark_unverified(run)
    return {"ok": True, "path": rel_path}


def get_file_from_changes(
    run: RunContext, rel_path: str, with_line_numbers: bool = False
) -> str:
    if rel_path in run.code_changes:
        text = run.code_changes[rel_path]["content"]
    else:
        text = _read_original(run, rel_path)
    if not with_line_numbers:
        return text
    lines = text.splitlines()
    return "\n".join(f"{i + 1:6d}|{line}" for i, line in enumerate(lines))


def update_file_lines(
    run: RunContext, rel_path: str, start_line: int, end_line: int, new_content: str
) -> dict[str, Any]:
    if err := _validate_rel_path(rel_path):
        return {"ok": False, "error": err}
    text = get_file_from_changes(run, rel_path, False)
    lines = text.splitlines()
    s, e = start_line - 1, end_line
    new_lines = new_content.splitlines()
    out = lines[:s] + new_lines + lines[e:]
    new_text = "\n".join(out) + ("\n" if out else "")
    if rel_path not in run.code_changes:
        run.code_changes[rel_path] = {
            "original": _read_original(run, rel_path),
            "content": "",
            "description": "",
        }
    run.code_changes[rel_path]["content"] = new_text
    _mark_unverified(run)
    return {"ok": True, "path": rel_path, "lines_changed": len(new_lines)}


def replace_in_file(
    run: RunContext, rel_path: str, old_str: str, new_str: str
) -> dict[str, Any]:
    if err := _validate_rel_path(rel_path):
        return {"ok": False, "error": err}
    text = get_file_from_changes(run, rel_path, False)
    c = text.count(old_str)
    if c != 1:
        return {"ok": False, "error": f"old_str count={c}, need exactly 1"}
    new_text = text.replace(old_str, new_str, 1)
    if rel_path not in run.code_changes:
        run.code_changes[rel_path] = {
            "original": _read_original(run, rel_path),
            "content": "",
            "description": "",
        }
    run.code_changes[rel_path]["content"] = new_text
    _mark_unverified(run)
    return {"ok": True, "path": rel_path}


def insert_lines(
    run: RunContext, rel_path: str, line: int, content: str, insert_after: bool = True
) -> dict[str, Any]:
    if err := _validate_rel_path(rel_path):
        return {"ok": False, "error": err}
    text = get_file_from_changes(run, rel_path, False)
    lines = text.splitlines()
    idx = line - 1 if insert_after else line - 2
    chunk = content.splitlines()
    ins = idx + 1 if insert_after else idx
    out = lines[:ins] + chunk + lines[ins:]
    new_text = "\n".join(out) + ("\n" if out else "")
    if rel_path not in run.code_changes:
        run.code_changes[rel_path] = {
            "original": _read_original(run, rel_path),
            "content": "",
            "description": "",
        }
    run.code_changes[rel_path]["content"] = new_text
    _mark_unverified(run)
    return {"ok": True, "path": rel_path}


def delete_lines(run: RunContext, rel_path: str, start_line: int, end_line: int | None = None) -> dict[str, Any]:
    if err := _validate_rel_path(rel_path):
        return {"ok": False, "error": err}
    text = get_file_from_changes(run, rel_path, False)
    lines = text.splitlines()
    e = end_line if end_line is not None else start_line
    s, e = start_line - 1, e
    out = lines[:s] + lines[e:]
    new_text = "\n".join(out) + ("\n" if out else "")
    if rel_path not in run.code_changes:
        run.code_changes[rel_path] = {
            "original": _read_original(run, rel_path),
            "content": "",
            "description": "",
        }
    run.code_changes[rel_path]["content"] = new_text
    _mark_unverified(run)
    return {"ok": True, "path": rel_path}


def delete_file_in_changes(run: RunContext, rel_path: str) -> dict[str, Any]:
    if err := _validate_rel_path(rel_path):
        return {"ok": False, "error": err}
    run.code_changes[rel_path] = {
        "original": _read_original(run, rel_path),
        "content": "",
        "description": "deleted",
        "deleted": True,
    }
    _mark_unverified(run)
    return {"ok": True, "path": rel_path}


def clear_file_from_changes(run: RunContext, rel_path: str) -> dict[str, Any]:
    if err := _validate_rel_path(rel_path):
        return {"ok": False, "error": err}
    run.code_changes.pop(rel_path, None)
    _mark_unverified(run)
    return {"ok": True}


def clear_all_changes(run: RunContext) -> dict[str, Any]:
    run.code_changes.clear()
    _mark_unverified(run)
    return {"ok": True}


def list_files_in_changes(run: RunContext) -> list[str]:
    return list(run.code_changes.keys())


def get_changes_summary(run: RunContext) -> dict[str, Any]:
    return {"files": len(run.code_changes), "paths": list(run.code_changes.keys())}


def get_changes_for_pr(run: RunContext) -> str:
    parts = []
    for p, st in run.code_changes.items():
        parts.append(f"## {p}\n{st.get('description', '')}\n")
    return "\n".join(parts) or "(no changes)"


def export_changes(run: RunContext, target_dir: str) -> dict[str, Any]:
    root = Path(target_dir)
    root.mkdir(parents=True, exist_ok=True)
    for rel, st in run.code_changes.items():
        if st.get("deleted"):
            continue
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(st["content"], encoding="utf-8")
    return {"ok": True, "dir": str(root)}


def show_updated_file(run: RunContext, rel_path: str) -> str:
    return get_file_from_changes(run, rel_path, False)


def get_file_diff(run: RunContext, rel_path: str) -> str:
    orig = run.code_changes.get(rel_path, {}).get("original", _read_original(run, rel_path))
    cur = get_file_from_changes(run, rel_path, False)
    return "\n".join(
        difflib.unified_diff(
            orig.splitlines(),
            cur.splitlines(),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
        )
    )


def revert_file(run: RunContext, rel_path: str) -> dict[str, Any]:
    if err := _validate_rel_path(rel_path):
        return {"ok": False, "error": err}
    if rel_path in run.code_changes:
        del run.code_changes[rel_path]
        _mark_unverified(run)
    return {"ok": True, "path": rel_path}


def show_diff_all(run: RunContext) -> str:
    chunks = []
    for rel in run.code_changes:
        chunks.append(get_file_diff(run, rel))
    return "\n\n".join(chunks) if chunks else "(no staged changes)"


def apply_to_worktree(run: RunContext) -> dict[str, Any]:
    if not run.code_changes:
        return {"ok": False, "error": "no staged changes in CCM", "files_written": 0}
    if not run.verification_passed:
        return {
            "ok": False,
            "error": "verification gate not satisfied; run verify and record PASS before apply",
            "files_written": 0,
        }
    if any(todo.get("status") in {"pending", "in_progress"} for todo in run.todos):
        return {
            "ok": False,
            "error": "cannot apply while todos are still pending or in progress",
            "files_written": 0,
        }
    root = Path(run.worktree_path or run.project_root)
    n = 0
    for rel, st in run.code_changes.items():
        if st.get("deleted"):
            p = root / rel
            if p.exists():
                p.unlink()
            n += 1
            continue
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(st["content"], encoding="utf-8")
        n += 1
    return {"ok": True, "files_written": n, "root": str(root)}
