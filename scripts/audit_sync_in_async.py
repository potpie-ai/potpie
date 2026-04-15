#!/usr/bin/env python3
"""
Audit script: find async functions that contain sync DB (or blocking) usage.

Sync DB patterns that block the event loop when called from async:
- SessionLocal()
- self.db / .db then .query, .add, .commit, .rollback, .execute, .refresh, .flush
- Service(database_session).sync_method() when that method does DB

Run from repo root: python scripts/audit_sync_in_async.py
Exits 0 if only allowed patterns (e.g. run_in_executor) are found; 1 if blocking found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Sync DB patterns: (pattern_type, description)
# We look for these inside async function bodies.
SYNC_DB_PATTERNS = [
    "SessionLocal",
    ".query(",
    ".add(",
    ".commit(",
    ".rollback(",
    ".execute(",
    ".refresh(",
    ".flush(",
]

# Substrings that suggest the sync call is wrapped in run_in_executor (allowed).
RUN_IN_EXECUTOR_MARKERS = ["run_in_executor", "_sync_", "run_in_executor"]

APP_DIR = Path(__file__).resolve().parent.parent / "app"
SKIP_DIRS = {"__pycache__", ".pytest_cache", "tests", "migrations"}

# (path_suffix, async_function_name) to skip (docstrings, async session usage, etc.)
SKIP_FUNCTIONS = {
    ("celery/tasks/base_task.py", "async_db"),  # docstring mentions SessionLocal
    ("core/database.py", "get_async_db"),  # uses AsyncSessionLocal, not SessionLocal
}


def get_async_functions_with_ranges(tree: ast.AST) -> list[tuple[str, int, int]]:
    """Return list of (name, start_line, end_line) for async functions."""
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            end = getattr(node, "end_lineno", node.lineno)
            result.append((node.name, node.lineno, end))
    return result


def get_nested_sync_def_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    """Return (start, end) line ranges for sync `def` nodes nested inside async defs.

    These are the run_in_executor helper pattern: a sync inner function defined
    inside an async function and passed to run_in_executor. Sync DB calls inside
    such helpers are intentional and should not be flagged as blocking.
    """
    ranges = []

    def _walk(node: ast.AST, inside_async: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.AsyncFunctionDef):
                _walk(child, inside_async=True)
            elif isinstance(child, ast.FunctionDef):
                if inside_async:
                    end = getattr(child, "end_lineno", child.lineno)
                    ranges.append((child.lineno, end))
                    # Don't recurse further; nested async inside sync-inside-async
                    # is unusual and we treat conservatively.
                else:
                    _walk(child, inside_async=False)
            else:
                _walk(child, inside_async=inside_async)

    _walk(tree, inside_async=False)
    return ranges


def line_contains_sync_db(line: str) -> list[str]:
    """If line contains sync DB usage (not async_db), return list of pattern names found."""
    found = []
    # SessionLocal() but not AsyncSessionLocal()
    if "SessionLocal()" in line and "AsyncSessionLocal" not in line:
        found.append("SessionLocal")
    # Sync session method calls: self.db.X, db.X, session.X (exclude async_db)
    if "async_db." in line or "async_db)" in line:
        return []
    # Exclude lines that are already properly awaited async session calls
    # e.g. "await self.session.execute(...)" or "result = await session.commit()"
    if "await self.session." in line or "await session." in line:
        return []
    for pattern in [".query(", ".add(", ".commit(", ".rollback(", ".execute(", ".refresh(", ".flush("]:
        if pattern not in line:
            continue
        # Must be on a sync session: self.db, " db.", or session.
        if "self.db" in line or " db." in line or " db)" in line or "session." in line:
            # Exclude set.add / list.append etc: require db or session before the method
            if pattern == ".add(" and "db.add(" not in line and "self.db.add(" not in line and "session.add(" not in line:
                continue
            # AsyncSession.add() is sync by design (no I/O); only .commit()/.flush() do I/O.
            # Exclude self.session.add() — it's the standard SQLAlchemy async add pattern.
            if pattern == ".add(" and "self.session.add(" in line:
                continue
            found.append(pattern)
    return found


def line_is_likely_run_in_executor(line: str) -> bool:
    """True if line looks like sync is inside run_in_executor."""
    return any(m in line for m in RUN_IN_EXECUTOR_MARKERS)


def audit_file(path: Path, app_dir: Path) -> list[dict]:
    """Return list of issues: {name, line, patterns, line_text, in_executor}."""
    try:
        text = path.read_text()
    except Exception as e:
        return [{"error": str(e)}]
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    async_funcs = get_async_functions_with_ranges(tree)
    if not async_funcs:
        return []

    try:
        rel_path = path.relative_to(app_dir)
        rel_str = str(rel_path).replace("\\", "/")
    except ValueError:
        rel_str = ""

    # Build a set of line numbers that are inside nested sync defs (run_in_executor helpers).
    nested_sync_ranges = get_nested_sync_def_ranges(tree)
    nested_sync_lines: set[int] = set()
    for ns_start, ns_end in nested_sync_ranges:
        nested_sync_lines.update(range(ns_start, ns_end + 1))

    issues = []
    for name, start, end in async_funcs:
        if (rel_str, name) in SKIP_FUNCTIONS:
            continue
        for i in range(start - 1, min(end, len(lines))):
            lineno = i + 1
            # Skip lines inside nested sync defs — those are run_in_executor helpers.
            if lineno in nested_sync_lines:
                continue
            line = lines[i]
            patterns = line_contains_sync_db(line)
            if not patterns:
                continue
            in_executor = line_is_likely_run_in_executor(line)
            issues.append(
                {
                    "func": name,
                    "line_no": i + 1,
                    "patterns": patterns,
                    "line_text": line.strip()[:80],
                    "in_executor": in_executor,
                }
            )
    return issues


def main() -> int:
    app_path = APP_DIR
    if not app_path.is_dir():
        print("App dir not found:", app_path)
        return 2

    all_reports: list[tuple[Path, list]] = []
    for py in sorted(app_path.rglob("*.py")):
        parts = py.relative_to(app_path).parts
        if any(s in parts for s in SKIP_DIRS):
            continue
        issues = audit_file(py, app_path)
        if issues:
            all_reports.append((py, issues))

    # Print report
    blocking_count = 0
    for path, issues in all_reports:
        rel = path.relative_to(app_path.parent)
        blocking_in_file = [i for i in issues if not i.get("in_executor") and "error" not in i]
        if blocking_in_file:
            blocking_count += len(blocking_in_file)
        allowed = [i for i in issues if i.get("in_executor") and "error" not in i]
        if not issues or ("error" in issues[0]):
            continue
        print(f"\n{rel}")
        print("-" * 60)
        for i in issues:
            if "error" in i:
                print(f"  Error: {i['error']}")
                continue
            tag = " [run_in_executor?]" if i.get("in_executor") else " <<< BLOCKING"
            print(f"  {i['func']} L{i['line_no']}: {i['patterns']}{tag}")
            print(f"      {i['line_text']}")

    print("\n" + "=" * 60)
    if blocking_count > 0:
        print(f"Total: {blocking_count} sync-DB-in-async usages (potential blocking)")
        return 1
    print("No sync-DB-in-async blocking usages found (or all under run_in_executor).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
