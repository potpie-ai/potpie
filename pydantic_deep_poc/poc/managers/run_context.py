"""Per-run state: CCM, todos, requirements, worktree path."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from poc.repo_setup import create_worktree


@dataclass
class RunContext:
    """Mirrors TodoManager + CodeChangesManager + RequirementsManager (in-memory)."""

    code_changes: dict[str, dict[str, Any]] = field(default_factory=dict)
    todos: list[dict[str, Any]] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_root: str = ""
    worktree_path: str = ""
    branch_name: str = ""
    verification_passed: bool = False
    verification_report: str = ""
    shell_failures: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.code_changes.clear()
        self.todos.clear()
        self.requirements.clear()
        self.session_id = str(uuid.uuid4())
        self.verification_passed = False
        self.verification_report = ""
        self.shell_failures.clear()


def init_run_context(branch: str | None = None) -> RunContext:
    """Fresh worktree per run; sets project_root to worktree path."""
    br = branch or f"poc/{uuid.uuid4().hex[:8]}"
    path = create_worktree(br)
    ctx = RunContext()
    ctx.branch_name = br
    ctx.worktree_path = path
    ctx.project_root = path
    return ctx
