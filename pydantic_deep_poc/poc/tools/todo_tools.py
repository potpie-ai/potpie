"""Todo operations on RunContext.todos (list of dicts)."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic_ai import RunContext

from poc.managers.deps import PoCDeepDeps


def _run(ctx: RunContext[PoCDeepDeps]) -> Any:
    return ctx.deps.poc_run


async def read_todos(ctx: RunContext[PoCDeepDeps]) -> str:
    return str(_run(ctx).todos)


async def write_todos(ctx: RunContext[PoCDeepDeps], todos: list[dict[str, Any]]) -> str:
    _run(ctx).todos = list(todos)
    return "ok"


async def add_todo(
    ctx: RunContext[PoCDeepDeps],
    content: str,
    active_form: str = "",
) -> str:
    tid = str(uuid.uuid4())[:8]
    _run(ctx).todos.append(
        {
            "id": tid,
            "content": content,
            "active_form": active_form or content,
            "status": "pending",
            "subtasks": [],
            "deps": [],
        }
    )
    return tid


async def update_todo_status(
    ctx: RunContext[PoCDeepDeps], todo_id: str, status: str
) -> str:
    for t in _run(ctx).todos:
        if t.get("id") == todo_id:
            t["status"] = status
            return "ok"
    return "not found"


async def remove_todo(ctx: RunContext[PoCDeepDeps], todo_id: str) -> str:
    run = _run(ctx)
    run.todos = [t for t in run.todos if t.get("id") != todo_id]
    return "ok"


async def add_subtask(
    ctx: RunContext[PoCDeepDeps], parent_id: str, content: str
) -> str:
    for t in _run(ctx).todos:
        if t.get("id") == parent_id:
            t.setdefault("subtasks", []).append(content)
            return "ok"
    return "parent not found"


async def set_dependency(
    ctx: RunContext[PoCDeepDeps], todo_id: str, depends_on_id: str
) -> str:
    for t in _run(ctx).todos:
        if t.get("id") == todo_id:
            t.setdefault("deps", []).append(depends_on_id)
            return "ok"
    return "not found"


async def get_available_tasks(ctx: RunContext[PoCDeepDeps]) -> str:
    return str([t for t in _run(ctx).todos if t.get("status") != "completed"])
