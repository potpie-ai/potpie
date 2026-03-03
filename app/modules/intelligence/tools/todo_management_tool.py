"""
Todo management using pydantic-ai-todo.

Uses https://github.com/vstorm-co/pydantic-ai-todo for task planning and tracking:
read_todos, write_todos, add_todo, update_todo_status, remove_todo, and with
enable_subtasks: add_subtask, set_dependency, get_available_tasks.
"""

from contextvars import ContextVar
from typing import Any, Annotated, List, Optional

from pydantic import BaseModel, Field, WithJsonSchema

from pydantic_ai_todo import (
    TodoStorage,
    create_todo_toolset,
)
from pydantic_ai_todo.types import Todo, TodoItem
from app.modules.intelligence.tools.tool_schema import OnyxTool


# Context variable for todo storage - provides isolation per execution context
_todo_storage_ctx: ContextVar[Optional[TodoStorage]] = ContextVar(
    "_todo_storage_ctx", default=None
)


def get_todo_storage() -> TodoStorage:
    """Get the current todo storage for this execution context, creating one if needed."""
    storage = _todo_storage_ctx.get()
    if storage is None:
        storage = TodoStorage()
        _todo_storage_ctx.set(storage)
    return storage


def _reset_todo_manager() -> None:
    """Reset the todo list for a new agent run (clear all todos in current context)."""
    get_todo_storage().todos.clear()


def create_todo_management_toolset():
    """Create a pydantic-ai-todo toolset for use with Agent(toolsets=[...]).

    Uses the current context storage so that init_managers (which clears storage)
    keeps the same storage instance and the cached supervisor agent's toolset
    continues to work with cleared state.
    """
    return create_todo_toolset(
        storage=get_todo_storage(),
        enable_subtasks=True,
    )


# --- Sync wrappers for ToolService (same behavior as library's sync toolset) ---


def _get_todo_by_id(todo_id: str) -> Optional[Todo]:
    storage = get_todo_storage()
    for todo in storage.todos:
        if todo.id == todo_id:
            return todo
    return None


def _get_status_icon(status: str) -> str:
    icons = {
        "pending": "[ ]",
        "in_progress": "[*]",
        "completed": "[x]",
        "blocked": "[!]",
    }
    return icons.get(status, "[ ]")


def _is_blocked(todo: Todo) -> bool:
    for dep_id in todo.depends_on:
        dep = _get_todo_by_id(dep_id)
        if dep and dep.status != "completed":
            return True
    return False


def _format_current_todo_list() -> str:
    """Format the current todo list for display (used by tool_helpers for streaming)."""
    storage = get_todo_storage()
    if not storage.todos:
        return "📋 **Current Todo List:**\nNo todos remaining.\n"
    lines = ["📋 **Current Todo List:**"]
    for i, todo in enumerate(storage.todos, 1):
        icon = _get_status_icon(todo.status)
        lines.append(f"{i}. {icon} **{todo.content}** (ID: {todo.id}) - {todo.status}")
    return "\n".join(lines) + "\n"


# Pydantic input models for tool arguments (ToolService tools use these)
class AddTodoInput(BaseModel):
    content: str = Field(description="The task description in imperative form")
    active_form: str = Field(
        description="Present continuous form shown during execution (e.g., 'Implementing feature X')"
    )


class UpdateTodoStatusInput(BaseModel):
    todo_id: str = Field(description="ID of the todo to update")
    status: str = Field(
        description="New status: pending, in_progress, completed, or blocked"
    )


class RemoveTodoInput(BaseModel):
    todo_id: str = Field(description="ID of the todo to remove")


class AddSubtaskInput(BaseModel):
    parent_id: str = Field(description="ID of the parent todo")
    content: str = Field(description="The subtask description in imperative form")
    active_form: str = Field(
        description="Present continuous form (e.g., 'Implementing sub-step')"
    )


class SetDependencyInput(BaseModel):
    todo_id: str = Field(description="ID of the todo that depends on another")
    depends_on_id: str = Field(
        description="ID of the todo that must be completed first"
    )


class ReadTodosInput(BaseModel):
    hierarchical: bool = Field(
        default=False,
        description="If True, display todos as a tree with subtasks indented",
    )


# Inline JSON schema for write_todos items so APIs that don't resolve $ref get an
# explicit "type": "object" at items (required by e.g. OpenAI-compatible function calling).
_WRITE_TODOS_ITEMS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "content": {
            "type": "string",
            "description": "The task description in imperative form (e.g., 'Implement feature X')",
        },
        "status": {
            "type": "string",
            "enum": ["pending", "in_progress", "completed", "blocked"],
            "description": "Task status: pending, in_progress, completed, or blocked",
        },
        "active_form": {
            "type": "string",
            "description": "Present continuous form during execution (e.g., 'Implementing feature X')",
        },
        "parent_id": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
            "description": "ID of parent todo for subtask hierarchy. None for root tasks.",
        },
        "depends_on": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of todo IDs that must be completed before this task.",
        },
    },
    "required": ["content", "status", "active_form"],
}


class WriteTodosInput(BaseModel):
    """Input for bulk write_todos. Replaces the entire list with the given items."""

    todos: Annotated[
        List[TodoItem],
        Field(
            description="List of todo items (each with content, status, active_form; optional id, parent_id, depends_on)"
        ),
        WithJsonSchema({"type": "array", "items": _WRITE_TODOS_ITEMS_JSON_SCHEMA}),
    ]


def read_todos_tool(hierarchical: bool = False) -> str:
    """List all tasks (supports hierarchical view)."""
    storage = get_todo_storage()
    if not storage.todos:
        return "No todos in the list. Use write_todos or add_todo to create tasks."
    if hierarchical:
        return _format_hierarchical(storage.todos)
    lines = ["Current todos:"]
    for i, todo in enumerate(storage.todos, 1):
        icon = _get_status_icon(todo.status)
        lines.append(f"{i}. {icon} [{todo.id}] {todo.content}")
        if todo.parent_id:
            lines.append(f" (subtask of: {todo.parent_id})")
        if todo.depends_on:
            lines.append(f" (depends on: {', '.join(todo.depends_on)})")
    counts = {"pending": 0, "in_progress": 0, "completed": 0, "blocked": 0}
    for todo in storage.todos:
        counts[todo.status] = counts.get(todo.status, 0) + 1
    summary = f"{counts['completed']} completed"
    if counts.get("blocked", 0) > 0:
        summary += f", {counts['blocked']} blocked"
    summary += f", {counts['in_progress']} in progress, {counts['pending']} pending"
    return "\n".join(lines) + f"\n\nSummary: {summary}"


def _format_hierarchical(todos: List[Todo]) -> str:
    children_map: dict[Optional[str], List[Todo]] = {None: []}
    for todo in todos:
        pid = todo.parent_id
        if pid not in children_map:
            children_map[pid] = []
        children_map[pid].append(todo)
    lines = ["Current todos (hierarchical view):"]
    counter = [0]

    def render(parent_id: Optional[str], depth: int) -> None:
        for todo in children_map.get(parent_id, []):
            counter[0] += 1
            icon = _get_status_icon(todo.status)
            lines.append(
                " " * depth + f"{counter[0]}. {icon} [{todo.id}] {todo.content}"
            )
            if todo.depends_on:
                lines.append(" " * depth + f" depends on: {', '.join(todo.depends_on)}")
            render(todo.id, depth + 1)

    render(None, 0)
    return "\n".join(lines)


def write_todos_tool(todos: List[Any]) -> str:
    """Bulk write/update tasks."""
    storage = get_todo_storage()
    new_todos = []
    for t in todos:
        if isinstance(t, dict):
            t = TodoItem(**t)
        elif not isinstance(t, TodoItem):
            t = TodoItem(**dict(t))
        kwargs: dict[str, Any] = {
            "content": t.content,
            "status": t.status,
            "active_form": t.active_form,
        }
        if t.id is not None:
            kwargs["id"] = t.id
        kwargs["parent_id"] = t.parent_id
        kwargs["depends_on"] = t.depends_on or []
        new_todos.append(Todo(**kwargs))
    storage.todos = new_todos
    counts = {"pending": 0, "in_progress": 0, "completed": 0, "blocked": 0}
    for todo in storage.todos:
        counts[todo.status] = counts.get(todo.status, 0) + 1
    parts = [f"{counts['completed']} completed"]
    if counts.get("blocked", 0) > 0:
        parts.append(f"{counts['blocked']} blocked")
    parts.append(f"{counts['in_progress']} in progress")
    parts.append(f"{counts['pending']} pending")
    return f"Updated {len(todos)} todos: {', '.join(parts)}"


def add_todo_tool(content: str, active_form: str) -> str:
    """Add a single task."""
    storage = get_todo_storage()
    new_todo = Todo(
        content=content,
        status="pending",
        active_form=active_form,
    )
    storage.todos = [*storage.todos, new_todo]
    result = f"Added todo '{content}' with ID: {new_todo.id}\n\n"
    result += _format_current_todo_list()
    return result


def update_todo_status_tool(todo_id: str, status: str) -> str:
    """Update task status by ID."""
    valid = {"pending", "in_progress", "completed", "blocked"}
    if status not in valid:
        return f"Invalid status '{status}'. Must be one of: {', '.join(sorted(valid))}"
    for todo in get_todo_storage().todos:
        if todo.id == todo_id:
            if status == "in_progress" and _is_blocked(todo):
                return f"Cannot start '{todo.content}' - it has incomplete dependencies"
            todo.status = status  # type: ignore[assignment]
            result = f"Updated todo '{todo.content}' status to '{status}'\n\n"
            result += _format_current_todo_list()
            return result
    return f"Todo with ID '{todo_id}' not found"


def remove_todo_tool(todo_id: str) -> str:
    """Delete task by ID."""
    storage = get_todo_storage()
    for i, todo in enumerate(storage.todos):
        if todo.id == todo_id:
            removed = storage.todos.pop(i)
            return f"Removed todo '{removed.content}' (ID: {todo_id})"
    return f"Todo with ID '{todo_id}' not found"


def add_subtask_tool(parent_id: str, content: str, active_form: str) -> str:
    """Create child task."""
    parent = _get_todo_by_id(parent_id)
    if not parent:
        return f"Parent todo with ID '{parent_id}' not found"
    new_todo = Todo(
        content=content,
        status="pending",
        active_form=active_form,
        parent_id=parent_id,
    )
    get_todo_storage().todos = [*get_todo_storage().todos, new_todo]
    return f"Added subtask '{content}' with ID: {new_todo.id} (parent: {parent_id})"


def _has_cycle(todo_id: str, depends_on_id: str) -> bool:
    visited: set[str] = set()

    def visit(current_id: str) -> bool:
        if current_id == todo_id:
            return True
        if current_id in visited:
            return False
        visited.add(current_id)
        todo = _get_todo_by_id(current_id)
        if todo:
            for dep_id in todo.depends_on:
                if visit(dep_id):
                    return True
        return False

    return visit(depends_on_id)


def set_dependency_tool(todo_id: str, depends_on_id: str) -> str:
    """Link tasks with dependency."""
    todo = _get_todo_by_id(todo_id)
    if not todo:
        return f"Todo with ID '{todo_id}' not found"
    dependency = _get_todo_by_id(depends_on_id)
    if not dependency:
        return f"Dependency todo with ID '{depends_on_id}' not found"
    if todo_id == depends_on_id:
        return "A todo cannot depend on itself"
    if _has_cycle(todo_id, depends_on_id):
        return "Cannot add dependency: would create a cycle"
    if depends_on_id in todo.depends_on:
        return "Dependency already exists"
    todo.depends_on = [*todo.depends_on, depends_on_id]
    if dependency.status != "completed" and todo.status not in ("completed", "blocked"):
        todo.status = "blocked"  # type: ignore[assignment]
        return (
            f"Added dependency: '{todo.content}' now depends on '{dependency.content}'. "
            "Task automatically blocked."
        )
    return f"Added dependency: '{todo.content}' now depends on '{dependency.content}'"


def get_available_tasks_tool() -> str:
    """List tasks ready to work on (no incomplete dependencies)."""
    storage = get_todo_storage()
    available = [
        t
        for t in storage.todos
        if t.status not in ("completed", "blocked") and not _is_blocked(t)
    ]
    if not available:
        return "No available tasks. All tasks are either completed or blocked."
    lines = ["Available tasks (no blocking dependencies):"]
    for i, todo in enumerate(available, 1):
        icon = _get_status_icon(todo.status)
        lines.append(f"{i}. {icon} [{todo.id}] {todo.content}")
    return "\n".join(lines)


# SimpleTool for ToolService compatibility (name, description, func, args_schema)
def create_todo_management_tools() -> List[OnyxTool]:
    """Create todo tools for ToolService (by-name lookup) and delegate agents.

    Tool names match pydantic-ai-todo: read_todos, write_todos, add_todo,
    update_todo_status, remove_todo, add_subtask, set_dependency, get_available_tasks.
    """
    return [
        OnyxTool(
            name="read_todos",
            description="Read the current todo list. Use to check status before deciding what to work on next. Set hierarchical=True for tree view with subtasks.",
            func=read_todos_tool,
            args_schema=ReadTodosInput,
        ),
        OnyxTool(
            name="write_todos",
            description="Bulk write/update the todo list. Replaces the entire list with the given items (each with content, status, active_form; optional id, parent_id, depends_on).",
            func=write_todos_tool,
            args_schema=WriteTodosInput,
        ),
        OnyxTool(
            name="add_todo",
            description="Add a single new todo item. Use to add a task without replacing existing todos. Returns the new todo's ID.",
            func=add_todo_tool,
            args_schema=AddTodoInput,
        ),
        OnyxTool(
            name="update_todo_status",
            description="Update an existing todo's status by ID. Status: pending, in_progress, completed, or blocked.",
            func=update_todo_status_tool,
            args_schema=UpdateTodoStatusInput,
        ),
        OnyxTool(
            name="remove_todo",
            description="Remove a todo from the list by ID.",
            func=remove_todo_tool,
            args_schema=RemoveTodoInput,
        ),
        OnyxTool(
            name="add_subtask",
            description="Add a subtask to an existing todo. The subtask is linked to its parent via parent_id.",
            func=add_subtask_tool,
            args_schema=AddSubtaskInput,
        ),
        OnyxTool(
            name="set_dependency",
            description="Set a dependency: one task depends on another (must be completed first). Prevents cycles.",
            func=set_dependency_tool,
            args_schema=SetDependencyInput,
        ),
        OnyxTool(
            name="get_available_tasks",
            description="Get tasks that can be worked on now (no incomplete dependencies). Blocked tasks are excluded.",
            func=get_available_tasks_tool,
            args_schema=None,
        ),
    ]
