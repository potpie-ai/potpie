"""Per-execution-context state for sandbox-backed agent tools.

The agent run (Celery task or HTTP handler) sets these contextvars at the
start; sandbox tools read them when they need to materialise a workspace for
``(user_id, project_id, branch)``. The pattern mirrors
``code_changes_manager/context.py`` but is intentionally independent so the
sandbox tool group has no import-edge into the legacy stack.

Once CodeChangesManager is deleted (plan phase 8) the agent layer should set
both groups of contextvars from one place; until then we duplicate.
"""

from __future__ import annotations

from contextvars import ContextVar


_user_id_ctx: ContextVar[str | None] = ContextVar(
    "_sandbox_user_id_ctx", default=None
)
_conversation_id_ctx: ContextVar[str | None] = ContextVar(
    "_sandbox_conversation_id_ctx", default=None
)
_branch_ctx: ContextVar[str | None] = ContextVar(
    "_sandbox_branch_ctx", default=None
)
_auth_token_ctx: ContextVar[str | None] = ContextVar(
    "_sandbox_auth_token_ctx", default=None
)
_local_mode_ctx: ContextVar[bool] = ContextVar(
    "_sandbox_local_mode_ctx", default=False
)
_workspace_announced_ctx: ContextVar[bool] = ContextVar(
    "_sandbox_workspace_announced_ctx", default=False
)


def set_run_context(
    *,
    user_id: str | None = None,
    conversation_id: str | None = None,
    branch: str | None = None,
    auth_token: str | None = None,
    local_mode: bool | None = None,
) -> None:
    """Bulk-set sandbox run context. Pass only what you have; ``None`` is a no-op.

    Always resets the per-run "workspace announced" flag so the first sandbox
    tool call in a fresh run surfaces the setup banner again.
    """
    if user_id is not None:
        _user_id_ctx.set(user_id)
    if conversation_id is not None:
        _conversation_id_ctx.set(conversation_id)
    if branch is not None:
        _branch_ctx.set(branch)
    if auth_token is not None:
        _auth_token_ctx.set(auth_token)
    if local_mode is not None:
        _local_mode_ctx.set(local_mode)
    _workspace_announced_ctx.set(False)


def get_user_id() -> str | None:
    return _user_id_ctx.get()


def get_conversation_id() -> str | None:
    return _conversation_id_ctx.get()


def get_branch() -> str | None:
    return _branch_ctx.get()


def get_auth_token() -> str | None:
    return _auth_token_ctx.get()


def get_local_mode() -> bool:
    return _local_mode_ctx.get()


def is_workspace_announced() -> bool:
    return _workspace_announced_ctx.get()


def mark_workspace_announced() -> None:
    _workspace_announced_ctx.set(True)
