"""Context variables and getters/setters for code changes execution context."""

from contextvars import ContextVar
from typing import Optional, Any

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# Context variable for code changes manager - provides isolation per execution context
# This ensures parallel agent runs have separate, isolated state
_code_changes_manager_ctx: ContextVar[Optional[Any]] = ContextVar(
    "_code_changes_manager_ctx", default=None
)

# Context variable for conversation_id - used to persist changes across messages
_conversation_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "_conversation_id_ctx", default=None
)

# Context variable for agent_id/agent_type - used to determine routing
_agent_id_ctx: ContextVar[Optional[str]] = ContextVar("_agent_id_ctx", default=None)

# Context variable for user_id - used for tunnel routing
_user_id_ctx: ContextVar[Optional[str]] = ContextVar("_user_id_ctx", default=None)

# Context variable for tunnel_url - used for tunnel routing (takes priority over stored state)
_tunnel_url_ctx: ContextVar[Optional[str]] = ContextVar("_tunnel_url_ctx", default=None)

# Context variable for local_mode - only True for requests from VS Code extension
_local_mode_ctx: ContextVar[bool] = ContextVar("_local_mode_ctx", default=False)

# Context variable for repository (e.g. owner/repo) - used for tunnel lookup by workspace
_repository_ctx: ContextVar[Optional[str]] = ContextVar("_repository_ctx", default=None)
# Context variable for branch - used for tunnel lookup by workspace
_branch_ctx: ContextVar[Optional[str]] = ContextVar("_branch_ctx", default=None)


def get_code_changes_manager_ctx() -> ContextVar:
    """Get the context var for the manager (used by lifecycle module)."""
    return _code_changes_manager_ctx


def _set_conversation_id(conversation_id: Optional[str]) -> None:
    """Set the conversation_id for the current execution context."""
    _conversation_id_ctx.set(conversation_id)
    logger.info(f"CodeChangesManager: Set conversation_id to {conversation_id}")


def _get_conversation_id() -> Optional[str]:
    """Get the conversation_id for the current execution context."""
    return _conversation_id_ctx.get()


def _set_agent_id(agent_id: Optional[str]) -> None:
    """Set the agent_id for the current execution context."""
    _agent_id_ctx.set(agent_id)
    logger.info(f"CodeChangesManager: Set agent_id to {agent_id}")


def _get_agent_id() -> Optional[str]:
    """Get the agent_id for the current execution context."""
    return _agent_id_ctx.get()


def _set_user_id(user_id: Optional[str]) -> None:
    """Set the user_id for the current execution context."""
    _user_id_ctx.set(user_id)


def _get_user_id() -> Optional[str]:
    """Get the user_id for the current execution context."""
    return _user_id_ctx.get()


def _set_tunnel_url(tunnel_url: Optional[str]) -> None:
    """Set the tunnel_url for the current execution context."""
    _tunnel_url_ctx.set(tunnel_url)
    logger.info(
        f"CodeChangesManager: _set_tunnel_url called with tunnel_url={tunnel_url}"
    )


def _get_tunnel_url() -> Optional[str]:
    """Get the tunnel_url for the current execution context."""
    return _tunnel_url_ctx.get()


def _set_local_mode(local_mode: bool) -> None:
    """Set local_mode for the current execution context (VS Code extension only)."""
    _local_mode_ctx.set(local_mode)


def _get_local_mode() -> bool:
    """Get local_mode for the current execution context."""
    return _local_mode_ctx.get()


def _set_repository(repository: Optional[str]) -> None:
    """Set the repository (e.g. owner/repo) for the current execution context."""
    _repository_ctx.set(repository)


def _get_repository() -> Optional[str]:
    """Get the repository for the current execution context."""
    return _repository_ctx.get()


def _set_branch(branch: Optional[str]) -> None:
    """Set the branch for the current execution context."""
    _branch_ctx.set(branch)


def _get_branch() -> Optional[str]:
    """Get the branch for the current execution context."""
    return _branch_ctx.get()
