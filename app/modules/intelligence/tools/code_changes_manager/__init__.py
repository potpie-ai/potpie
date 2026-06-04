"""
Code Changes Manager Tool for Agent State Management

This tool allows agents to manage code changes in Redis, reducing token usage
by storing code modifications separately from response text. Changes are tracked
per-file and can be searched, retrieved, and serialized for persistence.

Changes persist across messages within a conversation (keyed by conversation_id).
"""

from .manager import CodeChangesManager
from .constants import (
    CODE_CHANGES_KEY_PREFIX,
    CODE_CHANGES_TTL_SECONDS,
    MAX_FILE_SIZE_BYTES,
    DB_QUERY_TIMEOUT,
    DB_SESSION_CREATE_TIMEOUT,
    MEMORY_PRESSURE_THRESHOLD,
)
from .models import ChangeType, FileChange
from .context import (
    _set_conversation_id,
    _get_conversation_id,
    _set_agent_id,
    _get_agent_id,
    _set_user_id,
    _get_user_id,
    _set_tunnel_url,
    _get_tunnel_url,
    _set_local_mode,
    _get_local_mode,
    _set_repository,
    _get_repository,
    _set_branch,
    _get_branch,
)
from .lifecycle import (
    _get_code_changes_manager,
    _init_code_changes_manager,
    _reset_code_changes_manager,
    _get_project_id_from_conversation_id,
)
from .tools import (
    create_code_changes_management_tools,
    CODE_CHANGES_TOOLS_EXCLUDE_IN_LOCAL,
    CODE_CHANGES_TOOLS_EXCLUDE_WHEN_NON_LOCAL,
    SimpleTool,
)

__all__ = [
    "CodeChangesManager",
    "CODE_CHANGES_KEY_PREFIX",
    "CODE_CHANGES_TTL_SECONDS",
    "MAX_FILE_SIZE_BYTES",
    "DB_QUERY_TIMEOUT",
    "DB_SESSION_CREATE_TIMEOUT",
    "MEMORY_PRESSURE_THRESHOLD",
    "ChangeType",
    "FileChange",
    "_set_conversation_id",
    "_get_conversation_id",
    "_set_agent_id",
    "_get_agent_id",
    "_set_user_id",
    "_get_user_id",
    "_set_tunnel_url",
    "_get_tunnel_url",
    "_set_local_mode",
    "_get_local_mode",
    "_set_repository",
    "_get_repository",
    "_set_branch",
    "_get_branch",
    "_get_code_changes_manager",
    "_init_code_changes_manager",
    "_reset_code_changes_manager",
    "_get_project_id_from_conversation_id",
    "create_code_changes_management_tools",
    "CODE_CHANGES_TOOLS_EXCLUDE_IN_LOCAL",
    "CODE_CHANGES_TOOLS_EXCLUDE_WHEN_NON_LOCAL",
    "SimpleTool",
]
