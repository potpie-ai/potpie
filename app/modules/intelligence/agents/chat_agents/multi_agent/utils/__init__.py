"""Utility modules for multi-agent system"""

from .delegation_utils import (
    is_delegation_tool,
    extract_agent_type_from_delegation_tool,
    extract_task_result_from_response,
    create_delegation_prompt,
    format_delegation_error,
    create_delegation_cache_key,
)
from .tool_utils import (
    handle_exception,
    create_tool_call_response,
    create_tool_result_response,
    wrap_structured_tools,
    deduplicate_tools_by_name,
    create_error_response,
)
from .message_history_utils import (
    validate_and_fix_message_history,
    prepare_multimodal_message_history,
)
from .context_utils import (
    create_project_context_info,
    create_supervisor_task_description,
)
from .multimodal_utils import (
    create_multimodal_user_content,
)

__all__ = [
    # Delegation utils
    "is_delegation_tool",
    "extract_agent_type_from_delegation_tool",
    "extract_task_result_from_response",
    "create_delegation_prompt",
    "format_delegation_error",
    "create_delegation_cache_key",
    # Tool utils
    "handle_exception",
    "create_tool_call_response",
    "create_tool_result_response",
    "wrap_structured_tools",
    "deduplicate_tools_by_name",
    "create_error_response",
    # Message history utils
    "validate_and_fix_message_history",
    "prepare_multimodal_message_history",
    # Context utils
    "create_project_context_info",
    "create_supervisor_task_description",
    # Multimodal utils
    "create_multimodal_user_content",
]
