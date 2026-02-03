"""Tool utility functions for multi-agent system"""

import functools
import json
from typing import List, Sequence, Any

from pydantic_ai import Tool
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent

from .delegation_utils import (
    is_delegation_tool,
    extract_agent_type_from_delegation_tool,
)
from app.modules.intelligence.agents.chat_agent import (
    ToolCallEventType,
    ToolCallResponse,
    ChatAgentResponse,
)
from ...tool_helpers import (
    get_tool_call_info_content,
    get_tool_response_message,
    get_tool_result_info_content,
    get_tool_run_message,
    get_delegation_call_message,
    get_delegation_response_message,
    get_delegation_info_content,
    get_delegation_result_content,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


def _repair_truncated_tool_args_json(raw: str) -> dict | None:
    """Attempt to repair truncated JSON from streamed tool call args. Returns parsed dict or None."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s or s == "{}":
        return {} if s == "{}" else None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # EOF while parsing usually means the string was cut off mid-object or mid-string
    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")
    last = s.rstrip()[-1] if s.rstrip() else ""
    # If we're inside an unclosed string (last char is not ", }, ], ,, :), close it first
    suffix = ""
    if last and last not in ('"', "}", "]", ",", ":"):
        suffix = '"'
    repaired = s.rstrip() + suffix + "]" * open_brackets + "}" * open_braces
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    # Try without closing string (e.g. ended right after "...)
    repaired2 = s.rstrip() + "}" * open_braces + "]" * open_brackets
    try:
        return json.loads(repaired2)
    except json.JSONDecodeError:
        return None


def handle_exception(tool_func):
    @functools.wraps(tool_func)
    def wrapper(*args, **kwargs):
        try:
            return tool_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in tool function: {e}")
            return "An internal error occurred. Please try again later."

    return wrapper


def create_tool_call_response(event: FunctionToolCallEvent) -> ToolCallResponse:
    """Create appropriate tool call response for regular or delegation tools"""
    tool_name = event.part.tool_name

    # Safely parse tool arguments with error handling for malformed/truncated JSON
    try:
        args_dict = event.part.args_as_dict()
    except (ValueError, json.JSONDecodeError) as json_error:
        raw_args = getattr(event.part, "args", "N/A")
        raw_str = str(raw_args) if raw_args != "N/A" else ""

        # Try to repair truncated JSON (common when tool args are streamed and cut off)
        repaired = _repair_truncated_tool_args_json(raw_str)
        if repaired is not None:
            args_dict = repaired
            try:
                setattr(event.part, "args", json.dumps(repaired))
            except Exception as sanitize_error:
                logger.warning(
                    "Unable to sanitize repaired tool call arguments for '%s': %s",
                    tool_name,
                    sanitize_error,
                )
            logger.info(
                "Repaired truncated JSON for tool call '%s' (recovered %d keys)",
                tool_name,
                len(args_dict),
            )
        else:
            # Repair failed; use empty dict and log
            logger.error(
                "JSON parsing error in tool call '%s': %s. "
                "Tool args (raw, first 300 chars): %s. "
                "This may cause issues when pydantic_ai tries to serialize the message history.",
                tool_name,
                json_error,
                raw_str[:300],
            )
            args_dict = {}
            try:
                setattr(event.part, "args", "{}")
            except Exception as sanitize_error:
                logger.warning(
                    "Unable to sanitize malformed tool call arguments for '%s': %s",
                    tool_name,
                    sanitize_error,
                )

    if is_delegation_tool(tool_name):
        agent_type = extract_agent_type_from_delegation_tool(tool_name)
        task_description = args_dict.get("task_description", "")
        context = args_dict.get("context", "")

        return ToolCallResponse(
            call_id=event.part.tool_call_id or "",
            event_type=ToolCallEventType.DELEGATION_CALL,
            tool_name=tool_name,
            tool_response=get_delegation_call_message(agent_type),
            tool_call_details={
                "summary": get_delegation_info_content(
                    agent_type, task_description, context
                )
            },
        )
    else:
        return ToolCallResponse(
            call_id=event.part.tool_call_id or "",
            event_type=ToolCallEventType.CALL,
            tool_name=tool_name,
            tool_response=get_tool_run_message(tool_name, args_dict),
            tool_call_details={
                "summary": get_tool_call_info_content(tool_name, args_dict)
            },
        )


def create_tool_result_response(event: FunctionToolResultEvent) -> ToolCallResponse:
    """Create appropriate tool result response for regular or delegation tools"""
    tool_name = event.result.tool_name or "unknown tool"

    if is_delegation_tool(tool_name):
        agent_type = extract_agent_type_from_delegation_tool(tool_name)
        result_content = str(event.result.content) if event.result.content else ""

        return ToolCallResponse(
            call_id=event.result.tool_call_id or "",
            event_type=ToolCallEventType.DELEGATION_RESULT,
            tool_name=tool_name,
            tool_response=get_delegation_response_message(agent_type),
            tool_call_details={
                "summary": get_delegation_result_content(agent_type, result_content)
            },
            is_complete=True,  # Explicitly mark delegation results as complete
        )
    else:
        return ToolCallResponse(
            call_id=event.result.tool_call_id or "",
            event_type=ToolCallEventType.RESULT,
            tool_name=tool_name,
            tool_response=get_tool_response_message(tool_name),
            tool_call_details={
                "summary": get_tool_result_info_content(tool_name, event.result.content)
            },
        )


def wrap_structured_tools(tools: Sequence[Any]) -> List[Tool]:
    """Convert tool instances (StructuredTool or similar) to PydanticAI Tool instances"""
    return [
        Tool(
            name=tool.name,
            description=tool.description,
            function=handle_exception(tool.func),  # type: ignore
        )
        for tool in tools
    ]


def deduplicate_tools_by_name(tools: List[Tool]) -> List[Tool]:
    """Deduplicate tools by name, keeping the first occurrence of each tool name.

    Note: Duplicates are expected when the multi-agent system combines tools from
    multiple sources (agent-provided tools + built-in tools). This is by design.
    """
    seen_names = set()
    deduplicated = []
    duplicate_count = 0
    for tool in tools:
        if tool.name not in seen_names:
            seen_names.add(tool.name)
            deduplicated.append(tool)
        else:
            duplicate_count += 1

    # Log summary at debug level instead of individual warnings
    if duplicate_count > 0:
        logger.debug(
            f"Deduplicated {duplicate_count} duplicate tool(s), kept {len(deduplicated)} unique tools"
        )
    return deduplicated


def create_error_response(message: str) -> ChatAgentResponse:
    """Create a standardized error response"""
    return ChatAgentResponse(
        response=f"\n\n{message}\n\n",
        tool_calls=[],
        citations=[],
    )
