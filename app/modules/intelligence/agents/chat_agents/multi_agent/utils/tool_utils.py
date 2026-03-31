"""Tool utility functions for multi-agent system"""

import copy
import functools
import hashlib
import inspect
import json
import re
from typing import Any, List, Sequence

from pydantic import BaseModel
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

# Max chars of tool result content streamed to browser (prevents OOM on large codebases)
_MAX_TOOL_RESULT_STREAM_CHARS = 10_000


def truncate_result_content(content: str) -> tuple[str, bool, int | None]:
    """Truncate raw tool result content to browser-safe length.
    Returns: (content, is_truncated, original_length_or_None)
    """
    if not content or len(content) <= _MAX_TOOL_RESULT_STREAM_CHARS:
        return content, False, None
    original_length = len(content)
    truncated = (
        content[:_MAX_TOOL_RESULT_STREAM_CHARS]
        + f"\n... [truncated — {original_length:,} chars total, showing first {_MAX_TOOL_RESULT_STREAM_CHARS:,}]"
    )
    logger.info(
        "Tool result truncated for browser stream: %d → %d chars",
        original_length,
        _MAX_TOOL_RESULT_STREAM_CHARS,
    )
    return truncated, True, original_length


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


def _safe_parse_tool_args(
    event: FunctionToolCallEvent, tool_name: str
) -> dict[str, Any]:
    """Parse streamed tool args defensively and sanitize malformed payloads."""
    try:
        return event.part.args_as_dict()
    except (ValueError, json.JSONDecodeError) as json_error:
        raw_args = getattr(event.part, "args", "N/A")
        raw_str = str(raw_args) if raw_args != "N/A" else ""
        repaired = _repair_truncated_tool_args_json(raw_str)
        if repaired is not None:
            if not isinstance(repaired, dict):
                logger.warning(
                    "Repaired JSON for tool call '%s' is not a dict (type=%s); normalizing to {}",
                    tool_name,
                    type(repaired).__name__,
                )
                repaired = {}
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
                len(repaired),
            )
            return repaired

        raw_digest = hashlib.sha256(raw_str.encode()).hexdigest()
        logger.error(
            "JSON parsing error in tool call '%s': %s. "
            "Tool args payload size=%d bytes, sha256=%s. "
            "This may cause issues when pydantic_ai tries to serialize the message history.",
            tool_name,
            json_error,
            len(raw_str),
            raw_digest,
        )
        try:
            setattr(event.part, "args", "{}")
        except Exception as sanitize_error:
            logger.warning(
                "Unable to sanitize malformed tool call arguments for '%s': %s",
                tool_name,
                sanitize_error,
            )
        return {}


def handle_exception(tool_func):
    # After _adapt_func_for_from_schema the wrapper is always async.
    # Guard here as well in case handle_exception is called on a raw sync func.
    if inspect.iscoroutinefunction(tool_func):

        @functools.wraps(tool_func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await tool_func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Exception in tool function: {e}")
                return "An internal error occurred. Please try again later."

        return async_wrapper
    else:

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
    args_dict = _safe_parse_tool_args(event, tool_name)

    command_tools = {"search_bash", "bash_command", "execute_terminal_command"}
    if tool_name in command_tools:
        command = str(args_dict.get("command", "") or "").strip()
        return ToolCallResponse(
            call_id=event.part.tool_call_id or "",
            event_type=ToolCallEventType.CALL,
            tool_name=tool_name,
            tool_response=command or get_tool_run_message(tool_name, args_dict),
            tool_call_details={"command": command} if command else {},
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
        full_result_content = str(event.result.content) if event.result.content else ""

        # Detect if the event already carries truncation metadata to avoid double-truncating
        result_is_truncated = getattr(event.result, "is_truncated", None)
        result_original_length = getattr(event.result, "original_length", None)
        already_truncated = result_is_truncated or (
            result_original_length is not None
            and result_original_length > len(full_result_content)
        )

        if already_truncated:
            truncated_result_content = full_result_content
            is_truncated = bool(result_is_truncated)
            original_length = result_original_length
        else:
            (
                truncated_result_content,
                is_truncated,
                original_length,
            ) = truncate_result_content(full_result_content)

        return ToolCallResponse(
            call_id=event.result.tool_call_id or "",
            event_type=ToolCallEventType.DELEGATION_RESULT,
            tool_name=tool_name,
            tool_response=get_delegation_response_message(agent_type),
            tool_call_details={
                "summary": get_delegation_result_content(
                    agent_type, full_result_content
                ),
                "content": truncated_result_content,
            },
            is_complete=True,
            is_truncated=is_truncated,
            original_length=original_length,
        )
    else:
        full_raw = str(event.result.content) if event.result.content else ""
        truncated_raw, is_truncated, original_length = truncate_result_content(full_raw)

        return ToolCallResponse(
            call_id=event.result.tool_call_id or "",
            event_type=ToolCallEventType.RESULT,
            tool_name=tool_name,
            tool_response=get_tool_response_message(tool_name, result=full_raw),
            tool_call_details={
                "summary": get_tool_result_info_content(tool_name, full_raw),
                "content": truncated_raw,
            },
            is_truncated=is_truncated,
            original_length=original_length,
        )


# OpenAI-compatible APIs require tool function names to match ^[a-zA-Z0-9_-]+$
_TOOL_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_-]+")


def sanitize_tool_name_for_api(name: str) -> str:
    """Sanitize a tool name so it matches OpenAI-style API requirement: ^[a-zA-Z0-9_-]+$.
    Returns lowercase so tool names match registry keys (e.g. web_search_tool) and models
    that normalize tool names; avoids mismatch when the model returns tool_calls."""
    if not name:
        return "unnamed_tool"
    sanitized = _TOOL_NAME_PATTERN.sub("_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    out = (sanitized or "unnamed_tool").lower()
    return out


def _inline_json_schema_refs(
    schema: dict[str, Any], defs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return a deep copy of the schema with all $ref inlined so every node has a 'type' key.

    Some APIs (e.g. OpenAI) require every schema node to have a 'type' and do not resolve $ref.
    """
    resolved_defs: dict[str, Any] = schema.get("$defs", {}) if defs is None else defs
    schema = copy.deepcopy(schema)

    def resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref = obj["$ref"]
                key = ref.split("/")[-1]
                return resolve(resolved_defs.get(key, obj))
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(x) for x in obj]
        return obj

    out = resolve(schema)
    out.pop("$defs", None)
    return out


def _get_tool_args_schema(tool: Any) -> dict[str, Any] | None:
    """Get JSON schema for a tool's args if it has an args_schema (Pydantic model or dict)."""
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return None
    if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
        schema_fn = getattr(args_schema, "model_json_schema", None) or getattr(
            args_schema, "schema", None
        )
        if schema_fn:
            return schema_fn()
    if isinstance(args_schema, dict):
        return args_schema
    return None


def _adapt_func_for_from_schema(tool: Any) -> Any:
    """Adapt the tool's func so Tool.from_schema's **kwargs are passed correctly.

    Tool.from_schema calls the function with **kwargs (schema property names). Two cases:

    1) Single Pydantic model arg: func(input_data: SomeInput). Wrap to build the model
       from kwargs and call func(model), so both styles work.

    2) Multiple params matching args_schema: func(project_id, paths, ...). Wrap to
       validate kwargs via the args_schema and call func(**model.model_dump()). This
       ensures required fields are validated (clear errors instead of "missing N
       required positional arguments") when the model sends empty or malformed args.

    The returned wrapper is always async. For sync tool funcs it uses asyncio.to_thread
    which explicitly copies the current contextvars before dispatching the thread.
    Python 3.13's loop.run_in_executor does NOT copy contextvars, so relying on
    pydantic-ai's default run_in_executor path loses user_id / tunnel_url context.
    """
    import asyncio as _asyncio

    raw_schema = getattr(tool, "args_schema", None)
    if not (isinstance(raw_schema, type) and issubclass(raw_schema, BaseModel)):
        return tool.func
    try:
        sig = inspect.signature(tool.func)
    except (TypeError, ValueError):
        return tool.func
    params = [p for p in sig.parameters.values() if p.name != "self"]
    if len(params) == 1:
        param = params[0]
        annotation = param.annotation
        if (
            annotation is not inspect.Parameter.empty
            and isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
        ):
            model_cls = annotation
            _is_async = inspect.iscoroutinefunction(tool.func)
            _func = tool.func

            if _is_async:

                async def _single_async(**kwargs: Any) -> Any:
                    return await _func(model_cls(**kwargs))

                return _single_async
            else:

                async def _single_sync(**kwargs: Any) -> Any:
                    return await _asyncio.to_thread(_func, model_cls(**kwargs))

                return _single_sync

    if len(params) >= 2:
        model_cls = raw_schema
        _is_async_multi = inspect.iscoroutinefunction(tool.func)
        _func_multi = tool.func

        if _is_async_multi:

            async def _multi_async(**kwargs: Any) -> Any:
                validated = model_cls(**kwargs)
                return await _func_multi(**validated.model_dump())

            return _multi_async
        else:

            async def _multi_sync(**kwargs: Any) -> Any:
                validated = model_cls(**kwargs)
                return await _asyncio.to_thread(
                    lambda: _func_multi(**validated.model_dump())
                )

            return _multi_sync

    return tool.func


def wrap_structured_tools(tools: Sequence[Any]) -> List[Tool]:
    """Convert tool instances (StructuredTool or similar) to PydanticAI Tool instances.
    Tool names are sanitized to match API requirement ^[a-zA-Z0-9_-]+$.
    When a tool has args_schema (e.g. SimpleTool with a Pydantic model), the schema is
    inlined so APIs that require a 'type' key in every node (e.g. OpenAI) accept it.

    Tools whose function takes a single Pydantic model argument (e.g. input_data: XInput)
    are adapted so that Tool.from_schema's **kwargs are converted to that model before calling.
    """
    result: List[Tool] = []
    for tool in tools:
        name = sanitize_tool_name_for_api(tool.name)
        description = tool.description
        func = _adapt_func_for_from_schema(tool)
        func = handle_exception(func)  # type: ignore[arg-type]
        args_schema = _get_tool_args_schema(tool)
        if args_schema is not None:
            # Inline $ref so APIs that require a 'type' key in every node (e.g. OpenAI) accept the schema
            json_schema = (
                _inline_json_schema_refs(args_schema)
                if args_schema.get("$defs")
                else args_schema
            )
            result.append(
                Tool.from_schema(
                    function=func,
                    name=name,
                    description=description,
                    json_schema=json_schema,
                )
            )
        else:
            result.append(
                Tool(
                    name=name,
                    description=description,
                    function=func,
                )
            )
    return result


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
