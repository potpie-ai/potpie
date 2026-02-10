"""Tool utility functions for multi-agent system"""

import copy
import functools
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
            tool_response=get_tool_response_message(
                tool_name, result=event.result.content
            ),
            tool_call_details={
                "summary": get_tool_result_info_content(tool_name, event.result.content)
            },
        )


# OpenAI-compatible APIs require tool function names to match ^[a-zA-Z0-9_-]+$
_TOOL_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_-]+")


def sanitize_tool_name_for_api(name: str) -> str:
    """Sanitize a tool name so it matches OpenAI-style API requirement: ^[a-zA-Z0-9_-]+$"""
    if not name:
        return "unnamed_tool"
    sanitized = _TOOL_NAME_PATTERN.sub("_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "unnamed_tool"


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
    """
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

            def single_arg_wrapper(**kwargs: Any) -> Any:
                return tool.func(model_cls(**kwargs))

            return single_arg_wrapper
    if len(params) >= 2:
        # Multiple params (e.g. project_id, paths, with_line_numbers): validate
        # kwargs with args_schema and call func(**validated) so missing/empty
        # args produce a clear ValidationError instead of "missing N required positional arguments".
        model_cls = raw_schema

        def multi_arg_wrapper(**kwargs: Any) -> Any:
            validated = model_cls(**kwargs)
            return tool.func(**validated.model_dump())

        return multi_arg_wrapper
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
