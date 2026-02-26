"""
Pure stateless helpers for message classification and extraction.

Used by history_processor and message_compressor for inspecting
ModelMessage / ModelRequest / ModelResponse and tool call/result parts.
"""

import json
from typing import List, Optional, Set, Tuple

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)


def is_user_message(msg: ModelMessage) -> bool:
    """Check if a message is a user message (should always be preserved)."""
    if isinstance(msg, ModelRequest):
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                return True
    return False


def is_tool_call_message(msg: ModelMessage) -> bool:
    """Check if a message contains tool calls (tool_use blocks)."""
    if isinstance(msg, ModelRequest):
        for part in msg.parts:
            if hasattr(part, "__dict__"):
                part_dict = part.__dict__
                if "part_kind" in part_dict:
                    if part_dict["part_kind"] == "tool-call":
                        return True
                    elif part_dict["part_kind"] == "tool-return":
                        return False
                if "tool_call_id" in part_dict and "tool_name" in part_dict:
                    if "result" not in part_dict and "content" not in part_dict:
                        return True
                if "part" in part_dict:
                    part_obj = part_dict["part"]
                    if hasattr(part_obj, "tool_call_id") and hasattr(part_obj, "tool_name"):
                        if not hasattr(part_obj, "result") and not hasattr(part_obj, "content"):
                            return True
    elif isinstance(msg, ModelResponse):
        for part in msg.parts:
            if hasattr(part, "__dict__"):
                part_dict = part.__dict__
                if "part_kind" in part_dict and part_dict["part_kind"] == "tool-call":
                    return True
                if "tool_name" in part_dict and "tool_call_id" in part_dict:
                    if "result" not in part_dict and "content" not in part_dict:
                        return True
    return False


def is_tool_result_message(msg: ModelMessage) -> bool:
    """Check if a message contains tool results (which can be large)."""
    if isinstance(msg, ModelRequest):
        for part in msg.parts:
            if hasattr(part, "__dict__"):
                part_dict = part.__dict__
                if "part_kind" in part_dict:
                    if part_dict["part_kind"] == "tool-return":
                        return True
                    elif part_dict["part_kind"] == "tool-call":
                        return False
                if any(k in part_dict for k in ["result", "tool_name", "tool_call_id"]):
                    if "result" in part_dict or "content" in part_dict:
                        return True
    elif isinstance(msg, ModelResponse):
        for part in msg.parts:
            if hasattr(part, "__dict__"):
                part_dict = part.__dict__
                if "part_kind" in part_dict and part_dict["part_kind"] == "tool-return":
                    return True
                if "result" in part_dict or (
                    "tool_name" in part_dict and "content" in part_dict
                ):
                    return True
    return False


def is_llm_response_message(msg: ModelMessage) -> bool:
    """True if message is an LLM response (ModelResponse with non-empty TextPart)."""
    if isinstance(msg, ModelResponse):
        for part in msg.parts:
            if isinstance(part, TextPart):
                content = part.content or ""
                if content.strip():
                    return True
    return False


def extract_tool_call_ids(msg: ModelMessage) -> Set[str]:
    """Extract all tool_call_ids from a message (calls and results)."""
    tool_call_ids: Set[str] = set()
    if isinstance(msg, ModelRequest):
        for part in msg.parts:
            if hasattr(part, "__dict__"):
                part_dict = part.__dict__
                if "tool_call_id" in part_dict:
                    tid = part_dict["tool_call_id"]
                    if tid:
                        tool_call_ids.add(tid)
                if "part" in part_dict:
                    part_obj = part_dict["part"]
                    if hasattr(part_obj, "tool_call_id") and part_obj.tool_call_id:
                        tool_call_ids.add(part_obj.tool_call_id)
    elif isinstance(msg, ModelResponse):
        for part in msg.parts:
            if hasattr(part, "__dict__"):
                part_dict = part.__dict__
                if "tool_call_id" in part_dict:
                    tid = part_dict["tool_call_id"]
                    if tid:
                        tool_call_ids.add(tid)
                if "result" in part_dict:
                    result_obj = part_dict["result"]
                    if hasattr(result_obj, "tool_call_id") and result_obj.tool_call_id:
                        tool_call_ids.add(result_obj.tool_call_id)
    return tool_call_ids


def extract_tool_call_info(msg: ModelMessage) -> Optional[Tuple[str, str, str]]:
    """Extract (tool_name, args_str, tool_call_id) from a tool call message. Returns None if not a tool call."""
    if isinstance(msg, ModelRequest):
        for part in msg.parts:
            if hasattr(part, "__dict__"):
                part_dict = part.__dict__
                if "tool_name" in part_dict and "tool_call_id" in part_dict and "result" not in part_dict:
                    tool_name = part_dict.get("tool_name", "unknown_tool")
                    tool_call_id = part_dict.get("tool_call_id", "")
                    args_str = ""
                    if "args" in part_dict:
                        args = part_dict["args"]
                        args_str = args if isinstance(args, str) else (json.dumps(args) if isinstance(args, dict) else str(args))
                    return (str(tool_name), args_str, str(tool_call_id))
                if "part" in part_dict:
                    part_obj = part_dict["part"]
                    if hasattr(part_obj, "tool_name") and hasattr(part_obj, "tool_call_id") and not hasattr(part_obj, "result"):
                        tool_name = getattr(part_obj, "tool_name", "unknown_tool")
                        tool_call_id = getattr(part_obj, "tool_call_id", "")
                        args = getattr(part_obj, "args", "") if hasattr(part_obj, "args") else ""
                        args_str = args if isinstance(args, str) else (json.dumps(args) if isinstance(args, dict) else str(args))
                        return (str(tool_name), args_str, str(tool_call_id))
    elif isinstance(msg, ModelResponse):
        for part in msg.parts:
            if hasattr(part, "__dict__"):
                part_dict = part.__dict__
                if part_dict.get("part_kind") == "tool-call" or (
                    "tool_name" in part_dict and "tool_call_id" in part_dict and "result" not in part_dict and "content" not in part_dict
                ):
                    tool_name = part_dict.get("tool_name", "unknown_tool")
                    tool_call_id = part_dict.get("tool_call_id", "")
                    args_str = ""
                    if "args" in part_dict:
                        args = part_dict["args"]
                        args_str = args if isinstance(args, str) else (json.dumps(args) if isinstance(args, dict) else str(args))
                    return (str(tool_name), args_str, str(tool_call_id))
    return None


def extract_tool_result_info(msg: ModelMessage) -> Optional[Tuple[str, str, str]]:
    """Extract (tool_name, result_content, tool_call_id) from a tool result message. Returns None if not a tool result."""
    if isinstance(msg, ModelRequest):
        for part in msg.parts:
            if hasattr(part, "__dict__"):
                part_dict = part.__dict__
                if "tool_name" in part_dict or "result" in part_dict or "tool_call_id" in part_dict:
                    tool_name = part_dict.get("tool_name", "unknown_tool")
                    tool_call_id = part_dict.get("tool_call_id", "")
                    result_content = ""
                    if "result" in part_dict:
                        result = part_dict["result"]
                        result_content = str(getattr(result, "content", result) if hasattr(result, "content") else result)
                    elif "content" in part_dict:
                        result_content = str(part_dict["content"])
                    return (str(tool_name), result_content, str(tool_call_id))
    elif isinstance(msg, ModelResponse):
        for part in msg.parts:
            if hasattr(part, "__dict__"):
                part_dict = part.__dict__
                if "tool_name" in part_dict or "result" in part_dict or "tool_call_id" in part_dict:
                    tool_name = part_dict.get("tool_name", "unknown_tool")
                    tool_call_id = part_dict.get("tool_call_id", "")
                    result_content = ""
                    if "result" in part_dict:
                        result = part_dict["result"]
                        result_content = str(getattr(result, "content", result) if hasattr(result, "content") else result)
                    elif "content" in part_dict:
                        result_content = str(part_dict["content"])
                    return (str(tool_name), result_content, str(tool_call_id))
    return None


def serialize_messages_to_text(messages: List[ModelMessage]) -> str:
    """Serialize messages to text for token counting."""
    parts = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, (SystemPromptPart, UserPromptPart)):
                    content = part.content or ""
                    parts.append(content if isinstance(content, str) else str(content))
                elif hasattr(part, "__dict__"):
                    try:
                        parts.append(json.dumps(part.__dict__, default=str))
                    except Exception:
                        parts.append(str(part))
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    parts.append(part.content or "")
                elif hasattr(part, "__dict__"):
                    try:
                        parts.append(json.dumps(part.__dict__, default=str))
                    except Exception:
                        parts.append(str(part))
    return "\n".join(parts)


def extract_system_prompt_from_messages(messages: List[ModelMessage]) -> str:
    """Extract system prompt from messages (first ModelRequest with SystemPromptPart)."""
    system_prompts = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    content = part.content or ""
                    if isinstance(content, str) and content.strip():
                        system_prompts.append(content)
                    elif content:
                        system_prompts.append(str(content))
    return "\n".join(system_prompts)
