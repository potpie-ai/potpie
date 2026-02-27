"""
Truncation and validation helpers for message history.

Used by history_processor for truncating tool result content in-place
and validating tool call/result pairing (safety net).
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

from .message_helpers import (
    extract_tool_call_ids,
    is_llm_response_message,
    is_tool_call_message,
    is_tool_result_message,
    is_user_message,
)

logger = logging.getLogger(__name__)

# Defaults for trim helpers (can be overridden by callers)
MAX_TOOL_ARGS_CHARS = 500
MAX_TOOL_RESULT_LINES = 6

# Truncation message for evicted tool results
DEFAULT_TRUNCATION_MESSAGE = "[Result truncated to save tokens — re-call tool if full output is needed]"


def truncate_tool_result_message(
    msg: ModelMessage,
    truncation_text: str = DEFAULT_TRUNCATION_MESSAGE,
) -> ModelMessage:
    """Replace ToolReturnPart content in a ModelRequest with a truncation notice.
    Returns a new ModelRequest; other message types are returned unchanged (caller should only pass tool-result messages).
    """
    if not isinstance(msg, ModelRequest):
        return msg
    new_parts = []
    for part in msg.parts:
        if isinstance(part, ToolReturnPart):
            try:
                model_copy = getattr(part, "model_copy", None)
                if callable(model_copy):
                    new_part = model_copy(update={"content": truncation_text})
                    new_parts.append(new_part)
                else:
                    new_parts.append(part)
            except Exception as e:
                logger.debug("ToolReturnPart.model_copy failed, keeping part as-is: %s", e)
                new_parts.append(part)
        elif hasattr(part, "__dict__") and part.__dict__.get("part_kind") == "tool-return":
            model_copy = getattr(part, "model_copy", None)
            if callable(model_copy) and hasattr(part, "content"):
                try:
                    new_parts.append(model_copy(update={"content": truncation_text}))
                except Exception:
                    new_parts.append(part)
            else:
                new_parts.append(part)
        else:
            new_parts.append(part)
    return ModelRequest(parts=new_parts)


def trim_tool_args(args_str: str, max_chars: int = MAX_TOOL_ARGS_CHARS) -> str:
    """Trim tool arguments to a maximum character limit."""
    if not args_str or len(args_str) <= max_chars:
        return args_str
    return args_str[:max_chars] + "... [args truncated]"


def trim_tool_result_lines(
    result_content: str, max_lines: int = MAX_TOOL_RESULT_LINES
) -> str:
    """Trim tool result to first N lines."""
    if not result_content:
        return "[No result content]"
    lines = result_content.split("\n")
    if len(lines) <= max_lines:
        return result_content
    trimmed = "\n".join(lines[:max_lines])
    return trimmed + f"\n... [result truncated - {len(lines) - max_lines} more lines]"


def remove_duplicate_tool_results(
    messages: List[ModelMessage],
) -> List[ModelMessage]:
    """Remove duplicate tool results (same tool_call_id). Keeps first occurrence only."""
    seen_tool_result_ids: Set[str] = set()
    filtered_messages: List[ModelMessage] = []
    removed_parts_count = 0
    removed_messages_count = 0

    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest):
            seen_in_message: Set[str] = set()
            filtered_parts = []
            parts_removed = False

            for part in msg.parts:
                tool_call_id = None
                is_tool_result_part = False
                if hasattr(part, "__dict__"):
                    part_dict = part.__dict__
                    if part_dict.get("part_kind") == "tool-return":
                        is_tool_result_part = True
                        tool_call_id = part_dict.get("tool_call_id")
                    elif "tool_call_id" in part_dict and (
                        "result" in part_dict or "content" in part_dict
                    ):
                        is_tool_result_part = True
                        tool_call_id = part_dict.get("tool_call_id")

                if is_tool_result_part and tool_call_id:
                    if tool_call_id in seen_in_message:
                        removed_parts_count += 1
                        parts_removed = True
                        logger.warning(
                            "[Message Compressor] Removing duplicate tool_result part "
                            "within message %s: tool_call_id=%s",
                            i,
                            tool_call_id,
                        )
                        continue
                    if tool_call_id in seen_tool_result_ids:
                        removed_parts_count += 1
                        parts_removed = True
                        logger.warning(
                            "[Message Compressor] Removing duplicate tool_result part "
                            "(cross-message) at message %s: tool_call_id=%s",
                            i,
                            tool_call_id,
                        )
                        continue
                    seen_in_message.add(tool_call_id)
                    seen_tool_result_ids.add(tool_call_id)

                filtered_parts.append(part)

            if parts_removed:
                if filtered_parts:
                    msg = ModelRequest(parts=filtered_parts)
                else:
                    removed_messages_count += 1
                    logger.warning(
                        "[Message Compressor] Removing message %s entirely (all parts were duplicate tool_results)",
                        i,
                    )
                    continue

        elif is_tool_result_message(msg):
            tool_call_ids = extract_tool_call_ids(msg)
            if any(tid in seen_tool_result_ids for tid in tool_call_ids):
                removed_messages_count += 1
                logger.warning(
                    "[Message Compressor] Removing duplicate tool result message %s: %s",
                    i,
                    tool_call_ids,
                )
                continue
            seen_tool_result_ids.update(tool_call_ids)

        filtered_messages.append(msg)

    if removed_parts_count > 0 or removed_messages_count > 0:
        logger.warning(
            "[Message Compressor] Removed %s duplicate tool_result part(s) and %s duplicate message(s)",
            removed_parts_count,
            removed_messages_count,
        )

    return filtered_messages


def strip_problematic_tool_calls(
    msg: ModelResponse, problematic_tool_call_ids: Set[str]
) -> Optional[ModelResponse]:
    """Strip orphaned ToolCallParts from a ModelResponse while keeping TextPart and other parts."""
    if not isinstance(msg, ModelResponse):
        return None
    kept_parts = []
    for part in msg.parts:
        if isinstance(part, TextPart):
            kept_parts.append(part)
        elif isinstance(part, ToolCallPart):
            tool_call_id = getattr(part, "tool_call_id", None)
            if tool_call_id and tool_call_id in problematic_tool_call_ids:
                logger.debug(
                    "[Message Compressor] Stripping problematic ToolCallPart with id=%s",
                    tool_call_id,
                )
                continue
            kept_parts.append(part)
        else:
            kept_parts.append(part)
    if not kept_parts:
        return None
    return ModelResponse(parts=kept_parts, model_name=msg.model_name)


def validate_and_fix_tool_pairing(
    messages: List[ModelMessage],
) -> List[ModelMessage]:
    """Ensure every tool_use has a tool_result in the next message; remove or fix broken pairs."""
    if not messages:
        return messages

    messages = remove_duplicate_tool_results(messages)

    message_tool_call_ids: List[Set[str]] = []
    tool_call_ids_with_calls: Set[str] = set()
    tool_call_ids_with_results: Set[str] = set()

    for msg in messages:
        call_ids = extract_tool_call_ids(msg)
        message_tool_call_ids.append(call_ids)
        if is_tool_call_message(msg):
            tool_call_ids_with_calls.update(call_ids)
        elif is_tool_result_message(msg):
            tool_call_ids_with_results.update(call_ids)

    orphaned_calls = tool_call_ids_with_calls - tool_call_ids_with_results
    orphaned_results = tool_call_ids_with_results - tool_call_ids_with_calls
    tool_call_ids_without_result: Set[str] = set()

    for i, msg in enumerate(messages):
        if is_tool_call_message(msg):
            call_ids = extract_tool_call_ids(msg)
            if i + 1 < len(messages):
                next_result_ids = extract_tool_call_ids(messages[i + 1])
                missing = call_ids - next_result_ids
                tool_call_ids_without_result.update(missing)
            else:
                tool_call_ids_without_result.update(call_ids)

    problematic = orphaned_calls | orphaned_results | tool_call_ids_without_result

    if not problematic:
        return messages

    logger.warning(
        "[Message Compressor] Found problematic tool calls: %s. Orphaned calls: %s, orphaned results: %s, calls without result: %s.",
        problematic,
        orphaned_calls,
        orphaned_results,
        tool_call_ids_without_result,
    )

    messages_to_skip: Set[int] = set()

    for i, msg in enumerate(messages):
        if i in messages_to_skip:
            continue
        call_ids = message_tool_call_ids[i]
        is_tool_call = is_tool_call_message(msg)
        is_tool_result = is_tool_result_message(msg)

        if is_user_message(msg):
            continue
        if not call_ids:
            continue

        any_problematic = any(tid in problematic for tid in call_ids)
        if not any_problematic:
            continue

        if is_llm_response_message(msg):
            continue

        messages_to_skip.add(i)
        if is_tool_call and i + 1 < len(messages):
            next_msg = messages[i + 1]
            if not is_user_message(next_msg) and not is_llm_response_message(next_msg):
                messages_to_skip.add(i + 1)
        if is_tool_result and i > 0:
            prev = messages[i - 1]
            if not is_user_message(prev) and is_tool_call_message(prev) and not is_llm_response_message(prev):
                messages_to_skip.add(i - 1)

    ids_whose_tool_results_must_skip: Set[str] = set()
    validated_messages: List[ModelMessage] = []

    for i, msg in enumerate(messages):
        if i in messages_to_skip:
            continue

        if is_tool_result_message(msg):
            call_ids = message_tool_call_ids[i]
            overlap = call_ids & ids_whose_tool_results_must_skip
            if overlap:
                ids_whose_tool_results_must_skip -= call_ids
                continue

        if isinstance(msg, ModelResponse) and is_llm_response_message(msg):
            call_ids = message_tool_call_ids[i]
            if any(tid in problematic for tid in call_ids):
                stripped = strip_problematic_tool_calls(msg, problematic)
                if stripped:
                    validated_messages.append(stripped)
                ids_whose_tool_results_must_skip |= call_ids & problematic
                continue

        validated_messages.append(msg)

    if not validated_messages and messages:
        validated_messages = [m for m in messages if is_user_message(m)]
    if not validated_messages and messages:
        validated_messages = [messages[0]]

    # Re-validate iteratively (up to 10 times) in case removal created new orphans
    for _ in range(10):
        prev_len = len(validated_messages)
        validated_messages = _one_validation_pass(validated_messages)
        if len(validated_messages) == prev_len:
            break

    if not validated_messages and messages:
        validated_messages = [m for m in messages if is_user_message(m)] or [messages[0]]

    return validated_messages


def _one_validation_pass(
    messages: List[ModelMessage],
) -> List[ModelMessage]:
    """Single pass: remove messages with problematic tool call/result pairing."""
    if not messages:
        return messages
    call_ids_per_msg = [extract_tool_call_ids(m) for m in messages]
    calls_set: Set[str] = set()
    results_set: Set[str] = set()
    for i, msg in enumerate(messages):
        if is_tool_call_message(msg):
            calls_set.update(call_ids_per_msg[i])
        elif is_tool_result_message(msg):
            results_set.update(call_ids_per_msg[i])

    without_result: Set[str] = set()
    for i, msg in enumerate(messages):
        if is_tool_call_message(msg):
            cids = call_ids_per_msg[i]
            if i + 1 < len(messages):
                next_ids = call_ids_per_msg[i + 1]
                without_result.update(cids - next_ids)
            else:
                without_result.update(cids)

    problematic = (calls_set - results_set) | (results_set - calls_set) | without_result
    if not problematic:
        return messages

    to_skip: Set[int] = set()
    for i, msg in enumerate(messages):
        if not call_ids_per_msg[i]:
            continue
        if is_llm_response_message(msg):
            continue
        if any(tid in problematic for tid in call_ids_per_msg[i]):
            to_skip.add(i)
            if is_tool_call_message(msg) and i + 1 < len(messages):
                if not is_llm_response_message(messages[i + 1]):
                    to_skip.add(i + 1)
            if is_tool_result_message(msg) and i > 0 and is_tool_call_message(messages[i - 1]):
                if not is_llm_response_message(messages[i - 1]):
                    to_skip.add(i - 1)

    return [m for i, m in enumerate(messages) if i not in to_skip]
