"""Message history utility functions for multi-agent system"""

from typing import List, Optional, Any
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
)

from app.modules.intelligence.agents.chat_agent import ChatContext
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


def validate_and_fix_message_history(
    messages: List[ModelMessage],
) -> List[ModelMessage]:
    """Validate message history and ensure tool calls/results are properly paired.

    Anthropic API requires:
    - Every tool_use block must have a tool_result block in the IMMEDIATELY NEXT message
    - Removes orphaned tool results (results without corresponding calls)
    - Removes orphaned tool calls (calls without corresponding results)
    - Removes tool calls that don't have results in the next message

    This prevents "tool_result without tool_use" and "tool_use without tool_result" errors.
    """
    if not messages:
        return messages

    def _extract_tool_call_ids(msg: ModelMessage) -> set:
        """Extract all tool_call_ids from a message."""
        ids = set()
        parts_to_check = []
        if isinstance(msg, ModelRequest):
            parts_to_check = msg.parts
        elif isinstance(msg, ModelResponse):
            parts_to_check = msg.parts

        for part in parts_to_check:
            if hasattr(part, "__dict__"):
                part_dict = part.__dict__
                if "tool_call_id" in part_dict and part_dict["tool_call_id"]:
                    ids.add(part_dict["tool_call_id"])
                if "part" in part_dict:
                    part_obj = part_dict["part"]
                    if hasattr(part_obj, "tool_call_id") and part_obj.tool_call_id:
                        ids.add(part_obj.tool_call_id)
                if "result" in part_dict:
                    result_obj = part_dict["result"]
                    if hasattr(result_obj, "tool_call_id") and result_obj.tool_call_id:
                        ids.add(result_obj.tool_call_id)
        return ids

    def _is_tool_call_msg(msg: ModelMessage) -> bool:
        """Check if message contains tool calls (not results)."""
        parts_to_check = []
        if isinstance(msg, ModelRequest):
            parts_to_check = msg.parts
        elif isinstance(msg, ModelResponse):
            parts_to_check = msg.parts

        for part in parts_to_check:
            if hasattr(part, "__dict__"):
                part_dict = part.__dict__
                # Check part_kind first
                if part_dict.get("part_kind") == "tool-call":
                    return True
                # Has tool_call_id and tool_name but no result
                if "tool_call_id" in part_dict and "tool_name" in part_dict:
                    if "result" not in part_dict and "content" not in part_dict:
                        return True
                if "part" in part_dict:
                    part_obj = part_dict["part"]
                    if hasattr(part_obj, "tool_call_id") and hasattr(
                        part_obj, "tool_name"
                    ):
                        if not hasattr(part_obj, "result"):
                            return True
        return False

    def _is_tool_result_msg(msg: ModelMessage) -> bool:
        """Check if message contains tool results."""
        parts_to_check = []
        if isinstance(msg, ModelRequest):
            parts_to_check = msg.parts
        elif isinstance(msg, ModelResponse):
            parts_to_check = msg.parts

        for part in parts_to_check:
            if hasattr(part, "__dict__"):
                part_dict = part.__dict__
                # Check part_kind first
                if part_dict.get("part_kind") == "tool-return":
                    return True
                # Has result or content with tool_call_id
                if "result" in part_dict or (
                    "tool_call_id" in part_dict and "content" in part_dict
                ):
                    return True
        return False

    # Track tool_call_ids that have been used (tool calls)
    tool_call_ids_with_calls: set = set()
    # Track tool_call_ids that have results
    tool_call_ids_with_results: set = set()
    # Track message metadata
    message_tool_call_ids: list = []

    # First pass: identify all tool calls and results
    for i, msg in enumerate(messages):
        call_ids = _extract_tool_call_ids(msg)
        message_tool_call_ids.append(call_ids)

        if _is_tool_call_msg(msg):
            tool_call_ids_with_calls.update(call_ids)
        elif _is_tool_result_msg(msg):
            tool_call_ids_with_results.update(call_ids)

    # Find orphaned tool calls and results
    orphaned_calls = tool_call_ids_with_calls - tool_call_ids_with_results
    orphaned_results = tool_call_ids_with_results - tool_call_ids_with_calls

    # CRITICAL: Find tool calls that don't have results in the IMMEDIATELY NEXT message
    tool_calls_without_next_result: set = set()
    for i, msg in enumerate(messages):
        if _is_tool_call_msg(msg):
            call_ids = message_tool_call_ids[i]
            if i + 1 < len(messages):
                next_result_ids = message_tool_call_ids[i + 1]
                # All call_ids must have results in next message
                missing = call_ids - next_result_ids
                if missing:
                    tool_calls_without_next_result.update(missing)
            else:
                # No next message
                tool_calls_without_next_result.update(call_ids)

    # Combine all problematic tool_call_ids
    problematic_ids = orphaned_calls | orphaned_results | tool_calls_without_next_result

    if problematic_ids:
        logger.warning(
            f"[Message History Validation] Found {len(problematic_ids)} problematic tool_call_ids: "
            f"orphaned_calls={orphaned_calls}, orphaned_results={orphaned_results}, "
            f"calls_without_next_result={tool_calls_without_next_result}. Removing to prevent API errors."
        )

        # Second pass: identify messages to remove (including their pairs)
        messages_to_skip: set = set()

        for i, msg in enumerate(messages):
            call_ids = message_tool_call_ids[i]
            is_tool_call = _is_tool_call_msg(msg)
            is_tool_result = _is_tool_result_msg(msg)

            if not call_ids:
                continue

            # CRITICAL: Remove if ANY tool_call_id is problematic
            if any(tid in problematic_ids for tid in call_ids):
                messages_to_skip.add(i)

                # Also remove paired messages
                if is_tool_call and i + 1 < len(messages):
                    messages_to_skip.add(i + 1)
                if is_tool_result and i > 0 and _is_tool_call_msg(messages[i - 1]):
                    messages_to_skip.add(i - 1)

        # Build filtered messages
        filtered_messages = [
            msg for i, msg in enumerate(messages) if i not in messages_to_skip
        ]

        # Safety check
        if not filtered_messages and messages:
            logger.error(
                "[Message History Validation] All messages removed! Keeping original."
            )
            return messages

        return filtered_messages

    return messages


async def prepare_multimodal_message_history(
    ctx: ChatContext,
    history_processor: Any,
) -> List[ModelMessage]:
    """Prepare message history with multimodal support.

    CRITICAL: This method now prioritizes using compressed message history from previous runs.
    It retrieves compressed messages from the history processor's internal storage.
    Otherwise, it falls back to rebuilding from ctx.history (text strings).

    This ensures that compression benefits are preserved across multiple agent runs within
    the same execution context.
    """
    # Try to get compressed history from the history processor
    # The processor stores compressed messages keyed by run_id from RunContext
    # We need to check if we have a stored run_id from a previous run
    # For now, we'll use a simple approach: check if processor has any stored history
    # and use the most recent one (since we're in the same execution context)

    # Access the processor instance from the history processor function
    processor = getattr(history_processor, "processor", None)
    if processor:
        # Get all stored compressed histories (there should be at most one per run)
        stored_keys = list(processor._last_compressed_output.keys())
        if stored_keys:
            # Use the most recent key (last one added)
            latest_key = stored_keys[-1]
            compressed_history = processor.get_compressed_history(latest_key)
            if compressed_history:
                # Validate the compressed history before using it
                return validate_and_fix_message_history(compressed_history)

    # Fallback: rebuild from ctx.history if no compressed history available
    history_messages = []

    # Limit history to prevent token bloat (max 8 messages or ~50k tokens estimated)
    # This prevents "prompt too long" errors and reduces chance of duplicate tool_result issues
    max_history_messages = 8
    limited_history = (
        ctx.history[-max_history_messages:]
        if len(ctx.history) > max_history_messages
        else ctx.history
    )

    if len(ctx.history) > max_history_messages:
        logger.warning(
            f"Message history truncated from {len(ctx.history)} to {len(limited_history)} messages "
            f"to prevent token limit issues"
        )

    for msg in limited_history:
        # For now, keep history as text-only to avoid token bloat
        # Images are only added to the current query
        history_messages.append(ModelResponse([TextPart(content=str(msg))]))

    # Validate and fix message history to ensure tool calls/results are paired
    history_messages = validate_and_fix_message_history(history_messages)

    return history_messages
