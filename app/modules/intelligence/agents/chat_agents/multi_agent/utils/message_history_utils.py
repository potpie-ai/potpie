"""Message history utility functions for multi-agent system"""

from typing import List, Optional, Any, Set
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)

from app.modules.intelligence.agents.chat_agents.token_utils import count_tokens
from app.modules.intelligence.agents.context_config import (
    get_history_token_budget,
)


def _is_llm_response_with_text(msg: ModelMessage) -> bool:
    """Check if a message is an LLM response with text content.

    CRITICAL: LLM responses should NEVER be removed from history.
    The LLM needs to see what it already said to avoid repeating itself.
    """
    if isinstance(msg, ModelResponse):
        for part in msg.parts:
            if isinstance(part, TextPart):
                content = part.content or ""
                if content.strip():
                    return True
    return False


def _strip_problematic_tool_calls(
    msg: ModelResponse, problematic_ids: Set[str]
) -> Optional[ModelResponse]:
    """Strip problematic tool calls from a ModelResponse while preserving text content.

    CRITICAL: This preserves the LLM's conversational text while removing orphaned
    tool_use blocks that would cause Anthropic API to reject the request.

    Args:
        msg: The ModelResponse to process
        problematic_ids: Set of tool_call_ids that are orphaned/problematic

    Returns:
        A new ModelResponse with only non-problematic parts, or None if no parts remain
    """
    if not isinstance(msg, ModelResponse):
        return None

    kept_parts = []

    for part in msg.parts:
        # Always keep TextPart
        if isinstance(part, TextPart):
            kept_parts.append(part)
        # For ToolCallPart, only keep if not problematic
        elif isinstance(part, ToolCallPart):
            tool_call_id = getattr(part, "tool_call_id", None)
            if tool_call_id and tool_call_id in problematic_ids:
                continue  # Skip this problematic tool call
            kept_parts.append(part)
        else:
            # Keep any other part types
            kept_parts.append(part)

    if not kept_parts:
        return None

    return ModelResponse(parts=kept_parts, model_name=msg.model_name)


from app.modules.intelligence.agents.chat_agent import ChatContext
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


def _history_strings_to_model_messages(
    history_list: List[str],
) -> List[ModelMessage]:
    """Convert history strings (e.g. 'human: ...' / 'ai: ...') to proper ModelMessage list.

    Conversation history is built in conversation_service as f'{msg.type}: {msg.content}'
    (LangChain HumanMessage.type is 'human', AIMessage.type is 'ai'). We must preserve
    user vs assistant roles so the LLM receives correct system / user / assistant mapping:
    - System = agent instructions (handled by PydanticAI).
    - User messages = ModelRequest with UserPromptPart.
    - Assistant messages = ModelResponse with TextPart.
    """
    result: List[ModelMessage] = []
    for raw in history_list:
        s = str(raw).strip()
        if not s:
            continue
        # Support "human: ...", "HUMAN: ...", "ai: ...", "AI_GENERATED: ..."
        lower = s.lower()
        if lower.startswith("human:"):
            content = s[6:].strip()  # len("human:") = 6
            if content:
                result.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif lower.startswith("ai_generated:") or lower.startswith("ai:"):
            prefix_len = 12 if lower.startswith("ai_generated:") else 3
            content = s[prefix_len:].strip()
            if content:
                result.append(ModelResponse(parts=[TextPart(content=content)]))
        else:
            # Legacy or unknown format: treat as assistant to avoid mislabeling user content
            result.append(ModelResponse(parts=[TextPart(content=s)]))
    return result


def _remove_duplicate_tool_results(messages: List[ModelMessage]) -> List[ModelMessage]:
    """Remove duplicate tool_result blocks with the same tool_call_id.

    Handles duplicates both within a single message and across messages.
    Anthropic API requires each tool_use to have exactly one tool_result.
    """
    seen_tool_result_ids: Set[str] = set()
    filtered_messages: List[ModelMessage] = []

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
                    # Check for duplicate within message or across messages
                    if (
                        tool_call_id in seen_in_message
                        or tool_call_id in seen_tool_result_ids
                    ):
                        parts_removed = True
                        logger.warning(
                            f"[Message History] Removing duplicate tool_result: {tool_call_id}"
                        )
                        continue
                    seen_in_message.add(tool_call_id)
                    seen_tool_result_ids.add(tool_call_id)

                filtered_parts.append(part)

            if parts_removed:
                if filtered_parts:
                    msg = ModelRequest(parts=filtered_parts)
                else:
                    continue  # Skip message if all parts were duplicates

        filtered_messages.append(msg)

    return filtered_messages


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

    # First, remove any duplicate tool_results (same tool_call_id appearing multiple times)
    messages = _remove_duplicate_tool_results(messages)

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

        # Track messages that need tool call stripping (LLM responses with problematic tool calls)
        messages_to_strip: set = set()

        for i, msg in enumerate(messages):
            call_ids = message_tool_call_ids[i]
            is_tool_call = _is_tool_call_msg(msg)
            is_tool_result = _is_tool_result_msg(msg)

            if not call_ids:
                continue

            # CRITICAL: Remove if ANY tool_call_id is problematic
            if any(tid in problematic_ids for tid in call_ids):
                # For LLM response messages with text, we don't remove the entire message.
                # Instead we mark it for tool call stripping - keep the text, remove the tool calls.
                if _is_llm_response_with_text(msg):
                    logger.debug(
                        f"[Message History Validation] LLM response at message {i} has problematic "
                        f"tool_call_ids: {call_ids}. Will strip tool calls but preserve text."
                    )
                    messages_to_strip.add(i)
                    continue

                messages_to_skip.add(i)

                # Also remove paired messages (but NOT LLM responses with text)
                if is_tool_call and i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    if not _is_llm_response_with_text(next_msg):
                        messages_to_skip.add(i + 1)
                if is_tool_result and i > 0 and _is_tool_call_msg(messages[i - 1]):
                    prev_msg = messages[i - 1]
                    if not _is_llm_response_with_text(prev_msg):
                        messages_to_skip.add(i - 1)

        # Build filtered messages, stripping tool calls from LLM responses as needed
        filtered_messages = []
        for i, msg in enumerate(messages):
            if i in messages_to_skip:
                continue

            # Strip problematic tool calls from LLM responses
            if i in messages_to_strip and isinstance(msg, ModelResponse):
                stripped_msg = _strip_problematic_tool_calls(msg, problematic_ids)
                if stripped_msg:
                    filtered_messages.append(stripped_msg)
                    logger.debug(
                        f"[Message History Validation] Stripped problematic tool calls from "
                        f"message {i}, kept text content"
                    )
                continue

            filtered_messages.append(msg)

        # Safety check
        if not filtered_messages and messages:
            logger.error(
                "[Message History Validation] All messages removed! Keeping original."
            )
            return messages

        return filtered_messages

    return messages


def _trim_history_to_token_budget(
    history: List[str],
    token_budget: int,
    model_name: Optional[str] = None,
) -> List[str]:
    """Trim history from the start until total tokens <= token_budget; keep the tail."""
    if not history:
        return []
    if token_budget <= 0:
        return []  # No budget => no history (avoids returning full history by mistake)
    total = 0
    start_index = len(history)
    for i in range(len(history) - 1, -1, -1):
        total += count_tokens(str(history[i]), model_name)
        if total > token_budget:
            start_index = i + 1
            break
        start_index = i
    return history[start_index:]


async def prepare_multimodal_message_history(
    ctx: ChatContext,
    history_processor: Any,
    token_budget: Optional[int] = None,
    model_name: Optional[str] = None,
) -> List[ModelMessage]:
    """Prepare message history for the agent from ctx.history (plain text strings).

    Previous conversation turns are represented as plain user/assistant text messages —
    no tool calls or tool results. Tool calls only exist within the current agent run
    (accumulated by pydantic_ai during the run and processed by the history_processor).

    Args:
        ctx: Chat context with history list (strings like "human: ..." / "ai: ...").
        history_processor: History processor passed through (used during the run, not here).
        token_budget: Max tokens for history. When None, uses get_history_token_budget(None).
        model_name: Optional model for token counting and model-aware budget.

    Returns:
        List of ModelMessage (plain text only) for the agent's message_history parameter.
    """
    budget = (
        token_budget
        if token_budget is not None
        else get_history_token_budget(model_name)
    )
    history_list = ctx.history if ctx.history is not None else []
    limited_history = _trim_history_to_token_budget(history_list, budget, model_name)

    if len(limited_history) < len(history_list):
        logger.warning(
            "Message history trimmed from %s to %s messages (token budget %s)",
            len(history_list),
            len(limited_history),
            budget,
        )

    history_messages = _history_strings_to_model_messages(limited_history)
    return validate_and_fix_message_history(history_messages)
