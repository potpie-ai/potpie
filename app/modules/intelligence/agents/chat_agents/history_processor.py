"""
History processor for managing token usage in Pydantic AI agents.

This module provides a history processor that tracks token usage and automatically
summarizes message history when approaching token limits to maintain context while
reducing token consumption.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import tiktoken
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
    SystemPromptPart,
)

logger = logging.getLogger(__name__)

# Debug directory for tokenizer inputs
DEBUG_TOKENIZER_DIR = ".data/tokenizer_debug"

# Token thresholds
TOKEN_LIMIT_THRESHOLD = 35000  # Start evicting/compressing when approaching this limit
TARGET_SUMMARY_TOKENS = 10000  # Target token count for summarized history
LARGE_TOOL_RESULT_THRESHOLD = (
    20000  # Tool results over this will be compressed/summarized
)
MAX_TOOL_RESULT_TOKENS = 20000  # Maximum tokens to keep in large tool results
RECENT_TOOL_RESULTS_TO_KEEP = 7  # Always keep last N tool results in full
MAX_TOOL_ARGS_CHARS = 500  # Maximum characters for tool args in compressed metadata
MAX_TOOL_RESULT_LINES = 6  # Maximum lines of tool result to keep in compressed metadata


class TokenAwareHistoryProcessor:
    """
    History processor that tracks token usage and summarizes message history
    when approaching token limits.

    This processor:
    1. Counts total context tokens (system prompt + tool schemas + message history)
    2. When exceeding the token limit, summarizes old messages while preserving recent ones
    3. Preserves tool call/result pairs to avoid breaking message relationships
    """

    def __init__(
        self,
        summarize_agent: Optional[Agent] = None,
        token_limit: int = TOKEN_LIMIT_THRESHOLD,
        target_summary_tokens: int = TARGET_SUMMARY_TOKENS,
        large_tool_result_threshold: int = LARGE_TOOL_RESULT_THRESHOLD,
        max_tool_result_tokens: int = MAX_TOOL_RESULT_TOKENS,
    ):
        """
        Initialize the history processor.

        Args:
            summarize_agent: Agent to use for summarization. If None, summarization
                            will fall back to keeping only recent messages.
            token_limit: Token threshold at which to trigger summarization
            target_summary_tokens: Target token count for summarized history
            large_tool_result_threshold: Tool results exceeding this token count will be compressed
            max_tool_result_tokens: Maximum tokens to keep for large tool results
        """
        self.token_limit = token_limit
        self.target_summary_tokens = target_summary_tokens
        self.summarize_agent = summarize_agent
        self.large_tool_result_threshold = large_tool_result_threshold
        self.max_tool_result_tokens = max_tool_result_tokens
        # Cache for agent instructions and tools (keyed by agent instance id)
        self._agent_cache: dict[int, Tuple[str, str]] = {}
        # Store last compressed output per conversation (key: conversation_key, value: compressed messages)
        # This allows retrieval of compressed messages for subsequent runs within the same agent execution
        self._last_compressed_output: Dict[str, List[ModelMessage]] = {}

    def _get_tokenizer(self, model_name: Optional[str] = None):
        """Get tiktoken tokenizer for the given model, or fallback to cl100k_base."""
        try:
            if model_name:
                # Try to get encoding for the specific model
                try:
                    return tiktoken.encoding_for_model(model_name)
                except KeyError:
                    logger.debug(
                        f"Model {model_name} not found in tiktoken, using cl100k_base"
                    )
            # Fallback to cl100k_base (used by GPT-3.5, GPT-4, etc.)
            return tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to get tokenizer: {e}, using cl100k_base")
            return tiktoken.get_encoding("cl100k_base")

    def _count_tokens_exact(self, text: str, model_name: Optional[str] = None) -> int:
        """Count exact tokens using tiktoken."""
        if not text:
            return 0
        try:
            encoding = self._get_tokenizer(model_name)
            return len(encoding.encode(str(text), disallowed_special=set()))
        except Exception as e:
            logger.warning(f"Failed to count tokens with tiktoken: {e}, using fallback")
            # Fallback to character-based estimate (roughly 4 chars per token)
            return len(str(text)) // 4

    def _extract_system_prompt_from_messages(self, messages: List[ModelMessage]) -> str:
        """Extract system prompt from messages (usually in first ModelRequest with SystemPromptPart)."""
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

    def _serialize_messages_to_text(self, messages: List[ModelMessage]) -> str:
        """Serialize messages to text for token counting."""
        parts = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, (SystemPromptPart, UserPromptPart)):
                        content = part.content or ""
                        if isinstance(content, str):
                            parts.append(content)
                        else:
                            parts.append(str(content))
                    elif hasattr(part, "__dict__"):
                        # Serialize structured parts (tool calls, etc.)
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

    async def _summarize_messages(
        self,
        messages: List[ModelMessage],
        summarize_agent: Agent,
        model_name: Optional[str] = None,
    ) -> List[ModelMessage]:
        """
        Summarize a list of messages using the provided agent.

        Uses progressive summarization for large message sets to improve quality.

        Returns a new list of messages containing the summary.
        """
        if not messages:
            return []

        # For large message sets, summarize in chunks for better quality
        total_msg_tokens = sum(
            self._count_message_tokens(msg, model_name) for msg in messages
        )

        # Progressive chunking is commented out (uses LLM summarization)
        # # If messages are very large, summarize in progressive chunks
        # if len(messages) > 20 or total_msg_tokens > 20000:
        #     logger.info(
        #         f"Summarizing {len(messages)} messages ({total_msg_tokens} tokens) "
        #         f"using progressive chunking"
        #     )
        #     # Split into chunks and summarize progressively
        #     chunk_size = max(10, len(messages) // 3)
        #     chunks = [
        #         messages[i : i + chunk_size]
        #         for i in range(0, len(messages), chunk_size)
        #     ]
        #
        #     summarized_chunks: List[ModelMessage] = []
        #     for i, chunk in enumerate(chunks):
        #         logger.debug(
        #             f"Summarizing chunk {i+1}/{len(chunks)} ({len(chunk)} messages)"
        #         )
        #         chunk_summary = await self._summarize_messages(
        #             chunk, summarize_agent, model_name
        #         )
        #         summarized_chunks.extend(chunk_summary)
        #
        #     # If we have multiple chunks, summarize the chunk summaries
        #     if len(summarized_chunks) > 1:
        #         logger.debug(
        #             f"Creating final summary from {len(summarized_chunks)} chunk summaries"
        #         )
        #         return await self._summarize_messages(
        #             summarized_chunks, summarize_agent, model_name
        #         )
        #     return summarized_chunks

        logger.info(
            f"Summarization disabled: would have summarized {len(messages)} messages ({total_msg_tokens} tokens)"
        )

        # Create a summary prompt

        # LLM summarization is commented out - using simple truncation fallback instead
        # # Convert messages to a readable format for summarization
        # # We'll use the message history directly with the summarize agent
        # try:
        #     result = await summarize_agent.run(
        #         summary_prompt,
        #         message_history=messages,
        #     )
        #
        #     # Create a summary message from the result
        #     summary_message = ModelRequest(
        #         parts=[
        #             SystemPromptPart(
        #                 content="This is a summary of previous conversation history to preserve context while reducing token usage."
        #             ),
        #             UserPromptPart(
        #                 content=f"## Conversation Summary\n\n{result.output}"
        #             ),
        #         ]
        #     )
        #
        #     logger.info(
        #         f"Successfully summarized {len(messages)} messages into summary message"
        #     )
        #     return [summary_message]
        # except Exception as e:
        #     logger.error(f"Error during message summarization: {e}", exc_info=True)
        #     # Fallback: return recent messages if summarization fails
        #     logger.warning("Falling back to keeping only recent messages")
        #     return messages[-5:] if len(messages) > 5 else messages

        # Fallback: return recent messages (summarization disabled)
        logger.info(
            f"Summarization disabled: returning {min(5, len(messages))} recent messages "
            f"from {len(messages)} messages to summarize"
        )
        return messages[-5:] if len(messages) > 5 else messages

    def _remove_duplicate_tool_results(
        self, messages: List[ModelMessage]
    ) -> List[ModelMessage]:
        """Remove duplicate tool results (multiple tool_result blocks with same tool_call_id).

        Anthropic API requires each tool_use to have exactly one tool_result.
        This method handles duplicates both:
        1. WITHIN a single message (multiple parts with same tool_call_id)
        2. ACROSS messages (same tool_call_id appearing in multiple messages)

        If we find multiple tool_result blocks with the same tool_call_id, we keep only the first one.
        """
        seen_tool_result_ids: Set[str] = set()
        filtered_messages: List[ModelMessage] = []
        removed_parts_count = 0
        removed_messages_count = 0

        for i, msg in enumerate(messages):
            # First, check for and remove duplicate parts WITHIN this message
            if isinstance(msg, ModelRequest):
                seen_in_message: Set[str] = set()
                filtered_parts = []
                parts_removed = False

                for part in msg.parts:
                    tool_call_id = None
                    is_tool_result_part = False

                    if hasattr(part, "__dict__"):
                        part_dict = part.__dict__
                        # Check if this is a tool-return part
                        if part_dict.get("part_kind") == "tool-return":
                            is_tool_result_part = True
                            tool_call_id = part_dict.get("tool_call_id")
                        elif "tool_call_id" in part_dict and (
                            "result" in part_dict or "content" in part_dict
                        ):
                            is_tool_result_part = True
                            tool_call_id = part_dict.get("tool_call_id")

                    if is_tool_result_part and tool_call_id:
                        # Check for duplicate within message
                        if tool_call_id in seen_in_message:
                            removed_parts_count += 1
                            parts_removed = True
                            logger.warning(
                                f"[History Processor] Removing duplicate tool_result part "
                                f"within message {i}: tool_call_id={tool_call_id}"
                            )
                            continue
                        # Check for duplicate across messages
                        if tool_call_id in seen_tool_result_ids:
                            removed_parts_count += 1
                            parts_removed = True
                            logger.warning(
                                f"[History Processor] Removing duplicate tool_result part "
                                f"(cross-message) at message {i}: tool_call_id={tool_call_id}"
                            )
                            continue
                        seen_in_message.add(tool_call_id)
                        seen_tool_result_ids.add(tool_call_id)

                    filtered_parts.append(part)

                # If we removed parts, create a new message with filtered parts
                if parts_removed:
                    if filtered_parts:
                        msg = ModelRequest(parts=filtered_parts)
                    else:
                        # All parts were duplicates - skip this message entirely
                        removed_messages_count += 1
                        logger.warning(
                            f"[History Processor] Removing message {i} entirely "
                            f"(all parts were duplicate tool_results)"
                        )
                        continue

            elif self._is_tool_result_message(msg):
                # For non-ModelRequest tool result messages (edge case)
                tool_call_ids = self._extract_tool_call_ids_from_message(msg)
                is_duplicate = any(tid in seen_tool_result_ids for tid in tool_call_ids)

                if is_duplicate:
                    removed_messages_count += 1
                    logger.warning(
                        f"[History Processor] Removing duplicate tool result message {i}: "
                        f"{tool_call_ids}"
                    )
                    continue

                seen_tool_result_ids.update(tool_call_ids)

            filtered_messages.append(msg)

        if removed_parts_count > 0 or removed_messages_count > 0:
            logger.warning(
                f"[History Processor] Removed {removed_parts_count} duplicate tool_result part(s) "
                f"and {removed_messages_count} duplicate message(s)"
            )

        return filtered_messages

    def _is_user_message(self, msg: ModelMessage) -> bool:
        """Check if a message is a user message (should always be preserved)."""
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    return True
        return False

    def _is_tool_call_message(self, msg: ModelMessage) -> bool:
        """Check if a message contains tool calls (tool_use blocks)."""
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if hasattr(part, "__dict__"):
                    part_dict = part.__dict__
                    # Check part_kind first (most reliable)
                    if "part_kind" in part_dict:
                        if part_dict["part_kind"] == "tool-call":
                            return True
                        elif part_dict["part_kind"] == "tool-return":
                            return False  # It's a result, not a call

                    # Check if this is a tool call (has tool_call_id and tool_name, but no result/content)
                    if "tool_call_id" in part_dict and "tool_name" in part_dict:
                        # It's a call if it doesn't have result or content
                        if "result" not in part_dict and "content" not in part_dict:
                            return True
                    # Check nested structures (FunctionToolCallEvent)
                    if "part" in part_dict:
                        part_obj = part_dict["part"]
                        if hasattr(part_obj, "tool_call_id") and hasattr(
                            part_obj, "tool_name"
                        ):
                            if not hasattr(part_obj, "result") and not hasattr(
                                part_obj, "content"
                            ):
                                return True
        elif isinstance(msg, ModelResponse):
            # ModelResponse can contain ToolCallPart
            for part in msg.parts:
                if hasattr(part, "__dict__"):
                    part_dict = part.__dict__
                    if (
                        "part_kind" in part_dict
                        and part_dict["part_kind"] == "tool-call"
                    ):
                        return True
                    # Check for tool call parts
                    if "tool_name" in part_dict and "tool_call_id" in part_dict:
                        if "result" not in part_dict and "content" not in part_dict:
                            return True
        return False

    def _is_tool_result_message(self, msg: ModelMessage) -> bool:
        """Check if a message contains tool results (which can be large)."""
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                # Check if part represents a tool result
                if hasattr(part, "__dict__"):
                    part_dict = part.__dict__
                    # Check part_kind first (most reliable)
                    if "part_kind" in part_dict:
                        if part_dict["part_kind"] == "tool-return":
                            return True
                        elif part_dict["part_kind"] == "tool-call":
                            return False  # It's a call, not a result

                    # Tool results typically have 'result', 'content', or 'tool_name'
                    if any(
                        key in part_dict
                        for key in ["result", "tool_name", "tool_call_id"]
                    ):
                        # Check if it's a result (has result or content)
                        if "result" in part_dict or "content" in part_dict:
                            return True
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if hasattr(part, "__dict__"):
                    part_dict = part.__dict__
                    if (
                        "part_kind" in part_dict
                        and part_dict["part_kind"] == "tool-return"
                    ):
                        return True
                    if "result" in part_dict or (
                        "tool_name" in part_dict and "content" in part_dict
                    ):
                        return True
        return False

    def _is_llm_response_message(self, msg: ModelMessage) -> bool:
        """Check if a message is an LLM response (ModelResponse with any TextPart).

        CRITICAL: LLM responses should NEVER be removed from history, regardless of length.
        The LLM needs to see what it already said to avoid repeating itself.

        A ModelResponse can contain BOTH TextPart AND ToolCallPart. We must ALWAYS preserve
        the entire message to maintain context, even if the text is short.

        Args:
            msg: The message to check

        Returns:
            True if this is an LLM response that should ALWAYS be preserved
        """
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    content = part.content or ""
                    # ANY text content means this is an LLM response - preserve it
                    # Don't use min_length - even short responses are important context
                    if content.strip():
                        return True
        return False

    def _has_meaningful_text_content(
        self, msg: ModelMessage, min_length: int = 50
    ) -> bool:
        """DEPRECATED: Use _is_llm_response_message instead.

        This method is kept for backward compatibility but now just calls _is_llm_response_message
        with no minimum length requirement.
        """
        return self._is_llm_response_message(msg)

    def _strip_problematic_tool_calls(
        self, msg: ModelResponse, problematic_tool_call_ids: Set[str]
    ) -> Optional[ModelResponse]:
        """Strip problematic tool calls from a ModelResponse while preserving text content.

        CRITICAL: This is the key to preserving LLM text context while maintaining valid
        tool_use/tool_result pairing for the Anthropic API.

        When a ModelResponse contains BOTH TextPart (LLM's conversational text) AND
        ToolCallPart (tool calls), and some of those tool calls are orphaned (no matching
        tool_result), we need to:
        1. KEEP the TextPart (so the LLM doesn't repeat itself)
        2. REMOVE only the problematic ToolCallPart (so Anthropic doesn't reject the request)

        Args:
            msg: The ModelResponse to process
            problematic_tool_call_ids: Set of tool_call_ids that are orphaned/problematic

        Returns:
            A new ModelResponse with only non-problematic parts, or None if no parts remain
        """
        if not isinstance(msg, ModelResponse):
            return None

        # Separate parts into text parts and tool call parts
        kept_parts = []

        for part in msg.parts:
            # Always keep TextPart
            if isinstance(part, TextPart):
                kept_parts.append(part)
            # For ToolCallPart, only keep if not problematic
            elif isinstance(part, ToolCallPart):
                tool_call_id = getattr(part, "tool_call_id", None)
                if tool_call_id and tool_call_id in problematic_tool_call_ids:
                    logger.debug(
                        f"[History Processor] Stripping problematic ToolCallPart with id={tool_call_id}"
                    )
                    continue  # Skip this problematic tool call
                kept_parts.append(part)
            else:
                # Keep any other part types (e.g., RetryPromptPart, etc.)
                kept_parts.append(part)

        if not kept_parts:
            return None

        # Create a new ModelResponse with only the kept parts
        return ModelResponse(parts=kept_parts, model_name=msg.model_name)

    def _count_message_tokens(
        self, msg: ModelMessage, model_name: Optional[str] = None
    ) -> int:
        """Count tokens in a single message."""
        message_text = self._serialize_messages_to_text([msg])
        return self._count_tokens_exact(message_text, model_name)

    def _extract_tool_call_ids_from_message(self, msg: ModelMessage) -> Set[str]:
        """Extract all tool_call_ids from a message (both from tool calls and tool results)."""
        tool_call_ids: Set[str] = set()

        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if hasattr(part, "__dict__"):
                    part_dict = part.__dict__
                    # Check for tool_call_id in tool calls or results
                    if "tool_call_id" in part_dict:
                        tool_call_id = part_dict["tool_call_id"]
                        if tool_call_id:
                            tool_call_ids.add(tool_call_id)
                    # Also check nested structures
                    if "part" in part_dict:
                        part_obj = part_dict["part"]
                        if hasattr(part_obj, "tool_call_id") and part_obj.tool_call_id:
                            tool_call_ids.add(part_obj.tool_call_id)
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if hasattr(part, "__dict__"):
                    part_dict = part.__dict__
                    # Check for tool_call_id in tool results
                    if "tool_call_id" in part_dict:
                        tool_call_id = part_dict["tool_call_id"]
                        if tool_call_id:
                            tool_call_ids.add(tool_call_id)
                    # Check nested result structures
                    if "result" in part_dict:
                        result_obj = part_dict["result"]
                        if (
                            hasattr(result_obj, "tool_call_id")
                            and result_obj.tool_call_id
                        ):
                            tool_call_ids.add(result_obj.tool_call_id)

        return tool_call_ids

    def _extract_tool_call_info_from_message(
        self, msg: ModelMessage
    ) -> Optional[Tuple[str, str, str]]:
        """Extract tool name, arguments, and tool_call_id from a tool call message.

        Returns:
            Tuple of (tool_name, args_str, tool_call_id) or None if not a tool call
        """
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if hasattr(part, "__dict__"):
                    part_dict = part.__dict__
                    # Check if this is a tool call (has tool_name and tool_call_id, but no result)
                    if "tool_name" in part_dict and "tool_call_id" in part_dict:
                        if "result" not in part_dict:  # This is a call, not a result
                            tool_name = part_dict.get("tool_name", "unknown_tool")
                            tool_call_id = part_dict.get("tool_call_id", "")
                            args_str = ""
                            if "args" in part_dict:
                                args = part_dict["args"]
                                if isinstance(args, str):
                                    args_str = args
                                elif isinstance(args, dict):
                                    args_str = json.dumps(args)
                                else:
                                    args_str = str(args)
                            return (str(tool_name), args_str, str(tool_call_id))
                    # Check nested structures (FunctionToolCallEvent)
                    if "part" in part_dict:
                        part_obj = part_dict["part"]
                        if hasattr(part_obj, "tool_name") and hasattr(
                            part_obj, "tool_call_id"
                        ):
                            if not hasattr(part_obj, "result"):  # It's a call
                                tool_name = getattr(
                                    part_obj, "tool_name", "unknown_tool"
                                )
                                tool_call_id = getattr(part_obj, "tool_call_id", "")
                                args_str = ""
                                if hasattr(part_obj, "args"):
                                    args = getattr(part_obj, "args", "")
                                    if isinstance(args, str):
                                        args_str = args
                                    elif isinstance(args, dict):
                                        args_str = json.dumps(args)
                                    else:
                                        args_str = str(args)
                                return (str(tool_name), args_str, str(tool_call_id))
        return None

    def _extract_tool_info_from_message(
        self, msg: ModelMessage
    ) -> Optional[Tuple[str, str, str]]:
        """Extract tool name, result content, and tool_call_id from a tool result message.

        Returns:
            Tuple of (tool_name, result_content, tool_call_id) or None if not a tool result
        """
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if hasattr(part, "__dict__"):
                    part_dict = part.__dict__
                    if (
                        "tool_name" in part_dict
                        or "result" in part_dict
                        or "tool_call_id" in part_dict
                    ):
                        tool_name = part_dict.get("tool_name", "unknown_tool")
                        tool_call_id = part_dict.get("tool_call_id", "")
                        result_content = ""
                        if "result" in part_dict:
                            result = part_dict["result"]
                            if hasattr(result, "content"):
                                result_content = str(result.content)
                            elif isinstance(result, str):
                                result_content = result
                            else:
                                result_content = str(result)
                        elif "content" in part_dict:
                            result_content = str(part_dict["content"])
                        return (str(tool_name), result_content, str(tool_call_id))
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if hasattr(part, "__dict__"):
                    part_dict = part.__dict__
                    if (
                        "tool_name" in part_dict
                        or "result" in part_dict
                        or "tool_call_id" in part_dict
                    ):
                        tool_name = part_dict.get("tool_name", "unknown_tool")
                        tool_call_id = part_dict.get("tool_call_id", "")
                        result_content = ""
                        if "result" in part_dict:
                            result = part_dict["result"]
                            if hasattr(result, "content"):
                                result_content = str(result.content)
                            elif isinstance(result, str):
                                result_content = result
                            else:
                                result_content = str(result)
                        elif "content" in part_dict:
                            result_content = str(part_dict["content"])
                        return (str(tool_name), result_content, str(tool_call_id))
        return None

    def _trim_tool_args(
        self, args_str: str, max_chars: int = MAX_TOOL_ARGS_CHARS
    ) -> str:
        """Trim tool arguments to a maximum character limit."""
        if not args_str or len(args_str) <= max_chars:
            return args_str
        return args_str[:max_chars] + "... [args truncated]"

    def _trim_tool_result_lines(
        self, result_content: str, max_lines: int = MAX_TOOL_RESULT_LINES
    ) -> str:
        """Trim tool result to first N lines."""
        if not result_content:
            return "[No result content]"

        lines = result_content.split("\n")
        if len(lines) <= max_lines:
            return result_content

        trimmed = "\n".join(lines[:max_lines])
        return (
            trimmed + f"\n... [result truncated - {len(lines) - max_lines} more lines]"
        )

    def _create_compressed_tool_metadata(
        self,
        tool_name: str,
        tool_call_id: str,
        tool_args: str,
        result_content: str,
        original_msg: ModelMessage,
    ) -> ModelMessage:
        """Create a compressed metadata message preserving tool call information.

        NOTE: This method is currently NOT USED because it creates invalid message structures
        that break tool_use/tool_result pairing. Instead, we remove tool call/result pairs
        together to maintain proper structure.

        This method is kept for potential future use if we implement proper tool call compression.
        """
        # Trim args and result
        trimmed_args = self._trim_tool_args(tool_args)
        trimmed_result = self._trim_tool_result_lines(result_content)

        metadata_text = f"[Previous tool call - {tool_name}]\n"
        if trimmed_args:
            metadata_text += f"Args: {trimmed_args}\n"
        metadata_text += (
            f"Result (first {MAX_TOOL_RESULT_LINES} lines):\n{trimmed_result}"
        )

        # Return as a simple text message to avoid breaking tool structure
        # This is a fallback - ideally we should remove tool pairs together
        return ModelResponse(parts=[TextPart(content=metadata_text)])

    def _compress_tool_result_content(
        self,
        result_content: str,
        max_result_tokens: int = 200,
        model_name: Optional[str] = None,
    ) -> str:
        """Compress/truncate tool result content if it's too large.

        Returns:
            Compressed content with truncation notice if needed
        """
        if not result_content:
            return "[Tool executed successfully - no result content]"

        result_tokens = self._count_tokens_exact(result_content, model_name)

        if result_tokens <= max_result_tokens:
            # Small result - keep it as-is
            return result_content

        # Large result - truncate with notice
        # Estimate characters per token (rough approximation)
        estimated_chars_per_token = 4
        target_tokens = max_result_tokens - 30  # Leave room for truncation message
        max_chars = target_tokens * estimated_chars_per_token

        # Try to truncate at a word boundary near the limit
        if len(result_content) > max_chars:
            truncated = result_content[:max_chars]
            # Try to truncate at a reasonable boundary (space, newline)
            last_space = max(truncated.rfind("\n"), truncated.rfind(" "))
            if last_space > max_chars * 0.8:  # If we can find a good boundary
                truncated = truncated[:last_space]

            truncated_content = (
                truncated
                + f"\n\n[Result truncated from {result_tokens} tokens to ~{max_result_tokens} tokens - full content no longer in context]"
            )
            return truncated_content

        return result_content

    def _validate_and_fix_tool_pairing(
        self, messages: List[ModelMessage]
    ) -> List[ModelMessage]:
        """Validate and fix tool call/result pairing to prevent API errors.

        Anthropic API requires:
        - Every tool_use block must have a tool_result block in the NEXT message
        - We must ensure tool_use and tool_result are in consecutive messages
        - We remove any tool calls that don't have corresponding results in the next message
        - We remove any tool results that don't have corresponding calls in the previous message

        Returns:
            Validated messages with proper tool call/result pairing
        """
        if not messages:
            return messages

        # First, detect and remove duplicate tool results
        # Anthropic API requires each tool_use to have exactly one tool_result
        messages = self._remove_duplicate_tool_results(messages)

        # Track tool_call_ids that have calls and results
        tool_call_ids_with_calls: Set[str] = set()
        tool_call_ids_with_results: Set[str] = set()

        # Track which messages contain which tool_call_ids
        message_tool_call_ids: List[Set[str]] = []

        # First pass: identify all tool calls and results, and track per message
        for msg in messages:
            call_ids = self._extract_tool_call_ids_from_message(msg)
            message_tool_call_ids.append(call_ids)

            if self._is_tool_call_message(msg):
                tool_call_ids_with_calls.update(call_ids)
            elif self._is_tool_result_message(msg):
                tool_call_ids_with_results.update(call_ids)

        # Find orphaned tool calls (calls without results) and orphaned results (results without calls)
        orphaned_calls = tool_call_ids_with_calls - tool_call_ids_with_results
        orphaned_results = tool_call_ids_with_results - tool_call_ids_with_calls

        # Check for tool_use blocks that don't have corresponding tool_result in the NEXT message
        # CRITICAL: API requirement - tool_result MUST be in the message immediately after tool_use
        # Use tool_call_id matching to find pairs, but verify they're in consecutive messages
        tool_call_ids_without_result: Set[str] = set()

        # Build a map of all tool_call_ids and where they appear
        tool_call_id_locations: Dict[
            str, List[Tuple[int, bool]]
        ] = {}  # tool_call_id -> [(message_index, is_result), ...]

        for i, msg in enumerate(messages):
            call_ids = self._extract_tool_call_ids_from_message(msg)
            self._is_tool_call_message(msg)
            is_result = self._is_tool_result_message(msg)

            for tool_call_id in call_ids:
                if tool_call_id not in tool_call_id_locations:
                    tool_call_id_locations[tool_call_id] = []
                tool_call_id_locations[tool_call_id].append((i, is_result))

        # Find tool calls that don't have corresponding results in the NEXT message
        # This handles parallel tool calls where multiple calls can be in one message
        for i, msg in enumerate(messages):
            if self._is_tool_call_message(msg):
                call_ids = self._extract_tool_call_ids_from_message(msg)
                # Check if the next message has results for these calls
                if i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    next_result_ids = self._extract_tool_call_ids_from_message(next_msg)
                    # Find tool calls that don't have results in the next message
                    missing_results = call_ids - next_result_ids
                    if missing_results:
                        tool_call_ids_without_result.update(missing_results)
                        logger.debug(
                            f"[History Processor] Tool calls {missing_results} at message {i} "
                            f"don't have results in next message {i + 1}"
                        )
                else:
                    # Tool call at the end with no next message
                    tool_call_ids_without_result.update(call_ids)
                    logger.debug(
                        f"[History Processor] Tool calls {call_ids} at last message {i} "
                        f"have no next message for results"
                    )

        # Combine all problematic tool_call_ids
        problematic_tool_call_ids = (
            orphaned_calls | orphaned_results | tool_call_ids_without_result
        )

        if problematic_tool_call_ids:
            logger.warning(
                f"[History Processor] Found problematic tool calls: {problematic_tool_call_ids}. "
                f"Orphaned calls: {orphaned_calls}, orphaned results: {orphaned_results}, "
                f"calls without result: {tool_call_ids_without_result}. "
                f"Removing to prevent API errors."
            )

            # Remove messages with problematic tool calls or results
            # CRITICAL: Remove if ANY tool_call_id in the message is problematic
            # Anthropic API requires ALL tool_use blocks in a message to have results in NEXT message
            # If even ONE tool_call_id is missing its result, the entire message is invalid
            # ALWAYS keep user messages and non-tool messages
            #
            # IMPORTANT: When removing a tool_call message, we must also remove the next message
            # (which should be the tool_result message) to maintain proper pairing.
            validated_messages = []
            messages_to_skip: Set[int] = set()

            # First pass: identify which messages need to be removed (including their pairs)
            for i, msg in enumerate(messages):
                if i in messages_to_skip:
                    continue

                call_ids = message_tool_call_ids[i]
                is_tool_call = self._is_tool_call_message(msg)
                is_tool_result = self._is_tool_result_message(msg)

                # Always keep user messages
                if self._is_user_message(msg):
                    continue

                # If message has no tool calls/results, keep it
                if not call_ids:
                    continue

                # CRITICAL: Remove if ANY tool_call_id is problematic
                # This is different from before - we use 'any' not 'all'
                any_problematic = any(
                    tid in problematic_tool_call_ids for tid in call_ids
                )

                if any_problematic:
                    # CRITICAL: For LLM response messages with text content, we DON'T remove
                    # the entire message. Instead, we'll strip just the problematic tool calls
                    # in the second pass. This preserves the LLM's conversational text context
                    # while removing the orphaned tool_use blocks that Anthropic would reject.
                    if self._is_llm_response_message(msg):
                        logger.debug(
                            f"[History Processor] LLM response message {i} has problematic tool_call_ids: "
                            f"{[tid for tid in call_ids if tid in problematic_tool_call_ids]}. "
                            f"Will strip tool calls but preserve text."
                        )
                        # Don't add to messages_to_skip - we'll handle it specially in second pass
                        continue

                    messages_to_skip.add(i)
                    logger.debug(
                        f"[History Processor] Marking message {i} for removal (has problematic tool_call_ids: "
                        f"{[tid for tid in call_ids if tid in problematic_tool_call_ids]})"
                    )

                    # CRITICAL: If this is a tool_call message, also remove the next message
                    # (which should contain the tool_results)
                    if is_tool_call and i + 1 < len(messages):
                        next_msg = messages[i + 1]
                        # Don't remove if next message is an LLM response with text
                        if not self._is_llm_response_message(next_msg):
                            messages_to_skip.add(i + 1)
                            logger.debug(
                                f"[History Processor] Also marking message {i + 1} for removal "
                                f"(tool_result for removed tool_call)"
                            )

                    # CRITICAL: If this is a tool_result message, also remove the previous message
                    # (which should contain the tool_calls)
                    if is_tool_result and i > 0:
                        prev_msg = messages[i - 1]
                        # Don't remove if previous message is an LLM response with text
                        if self._is_tool_call_message(
                            prev_msg
                        ) and not self._is_llm_response_message(prev_msg):
                            messages_to_skip.add(i - 1)
                            logger.debug(
                                f"[History Processor] Also marking message {i - 1} for removal "
                                f"(tool_call for removed tool_result)"
                            )

            # Second pass: build the validated messages list
            # For LLM response messages with problematic tool calls, strip the tool calls
            # but preserve the text content
            for i, msg in enumerate(messages):
                if i in messages_to_skip:
                    call_ids = message_tool_call_ids[i]
                    logger.debug(
                        f"[History Processor] Removing message {i} with tool_call_ids: {call_ids}"
                    )
                    continue

                # Check if this is an LLM response that needs tool call stripping
                if isinstance(msg, ModelResponse) and self._is_llm_response_message(
                    msg
                ):
                    call_ids = message_tool_call_ids[i]
                    has_problematic = any(
                        tid in problematic_tool_call_ids for tid in call_ids
                    )

                    if has_problematic:
                        # Strip problematic tool calls but keep text
                        stripped_msg = self._strip_problematic_tool_calls(
                            msg, problematic_tool_call_ids
                        )
                        if stripped_msg:
                            validated_messages.append(stripped_msg)
                            logger.debug(
                                f"[History Processor] Stripped problematic tool calls from LLM response {i}, "
                                f"kept text content"
                            )
                        else:
                            # Shouldn't happen since we checked _is_llm_response_message
                            logger.warning(
                                f"[History Processor] Stripping tool calls from message {i} left no content"
                            )
                        continue

                validated_messages.append(msg)

            # Safety check: if we removed everything, keep at least user messages
            if not validated_messages:
                logger.error(
                    "[History Processor] Validation removed all messages! Keeping user messages only."
                )
                validated_messages = [
                    msg for msg in messages if self._is_user_message(msg)
                ]
                # If still empty, keep first message as fallback
                if not validated_messages and messages:
                    logger.error(
                        "[History Processor] No user messages found! Keeping first message as fallback."
                    )
                    validated_messages = [messages[0]]

            # After removing messages, validate again to catch any new orphaned pairs
            # This is necessary because removing a message might create new orphans
            # Use iterative approach with max iterations to prevent infinite loops
            max_iterations = 10
            iteration = 0
            current_messages = validated_messages
            len(messages)

            while iteration < max_iterations:
                previous_count = len(current_messages)
                working_messages = current_messages
                # Re-validate
                current_messages = []
                working_tool_call_ids = []

                for msg in working_messages:
                    call_ids = self._extract_tool_call_ids_from_message(msg)
                    working_tool_call_ids.append(call_ids)

                # Re-check for problematic tool calls
                working_tool_call_ids_with_calls = set()
                working_tool_call_ids_with_results = set()
                for i, msg in enumerate(working_messages):
                    if self._is_tool_call_message(msg):
                        working_tool_call_ids_with_calls.update(
                            working_tool_call_ids[i]
                        )
                    elif self._is_tool_result_message(msg):
                        working_tool_call_ids_with_results.update(
                            working_tool_call_ids[i]
                        )

                working_orphaned_calls = (
                    working_tool_call_ids_with_calls
                    - working_tool_call_ids_with_results
                )
                working_orphaned_results = (
                    working_tool_call_ids_with_results
                    - working_tool_call_ids_with_calls
                )

                # Check for tool calls without results (using tool_call_id matching)
                working_tool_call_ids_without_result = set()
                working_tool_call_id_locations: Dict[str, List[Tuple[int, bool]]] = {}

                for i, msg in enumerate(working_messages):
                    call_ids = working_tool_call_ids[i]
                    self._is_tool_call_message(msg)
                    is_result = self._is_tool_result_message(msg)

                    for tool_call_id in call_ids:
                        if tool_call_id not in working_tool_call_id_locations:
                            working_tool_call_id_locations[tool_call_id] = []
                        working_tool_call_id_locations[tool_call_id].append(
                            (i, is_result)
                        )

                # Find tool calls that don't have corresponding results in the NEXT message
                # CRITICAL: Anthropic requires tool_result to be in the IMMEDIATELY NEXT message
                for i, msg in enumerate(working_messages):
                    if self._is_tool_call_message(msg):
                        call_ids = working_tool_call_ids[i]
                        # Check if next message exists and has results for ALL these calls
                        if i + 1 < len(working_messages):
                            next_msg = working_messages[i + 1]
                            next_result_ids = self._extract_tool_call_ids_from_message(
                                next_msg
                            )
                            # Find tool calls without results in next message
                            missing = call_ids - next_result_ids
                            working_tool_call_ids_without_result.update(missing)
                        else:
                            # No next message - all calls are orphaned
                            working_tool_call_ids_without_result.update(call_ids)

                working_problematic = (
                    working_orphaned_calls
                    | working_orphaned_results
                    | working_tool_call_ids_without_result
                )

                if not working_problematic:
                    # No more issues
                    break

                # Build the list of messages to skip (including paired messages)
                working_messages_to_skip: Set[int] = set()
                for i, msg in enumerate(working_messages):
                    call_ids = working_tool_call_ids[i]
                    is_tool_call = self._is_tool_call_message(msg)
                    is_tool_result = self._is_tool_result_message(msg)

                    if not call_ids:
                        continue

                    # CRITICAL: ALWAYS preserve LLM response messages with text content
                    if self._is_llm_response_message(msg):
                        continue

                    # CRITICAL: Remove if ANY tool_call_id is problematic
                    if any(tid in working_problematic for tid in call_ids):
                        working_messages_to_skip.add(i)
                        # Also remove paired messages (but NOT LLM responses with text)
                        if is_tool_call and i + 1 < len(working_messages):
                            next_msg = working_messages[i + 1]
                            if not self._is_llm_response_message(next_msg):
                                working_messages_to_skip.add(i + 1)
                        if (
                            is_tool_result
                            and i > 0
                            and self._is_tool_call_message(working_messages[i - 1])
                        ):
                            prev_msg = working_messages[i - 1]
                            if not self._is_llm_response_message(prev_msg):
                                working_messages_to_skip.add(i - 1)

                for i, msg in enumerate(working_messages):
                    if i not in working_messages_to_skip:
                        current_messages.append(msg)
                    else:
                        logger.debug(
                            f"Removing message {i} in iteration {iteration}: {working_tool_call_ids[i]}"
                        )

                if len(current_messages) == previous_count:
                    # No change, stop iterating
                    break

                iteration += 1

            if iteration >= max_iterations:
                logger.warning(
                    f"[History Processor] Reached max iterations ({max_iterations}) in validation"
                )

            return current_messages

        return messages

    def _continue_compressing_until_under_limit(
        self,
        ctx: RunContext,
        messages: List[ModelMessage],
        model_name: Optional[str] = None,
    ) -> List[ModelMessage]:
        """Continue removing older tool calls/results until under token limit.

        This iteratively removes the oldest tool calls/results (both together)
        until we're under the limit to avoid breaking tool_use/tool_result pairing.

        CRITICAL: When removing a tool call message, we MUST also remove its
        corresponding tool result message (the next message) to maintain pairing.
        """
        current_tokens = self._count_total_context_tokens(ctx, messages, model_name)

        if current_tokens <= self.token_limit:
            return messages

        # Build message metadata with tool call/result pairing information
        # Use tool_call_id to match calls with results (not message position)
        message_metadata: List[
            Tuple[ModelMessage, int, bool, bool, bool, Set[str], Set[int]]
        ] = []

        # Map tool_call_id -> list of message indices that contain it
        # This handles parallel tool calls where multiple calls/results can be in one message
        tool_call_id_to_message_indices: Dict[str, List[int]] = {}

        for i, msg in enumerate(messages):
            is_user = self._is_user_message(msg)
            is_tool_call = self._is_tool_call_message(msg)
            is_tool_result = self._is_tool_result_message(msg)
            msg_tokens = self._count_message_tokens(msg, model_name)
            tool_call_ids = self._extract_tool_call_ids_from_message(msg)

            # Track which messages contain each tool_call_id
            for tool_call_id in tool_call_ids:
                if tool_call_id not in tool_call_id_to_message_indices:
                    tool_call_id_to_message_indices[tool_call_id] = []
                tool_call_id_to_message_indices[tool_call_id].append(i)

            # Find paired message indices (messages that contain matching tool_call_ids)
            # This finds all messages that have results for our calls, or calls for our results
            paired_indices: Set[int] = set()
            for tool_call_id in tool_call_ids:
                if tool_call_id in tool_call_id_to_message_indices:
                    # Get all messages that contain this tool_call_id
                    indices_with_same_id = tool_call_id_to_message_indices[tool_call_id]
                    # Add other messages (not current) that contain this tool_call_id
                    for idx in indices_with_same_id:
                        if idx != i:
                            paired_indices.add(idx)

            message_metadata.append(
                (
                    msg,
                    msg_tokens,
                    is_user,
                    is_tool_call,
                    is_tool_result,
                    tool_call_ids,
                    paired_indices,
                )
            )

        # Identify recent tool results to keep (last N)
        # Keep tool_call_ids from recent tool results AND their corresponding tool calls
        tool_call_ids_to_keep: Set[str] = set()
        seen_recent_results = 0
        for msg, _, _, _, is_tool_result, tool_call_ids, _ in reversed(
            message_metadata
        ):
            if is_tool_result:
                seen_recent_results += 1
                if seen_recent_results <= RECENT_TOOL_RESULTS_TO_KEEP:
                    tool_call_ids_to_keep.update(tool_call_ids)
                    # Also keep the corresponding tool calls for these results
                    for tool_call_id in tool_call_ids:
                        if tool_call_id in tool_call_id_to_message_indices:
                            # Mark all messages with this tool_call_id as needing to be kept
                            pass  # Already handled by tool_call_ids_to_keep

        # Remove oldest tool calls/results until under limit
        # CRITICAL: API requirement - if message N has tool_use, message N+1 MUST have tool_result
        # When keeping a tool call message, we MUST also keep the next message with its results
        filtered_messages: List[ModelMessage] = []
        removed_tool_metadata: List[Tuple[str, str, str]] = []
        messages_to_skip: Set[int] = set()  # Track messages we've already removed
        messages_to_keep_next: Set[int] = (
            set()
        )  # Track messages whose next message must be kept

        for i, (
            msg,
            msg_tokens,
            is_user,
            is_tool_call,
            is_tool_result,
            tool_call_ids,
            paired_indices,
        ) in enumerate(message_metadata):
            # Skip if already removed as part of a pair
            if i in messages_to_skip:
                continue

            # Always keep user messages
            if is_user:
                filtered_messages.append(msg)
                continue

            # CRITICAL: ALWAYS preserve LLM response messages (ModelResponse with any TextPart)
            # This is the most important rule - the LLM needs to see what it already said
            # to avoid repeating itself. This takes precedence over ALL other logic.
            if self._is_llm_response_message(msg):
                filtered_messages.append(msg)
                logger.debug(
                    f"[History Processor] ALWAYS preserving LLM response at message {i} during compression"
                )
                # If this is also a tool call, mark next message to be kept for tool_result pairing
                if is_tool_call and i + 1 < len(messages):
                    messages_to_keep_next.add(i + 1)
                continue

            # Check if this tool call/result should be kept
            is_recent = any(tid in tool_call_ids_to_keep for tid in tool_call_ids)
            # Also check if previous message requires us to keep this one (for tool results)
            must_keep = i in messages_to_keep_next

            if (is_tool_call or is_tool_result) and not is_recent and not must_keep:
                # This is an old tool call/result - remove it
                # CRITICAL: If this is a tool call message, we must also remove the next message
                # (which should contain the tool results)

                # Collect metadata for all tool_call_ids in this message
                for tool_call_id in tool_call_ids:
                    if any(tid == tool_call_id for _, _, tid in removed_tool_metadata):
                        continue  # Already collected

                    # Try to extract tool info
                    if is_tool_call:
                        call_info = self._extract_tool_call_info_from_message(msg)
                        if call_info:
                            tool_name, tool_args, extracted_id = call_info
                            if extracted_id == tool_call_id:
                                removed_tool_metadata.append(
                                    (tool_name, tool_args, tool_call_id)
                                )
                    elif is_tool_result:
                        result_info = self._extract_tool_info_from_message(msg)
                        if result_info:
                            tool_name, result_content, extracted_id = result_info
                            if extracted_id == tool_call_id:
                                # Find corresponding call from paired messages
                                tool_args = ""
                                for paired_idx in paired_indices:
                                    if paired_idx < len(message_metadata):
                                        paired_msg = message_metadata[paired_idx][0]
                                        paired_call_ids = (
                                            self._extract_tool_call_ids_from_message(
                                                paired_msg
                                            )
                                        )
                                        if tool_call_id in paired_call_ids:
                                            call_info = self._extract_tool_call_info_from_message(
                                                paired_msg
                                            )
                                            if call_info:
                                                _, tool_args, _ = call_info
                                                break
                                removed_tool_metadata.append(
                                    (tool_name, tool_args, tool_call_id)
                                )

                # Mark this message for removal
                messages_to_skip.add(i)

                # CRITICAL: If this is a tool call message, we MUST also remove the next message
                # (which should contain the tool results for these calls)
                if is_tool_call and i + 1 < len(messages):
                    messages_to_skip.add(i + 1)
                    logger.debug(
                        f"[History Processor] Removing tool call message {i} and its result message {i + 1}"
                    )
                else:
                    logger.debug(f"[History Processor] Removing tool message {i}")

                # Don't add to filtered_messages
                continue

            # Keep this message
            filtered_messages.append(msg)

            # CRITICAL: If this is a tool call message, mark the next message to be kept
            # (it should contain the tool results)
            if is_tool_call and i + 1 < len(messages):
                messages_to_keep_next.add(i + 1)

            # Check if we're under the limit
            current_tokens = self._count_total_context_tokens(
                ctx, filtered_messages, model_name
            )
            if current_tokens <= self.token_limit:
                break

        # CRITICAL: Validate tool call/result pairing after compression
        # This ensures we haven't broken any pairings during removal
        filtered_messages = self._validate_and_fix_tool_pairing(filtered_messages)

        # Add metadata summary if we removed tool calls
        if removed_tool_metadata:
            summary_lines = ["[Previous tool calls (removed to save tokens):]"]
            for tool_name, tool_args, tool_call_id in removed_tool_metadata:
                trimmed_args = self._trim_tool_args(tool_args) if tool_args else ""
                summary_lines.append(f"\n- Tool: {tool_name}")
                if trimmed_args:
                    summary_lines.append(f"  Args: {trimmed_args}")
            summary_text = "\n".join(summary_lines)
            summary_msg = ModelRequest(
                parts=[
                    SystemPromptPart(
                        content="Summary of previous tool calls removed from context."
                    ),
                    UserPromptPart(content=summary_text),
                ]
            )
            # Insert after user messages if any, otherwise at beginning
            insert_idx = 0
            for i, msg in enumerate(filtered_messages):
                if self._is_user_message(msg):
                    insert_idx = i + 1
                    break
            filtered_messages.insert(insert_idx, summary_msg)

        return filtered_messages

    def _get_history_key_from_context(self, ctx: RunContext) -> Optional[str]:
        """Generate a history key from RunContext to identify the conversation.

        Uses available context attributes to create a unique key for this run.
        Since RunContext attributes may vary, we use the context object id as the primary key.
        """
        try:
            # Use context object id as the primary key (most reliable within same execution)
            # This ensures each RunContext instance gets its own storage
            context_id = id(ctx)

            # Try to enhance with additional identifiers if available
            additional_info = []

            # Try to get agent instance id if available
            try:
                if hasattr(ctx, "agent") and getattr(ctx, "agent", None):
                    agent = getattr(ctx, "agent")
                    additional_info.append(f"agent_{id(agent)}")
            except Exception:
                pass

            # Try to get model info if available
            try:
                if hasattr(ctx, "model") and getattr(ctx, "model", None):
                    model = getattr(ctx, "model")
                    model_str = str(model)[:50]  # Truncate long model strings
                    additional_info.append(f"model_{hash(model_str)}")
            except Exception:
                pass

            # Combine context id with additional info if available
            if additional_info:
                return f"ctx_{context_id}_{'_'.join(additional_info)}"
            else:
                return f"ctx_{context_id}"
        except Exception as e:
            logger.debug(f"Failed to generate history key from context: {e}")
            # Last resort: use context object id only
            return f"ctx_{id(ctx)}"

    def get_compressed_history(self, history_key: str) -> Optional[List[ModelMessage]]:
        """Retrieve compressed message history for a given key.

        Args:
            history_key: Key identifying the conversation (from _get_history_key_from_context)

        Returns:
            Compressed messages if available, None otherwise
        """
        return self._last_compressed_output.get(history_key)

    def clear_compressed_history(self, history_key: Optional[str] = None):
        """Clear compressed history for a specific key or all keys.

        Args:
            history_key: Key to clear, or None to clear all
        """
        if history_key:
            self._last_compressed_output.pop(history_key, None)
            logger.debug(f"Cleared compressed history for key: {history_key}")
        else:
            self._last_compressed_output.clear()
            logger.debug("Cleared all compressed history")

    def _get_model_name_from_context(self, ctx: RunContext) -> Optional[str]:
        """Extract model name from RunContext if available."""
        try:
            if hasattr(ctx, "model") and ctx.model:
                model = ctx.model
                # Try to get model name from various attributes
                if hasattr(model, "model_name"):
                    return getattr(model, "model_name", None)
                # Try to extract from string representation
                model_str = str(model)
                # Look for common patterns like "gpt-4", "claude", etc.
                match = re.search(
                    r"(gpt-[0-9.]+|claude-[0-9.]+|gemini-[0-9.]+)",
                    model_str,
                    re.IGNORECASE,
                )
                if match:
                    return match.group(1)
        except Exception as e:
            logger.debug(f"Failed to extract model name from context: {e}")
        return None

    def _get_agent_from_context(self, ctx: RunContext) -> Optional[Agent]:
        """Extract agent from RunContext if available."""
        try:
            # Try various ways to access the agent
            if hasattr(ctx, "agent"):
                agent = getattr(ctx, "agent", None)
                if agent:
                    return agent
            if hasattr(ctx, "_agent"):
                agent = getattr(ctx, "_agent", None)
                if agent:
                    return agent
            # Try to get from run_state if available
            if hasattr(ctx, "run_state"):
                run_state = getattr(ctx, "run_state", None)
                if run_state and hasattr(run_state, "agent"):
                    agent = getattr(run_state, "agent", None)
                    if agent:
                        return agent
        except Exception as e:
            logger.debug(f"Failed to extract agent from context: {e}")
        return None

    def _cache_agent_info(self, agent: Agent) -> Tuple[str, str]:
        """Extract and cache agent instructions and tool schemas."""
        agent_id = id(agent)
        if agent_id in self._agent_cache:
            return self._agent_cache[agent_id]

        instructions = ""
        tool_schemas = ""

        try:
            # Try to get instructions
            if hasattr(agent, "instructions"):
                instructions = str(getattr(agent, "instructions", ""))
            elif hasattr(agent, "_instructions"):
                instructions = str(getattr(agent, "_instructions", ""))

            # Try to get tools
            if hasattr(agent, "tools"):
                tools = getattr(agent, "tools", None)
                if tools:
                    tool_list = []
                    for tool in tools:
                        tool_dict = {}
                        if hasattr(tool, "name"):
                            tool_dict["name"] = getattr(tool, "name", "")
                        if hasattr(tool, "description"):
                            tool_dict["description"] = getattr(tool, "description", "")
                        if hasattr(tool, "args_schema"):
                            try:
                                schema = getattr(tool, "args_schema", None)
                                if schema:
                                    tool_dict["args_schema"] = str(schema)
                            except Exception:
                                pass
                        tool_list.append(json.dumps(tool_dict, default=str))
                    tool_schemas = "\n".join(tool_list)
        except Exception as e:
            logger.debug(f"Failed to cache agent info: {e}")

        self._agent_cache[agent_id] = (instructions, tool_schemas)
        return (instructions, tool_schemas)

    def _write_tokenizer_debug_file(
        self,
        content: str,
        section: str,
        model_name: Optional[str] = None,
        token_count: Optional[int] = None,
        ctx: Optional[RunContext] = None,
    ):
        """Write tokenizer input to a debug file for inspection."""
        try:
            # Create debug directory if it doesn't exist
            os.makedirs(DEBUG_TOKENIZER_DIR, exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_{section}.txt"
            filepath = os.path.join(DEBUG_TOKENIZER_DIR, filename)

            # Write content with metadata
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("=== Tokenizer Debug File ===\n")
                f.write(f"Section: {section}\n")
                f.write(f"Model: {model_name or 'cl100k_base'}\n")
                if token_count is not None:
                    f.write(f"Token Count: {token_count}\n")
                f.write(f"Content Length: {len(content)} characters\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")

                # Log RunContext information if provided
                if ctx is not None:
                    f.write(f"\n{'=' * 50}\n")
                    f.write("RunContext Information:\n")
                    f.write(f"{'=' * 50}\n")
                    try:
                        # Try to extract useful information from RunContext
                        ctx_info = {}
                        if hasattr(ctx, "usage") and ctx.usage:
                            ctx_info["usage"] = {
                                "total_tokens": getattr(
                                    ctx.usage, "total_tokens", None
                                ),
                                "input_tokens": getattr(
                                    ctx.usage, "input_tokens", None
                                ),
                                "output_tokens": getattr(
                                    ctx.usage, "output_tokens", None
                                ),
                            }
                        if hasattr(ctx, "model"):
                            model = getattr(ctx, "model", None)
                            if model:
                                ctx_info["model"] = str(model)
                        if hasattr(ctx, "agent"):
                            agent = getattr(ctx, "agent", None)
                            if agent:
                                ctx_info["agent"] = str(type(agent))
                        if hasattr(ctx, "run_state"):
                            run_state = getattr(ctx, "run_state", None)
                            if run_state:
                                ctx_info["run_state"] = str(type(run_state))

                        # Get all attributes
                        ctx_attrs = [
                            attr for attr in dir(ctx) if not attr.startswith("__")
                        ]
                        ctx_info["available_attributes"] = ctx_attrs

                        f.write(json.dumps(ctx_info, indent=2, default=str))
                    except Exception as e:
                        f.write(f"Error extracting RunContext info: {e}\n")
                        f.write(f"RunContext type: {type(ctx)}\n")
                        f.write(f"RunContext repr: {repr(ctx)[:500]}\n")

                f.write(f"\n{'=' * 50}\n")
                f.write("CONTENT:\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(content)

            logger.info(
                f"[History Processor] Wrote tokenizer debug file: {filepath} "
                f"({len(content)} chars, {token_count or 'N/A'} tokens)"
            )
        except Exception as e:
            logger.warning(f"Failed to write tokenizer debug file: {e}")

    def _write_messages_to_debug_file(
        self,
        messages: List[ModelMessage],
        model_name: Optional[str] = None,
        token_count: Optional[int] = None,
        ctx: Optional[RunContext] = None,
    ):
        """Write the actual messages that will be sent to the LLM to a debug file."""
        try:
            # Create debug directory if it doesn't exist
            os.makedirs(DEBUG_TOKENIZER_DIR, exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_messages_to_llm.txt"
            filepath = os.path.join(DEBUG_TOKENIZER_DIR, filename)

            # Serialize messages to text
            message_text = self._serialize_messages_to_text(messages)

            # Write content with metadata
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("=== Messages Sent to LLM ===\n")
                f.write(f"Model: {model_name or 'cl100k_base'}\n")
                if token_count is not None:
                    f.write(f"Total Token Count: {token_count}\n")
                f.write(f"Message Count: {len(messages)}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")

                # Log RunContext information if provided
                if ctx is not None:
                    f.write(f"\n{'=' * 50}\n")
                    f.write("RunContext Information:\n")
                    f.write(f"{'=' * 50}\n")
                    try:
                        ctx_info = {}
                        if hasattr(ctx, "usage") and ctx.usage:
                            ctx_info["usage"] = {
                                "total_tokens": getattr(
                                    ctx.usage, "total_tokens", None
                                ),
                                "input_tokens": getattr(
                                    ctx.usage, "input_tokens", None
                                ),
                                "output_tokens": getattr(
                                    ctx.usage, "output_tokens", None
                                ),
                            }
                        if hasattr(ctx, "model"):
                            model = getattr(ctx, "model", None)
                            if model:
                                ctx_info["model"] = str(model)
                        if hasattr(ctx, "agent"):
                            agent = getattr(ctx, "agent", None)
                            if agent:
                                ctx_info["agent"] = str(type(agent))
                        if hasattr(ctx, "run_state"):
                            run_state = getattr(ctx, "run_state", None)
                            if run_state:
                                ctx_info["run_state"] = str(type(run_state))

                        # Get all attributes
                        ctx_attrs = [
                            attr for attr in dir(ctx) if not attr.startswith("__")
                        ]
                        ctx_info["available_attributes"] = ctx_attrs

                        f.write(json.dumps(ctx_info, indent=2, default=str))
                    except Exception as e:
                        f.write(f"Error extracting RunContext info: {e}\n")
                        f.write(f"RunContext type: {type(ctx)}\n")
                        f.write(f"RunContext repr: {repr(ctx)[:500]}\n")

                f.write(f"\n{'=' * 50}\n")
                f.write("MESSAGE CONTENT (as sent to LLM):\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(message_text)

                # Also write a detailed breakdown of each message
                f.write(f"\n\n{'=' * 50}\n")
                f.write("DETAILED MESSAGE BREAKDOWN:\n")
                f.write(f"{'=' * 50}\n\n")
                for i, msg in enumerate(messages):
                    f.write(f"--- Message {i} ---\n")
                    f.write(f"Type: {type(msg).__name__}\n")

                    # Extract tool call IDs if present
                    tool_call_ids = self._extract_tool_call_ids_from_message(msg)
                    if tool_call_ids:
                        f.write(f"Tool Call IDs: {tool_call_ids}\n")

                    # Check if it's a tool call or result
                    if self._is_tool_call_message(msg):
                        f.write("Message Type: TOOL_CALL\n")
                        call_info = self._extract_tool_call_info_from_message(msg)
                        if call_info:
                            tool_name, tool_args, tool_call_id = call_info
                            f.write(f"  Tool Name: {tool_name}\n")
                            f.write(f"  Tool Call ID: {tool_call_id}\n")
                            f.write(
                                f"  Args: {tool_args[:200]}...\n"
                                if len(tool_args) > 200
                                else f"  Args: {tool_args}\n"
                            )
                    elif self._is_tool_result_message(msg):
                        f.write("Message Type: TOOL_RESULT\n")
                        result_info = self._extract_tool_info_from_message(msg)
                        if result_info:
                            tool_name, result_content, tool_call_id = result_info
                            f.write(f"  Tool Name: {tool_name}\n")
                            f.write(f"  Tool Call ID: {tool_call_id}\n")
                            f.write(f"  Result Length: {len(result_content)} chars\n")
                            f.write(
                                f"  Result Preview: {result_content[:200]}...\n"
                                if len(result_content) > 200
                                else f"  Result: {result_content}\n"
                            )
                    elif self._is_user_message(msg):
                        f.write("Message Type: USER\n")
                    else:
                        f.write("Message Type: OTHER\n")

                    # Write message representation
                    try:
                        msg_repr = repr(msg)
                        if len(msg_repr) > 500:
                            msg_repr = msg_repr[:500] + "..."
                        f.write(f"Message Repr: {msg_repr}\n")
                    except Exception as e:
                        f.write(f"Error getting message repr: {e}\n")

                    f.write("\n")

            logger.info(
                f"[History Processor] Wrote messages to LLM debug file: {filepath} "
                f"({len(messages)} messages, {token_count or 'N/A'} tokens)"
            )
        except Exception as e:
            logger.warning(f"Failed to write messages to LLM debug file: {e}")

    def _count_total_context_tokens(
        self,
        ctx: RunContext,
        messages: List[ModelMessage],
        model_name: Optional[str] = None,
    ) -> int:
        """
        Count total tokens in the complete context that will be sent to the LLM.

        This includes:
        1. System prompt (agent instructions)
        2. Tool schemas (all tool definitions)
        3. Message history (conversation messages)

        Args:
            ctx: RunContext containing agent and model information
            messages: Message history
            model_name: Optional model name for tokenizer

        Returns:
            Total token count for the complete context
        """
        total = 0

        # Try to get agent from context and cache its info
        agent = self._get_agent_from_context(ctx)

        # Try to extract system prompt from agent
        system_prompt_text = ""
        tool_schemas_text = ""

        if agent:
            # Cache agent info (instructions and tools)
            instructions, tool_schemas = self._cache_agent_info(agent)
            system_prompt_text = instructions
            tool_schemas_text = tool_schemas

        # Fallback: Extract system prompt from messages if not found in agent
        if not system_prompt_text:
            system_prompt_text = self._extract_system_prompt_from_messages(messages)

        # Count system prompt tokens
        system_tokens = 0
        if system_prompt_text:
            system_tokens = self._count_tokens_exact(system_prompt_text, model_name)
            total += system_tokens

        # Count tool schema tokens
        tool_schema_tokens = 0
        if tool_schemas_text:
            tool_schema_tokens = self._count_tokens_exact(tool_schemas_text, model_name)
            total += tool_schema_tokens

        # Count message history tokens
        message_text = self._serialize_messages_to_text(messages)
        message_tokens = self._count_tokens_exact(message_text, model_name)
        total += message_tokens

        # Write debug files if enabled (only when debugging is needed)
        # Commented out - not needed right now
        # if logger.isEnabledFor(logging.DEBUG):
        #     if system_tokens > 0:
        #         self._write_tokenizer_debug_file(
        #             system_prompt_text, "system_prompt", model_name, system_tokens, ctx
        #         )
        #     if tool_schema_tokens > 0:
        #         self._write_tokenizer_debug_file(
        #             tool_schemas_text,
        #             "tool_schemas",
        #             model_name,
        #             tool_schema_tokens,
        #             ctx,
        #         )
        #     self._write_tokenizer_debug_file(
        #         message_text, "message_history", model_name, message_tokens, ctx
        #     )
        #     combined_text = f"=== SYSTEM PROMPT ===\n{system_prompt_text}\n\n"
        #     combined_text += f"=== TOOL SCHEMAS ===\n{tool_schemas_text}\n\n"
        #     combined_text += f"=== MESSAGE HISTORY ===\n{message_text}\n"
        #     self._write_tokenizer_debug_file(
        #         combined_text, "combined_context", model_name, total, ctx
        #     )

        return total

    async def __call__(
        self,
        ctx: RunContext,
        messages: List[ModelMessage],
    ) -> List[ModelMessage]:
        """
        Process message history based on token usage.

        SIMPLIFIED APPROACH: This processor now only logs token counts and returns
        messages unchanged to avoid breaking tool_call/tool_result pairing required
        by OpenAI and Anthropic APIs.

        The previous implementation tried to remove/modify messages to stay under
        token limits, but this was causing "tool messages must follow tool_calls"
        errors. It's better to exceed token limits than to send malformed messages.

        Args:
            ctx: RunContext containing usage information and model access
            messages: Current message history

        Returns:
            Original message history (unchanged)
        """
        # Get model name from context if available
        model_name = self._get_model_name_from_context(ctx)

        # Count total context tokens (system prompt + tool schemas + message history)
        total_tokens = self._count_total_context_tokens(ctx, messages, model_name)

        model_info = (
            f", model: {model_name}" if model_name else ", encoding: cl100k_base"
        )
        logger.info(
            f"[History Processor] Total context tokens: {total_tokens} "
            f"(system + tools + messages, using tiktoken{model_info})"
        )

        # SIMPLIFIED: Always return original messages to preserve message structure
        # The previous complex logic was breaking tool_call/tool_result pairing
        if total_tokens >= self.token_limit:
            logger.warning(
                f"[History Processor] Token count {total_tokens} exceeds limit {self.token_limit}, "
                f"but returning original messages to preserve message structure. "
                f"Message count: {len(messages)}"
            )
        else:
            logger.debug(
                f"Token count {total_tokens} < limit {self.token_limit}, no action needed"
            )

        return messages

        # ============================================================================
        # DISABLED: The complex message processing below was causing errors like:
        # "An assistant message with 'tool_calls' must be followed by tool messages
        #  responding to each 'tool_call_id'"
        #
        # The logic tried to remove old tool calls/results to save tokens, but this
        # broke the tool_call/tool_result pairing required by OpenAI and Anthropic.
        # ============================================================================

        # We need to summarize - split messages into old (to summarize) and recent (to keep)
        logger.info(
            f"Token count {total_tokens} >= limit {self.token_limit}, triggering summarization"
        )

        # Strategy: Prioritize user messages, then recent small tool results, then summarize large/old ones
        # CRITICAL: Tool results must have corresponding tool calls to avoid "tool_result without tool_use" errors

        # First pass: identify message types and count tokens per message
        user_messages: List[ModelMessage] = []
        tool_result_messages: List[ModelMessage] = []
        other_messages: List[ModelMessage] = []

        # Message metadata: (message, token_count, is_user, is_tool_call, is_tool_result, tool_call_ids)
        message_metadata: List[
            Tuple[ModelMessage, int, bool, bool, bool, Set[str]]
        ] = []

        for msg in messages:
            is_user = self._is_user_message(msg)
            is_tool_call = self._is_tool_call_message(msg)
            is_tool_result = self._is_tool_result_message(msg)
            msg_tokens = self._count_message_tokens(msg, model_name)
            tool_call_ids = self._extract_tool_call_ids_from_message(msg)

            message_metadata.append(
                (msg, msg_tokens, is_user, is_tool_call, is_tool_result, tool_call_ids)
            )

            if is_user:
                user_messages.append(msg)
            elif is_tool_result:
                tool_result_messages.append(msg)
            else:
                other_messages.append(msg)

        # Always keep all user messages (they're typically small and critical)
        # Identify the last N tool results to keep in full (newer than last N)
        recent_tool_results_to_keep = RECENT_TOOL_RESULTS_TO_KEEP
        tool_call_ids_to_keep_full: Set[str] = set()

        # Get the last N tool results (by position, not size)
        # CRITICAL: If there are fewer than N tool results, keep ALL of them
        if len(tool_result_messages) <= recent_tool_results_to_keep:
            # Keep ALL tool results - we have fewer than the limit
            for i, (msg, _, _, _, is_tool_result, tool_call_ids) in enumerate(
                message_metadata
            ):
                if is_tool_result:
                    tool_call_ids_to_keep_full.update(tool_call_ids)
            logger.debug(
                f"Keeping ALL {len(tool_result_messages)} tool results (under limit of {recent_tool_results_to_keep})"
            )
        else:
            # Get indices of the most recent tool result messages
            recent_indices = []
            for i in range(len(message_metadata) - 1, -1, -1):
                _, _, _, _, is_tool_result, _ = message_metadata[i]
                if is_tool_result:
                    recent_indices.append(i)
                    if len(recent_indices) >= recent_tool_results_to_keep:
                        break

            # Keep the most recent tool results (by position)
            for i in reversed(recent_indices):
                msg, _, _, _, _, tool_call_ids = message_metadata[i]
                tool_call_ids_to_keep_full.update(tool_call_ids)

            logger.debug(
                f"Identified {len(tool_call_ids_to_keep_full)} tool_call_ids to keep in full "
                f"from {recent_tool_results_to_keep} most recent tool results"
            )

        # Second pass: process messages and compress old tool calls/results
        # Process messages in forward order to maintain tool_use -> tool_result pairing
        messages_to_keep_list: List[ModelMessage] = []

        # Track which tool calls/results should be removed (older than last 5)
        # CRITICAL: We must remove BOTH tool calls AND their results together
        # to avoid "tool_use without tool_result" errors
        tool_call_ids_to_remove: Set[str] = set()

        # First, identify which tool results should be removed (older than last 5)
        for i, (
            msg,
            msg_tokens,
            is_user,
            is_tool_call,
            is_tool_result,
            tool_call_ids_in_msg,
        ) in enumerate(message_metadata):
            if is_tool_result:
                # Check if this is one of the recent ones to keep in full
                should_keep_full = any(
                    tid in tool_call_ids_to_keep_full for tid in tool_call_ids_in_msg
                )
                if not should_keep_full:
                    # Mark for removal (both call and result must be removed together)
                    tool_call_ids_to_remove.update(tool_call_ids_in_msg)

        # Collect metadata about removed tool calls for creating summary messages
        # Process in forward order to collect both calls and results
        removed_tool_metadata: Dict[
            str, Tuple[str, str, str]
        ] = {}  # tool_call_id -> (tool_name, tool_args, result_summary)

        # First pass: collect metadata for all tool calls/results to be removed
        for (
            msg,
            msg_tokens,
            is_user,
            is_tool_call,
            is_tool_result,
            tool_call_ids_in_msg,
        ) in message_metadata:
            should_remove = any(
                tid in tool_call_ids_to_remove for tid in tool_call_ids_in_msg
            )

            if should_remove:
                if is_tool_result:
                    result_info = self._extract_tool_info_from_message(msg)
                    if result_info:
                        tool_name, result_content, tool_call_id = result_info
                        result_summary = self._trim_tool_result_lines(result_content)
                        # Initialize or update metadata
                        if tool_call_id not in removed_tool_metadata:
                            removed_tool_metadata[tool_call_id] = (
                                tool_name,
                                "",
                                result_summary,
                            )
                        else:
                            # Update with result summary
                            name, args, _ = removed_tool_metadata[tool_call_id]
                            removed_tool_metadata[tool_call_id] = (
                                name,
                                args,
                                result_summary,
                            )
                elif is_tool_call:
                    call_info = self._extract_tool_call_info_from_message(msg)
                    if call_info:
                        tool_name, tool_args, tool_call_id = call_info
                        # Initialize or update metadata
                        if tool_call_id not in removed_tool_metadata:
                            removed_tool_metadata[tool_call_id] = (
                                tool_name,
                                tool_args,
                                "",
                            )
                        else:
                            # Update with call args
                            name, _, result_summary = removed_tool_metadata[
                                tool_call_id
                            ]
                            removed_tool_metadata[tool_call_id] = (
                                name,
                                tool_args,
                                result_summary,
                            )

        # Second pass: remove messages and keep only recent ones
        # CRITICAL: Anthropic requires tool_use to be immediately followed by tool_result
        # We must ensure consecutive pairing

        # First, mark which messages should be kept (not marked for removal)
        messages_to_keep_mask = [False] * len(message_metadata)

        # Count preserved LLM responses for logging
        preserved_llm_responses = 0

        for i, (
            msg,
            msg_tokens,
            is_user,
            is_tool_call,
            is_tool_result,
            tool_call_ids_in_msg,
        ) in enumerate(message_metadata):
            # Always keep user messages
            if is_user:
                messages_to_keep_mask[i] = True
                continue

            # CRITICAL: ALWAYS preserve LLM response messages (ModelResponse with any TextPart)
            # This is the most important rule - the LLM needs to see what it already said
            # to avoid repeating itself. This takes precedence over ALL other logic.
            if self._is_llm_response_message(msg):
                messages_to_keep_mask[i] = True
                preserved_llm_responses += 1
                logger.debug(
                    f"[History Processor] ALWAYS preserving LLM response at message {i} "
                    f"(may also have tool_call_ids: {tool_call_ids_in_msg})"
                )
                continue

            # For non-tool messages (no tool_call_ids), keep them
            if not tool_call_ids_in_msg:
                messages_to_keep_mask[i] = True
                continue

            # Check if this tool call/result should be removed (old)
            # ONLY remove pure tool call/result messages, not LLM responses
            should_remove = any(
                tid in tool_call_ids_to_remove for tid in tool_call_ids_in_msg
            )

            if should_remove:
                # Mark for removal - but only if this is a PURE tool message
                # (we already preserved LLM responses above)
                logger.debug(
                    f"[History Processor] Removing old tool message at {i}: "
                    f"is_tool_call={is_tool_call}, is_tool_result={is_tool_result}, "
                    f"tool_call_ids={tool_call_ids_in_msg}"
                )
                continue

            # Keep recent tool calls and results
            messages_to_keep_mask[i] = True

        logger.info(
            f"[History Processor] Preserved {preserved_llm_responses} LLM response messages"
        )

        # Now validate tool_use -> tool_result pairing
        # Remove tool calls that don't have results in the next message
        # Remove tool results that don't have calls in the previous message
        # CRITICAL: NEVER remove LLM responses with text content, even if tool pairing is broken
        for i, (
            msg,
            msg_tokens,
            is_user,
            is_tool_call,
            is_tool_result,
            tool_call_ids_in_msg,
        ) in enumerate(message_metadata):
            if not messages_to_keep_mask[i]:
                continue

            # CRITICAL: NEVER remove LLM responses with text content
            # This check must come BEFORE tool pairing validation
            if self._is_llm_response_message(msg):
                # This is an LLM response - keep it no matter what
                # The LLM needs to see what it already said to avoid repeating itself
                continue

            # For tool calls: must have result in next message
            # CRITICAL: Next message must be a tool_result message AND contain all tool_call_ids
            if is_tool_call and tool_call_ids_in_msg:
                has_next_result = False
                if i + 1 < len(message_metadata) and messages_to_keep_mask[i + 1]:
                    _, _, _, _, next_is_tool_result, next_tool_call_ids = (
                        message_metadata[i + 1]
                    )
                    # CRITICAL: Next message must be a tool_result message
                    if next_is_tool_result:
                        # Check if all tool_call_ids have results in next message
                        has_next_result = all(
                            tid in next_tool_call_ids for tid in tool_call_ids_in_msg
                        )
                    # If next message is not a tool_result, the tool call is orphaned

                if not has_next_result:
                    # Remove - tool call without result in next message
                    messages_to_keep_mask[i] = False
                    logger.debug(
                        f"Removing tool call without result in next message: {tool_call_ids_in_msg}"
                    )

            # For tool results: must have call in previous message
            elif is_tool_result and tool_call_ids_in_msg:
                has_prev_call = False
                if i > 0 and messages_to_keep_mask[i - 1]:
                    _, _, _, prev_is_tool_call, _, prev_tool_call_ids = (
                        message_metadata[i - 1]
                    )
                    if prev_is_tool_call:
                        # Check if all tool_call_ids have calls in previous message
                        has_prev_call = all(
                            tid in prev_tool_call_ids for tid in tool_call_ids_in_msg
                        )

                if not has_prev_call:
                    # Remove - tool result without call in previous message
                    messages_to_keep_mask[i] = False
                    logger.debug(
                        f"Removing tool result without call in previous message: {tool_call_ids_in_msg}"
                    )

        # Build the final list of messages to keep
        for i, (
            msg,
            msg_tokens,
            is_user,
            is_tool_call,
            is_tool_result,
            tool_call_ids_in_msg,
        ) in enumerate(message_metadata):
            if messages_to_keep_mask[i]:
                messages_to_keep_list.append(msg)

        messages_to_keep = messages_to_keep_list

        # Create metadata summary message for removed tool calls if any were removed
        # IMPORTANT: Do this BEFORE validation to avoid breaking tool_use/tool_result pairing
        if removed_tool_metadata:
            # Create a summary message with metadata about removed tool calls
            summary_lines = [
                "[Previous tool calls (removed from context to save tokens):]"
            ]
            for tool_call_id, (
                tool_name,
                tool_args,
                result_summary,
            ) in removed_tool_metadata.items():
                trimmed_args = self._trim_tool_args(tool_args) if tool_args else ""
                summary_lines.append(f"\n- Tool: {tool_name}")
                if trimmed_args:
                    summary_lines.append(f"  Args: {trimmed_args}")
                if result_summary:
                    # Truncate result summary to first 200 chars
                    summary_lines.append(
                        f"  Result (summary): {result_summary[:200]}..."
                    )
            summary_text = "\n".join(summary_lines)

            # Add summary as a user message to preserve context
            # CRITICAL: Insert at the BEGINNING to avoid breaking tool_use/tool_result pairing
            # Never insert between tool calls and their results
            summary_msg = ModelRequest(
                parts=[
                    SystemPromptPart(
                        content="Summary of previous tool calls that were removed from context to manage token usage."
                    ),
                    UserPromptPart(content=summary_text),
                ]
            )
            # Always insert at the beginning to be safe
            messages_to_keep.insert(0, summary_msg)
            logger.info(
                f"Created metadata summary for {len(removed_tool_metadata)} removed tool calls"
            )

        # CRITICAL: Validate and fix any orphaned tool calls/results to prevent API errors
        # Anthropic requires every tool_use to have a tool_result in the next message
        # This must happen AFTER inserting the summary to catch any issues
        messages_to_keep = self._validate_and_fix_tool_pairing(messages_to_keep)

        # Count tokens in messages to keep
        keep_tokens = self._count_total_context_tokens(
            ctx, messages_to_keep, model_name
        )

        # Count how many messages have meaningful text content (assistant responses)
        preserved_text_messages = sum(
            1 for msg in messages_to_keep if self._has_meaningful_text_content(msg)
        )

        logger.info(
            f"Filtered messages: keeping {len(messages_to_keep)} messages "
            f"({keep_tokens} tokens after compression), "
            f"preserved {preserved_text_messages} assistant text messages"
        )

        # If we're still over the limit, continue compressing older messages
        if keep_tokens > self.token_limit:
            logger.info(
                f"Still over limit ({keep_tokens} > {self.token_limit}), "
                f"compressing more messages..."
            )
            # Continue compressing older tool results until we're under the limit
            messages_to_keep = self._continue_compressing_until_under_limit(
                ctx, messages_to_keep, model_name
            )
            final_tokens = self._count_total_context_tokens(
                ctx, messages_to_keep, model_name
            )
            logger.info(
                f"Final token count: {final_tokens} (target: {self.token_limit})"
            )

        # LLM summarization is commented out - just return the messages we decided to keep
        # # Check if we have a summarize agent
        # if not self.summarize_agent:
        #     logger.warning(
        #         "No summarize agent provided, falling back to keeping only recent messages"
        #     )
        #     return messages_to_keep if messages_to_keep else messages[-5:]
        #
        # # Summarize old messages
        # try:
        #     logger.info(
        #         f"Summarizing {len(messages_to_summarize)} messages "
        #         f"(keeping {len(messages_to_keep)} recent messages)"
        #     )
        #
        #     summarized_messages = await self._summarize_messages(
        #         messages_to_summarize, self.summarize_agent, model_name
        #     )
        #
        #     # Combine summary with recent messages
        #     processed_messages = summarized_messages + messages_to_keep
        #
        #     # Count tokens for processed messages
        #     new_tokens = self._count_total_context_tokens(
        #         ctx, processed_messages, model_name
        #     )
        #
        #     reduction = total_tokens - new_tokens
        #     reduction_pct = (100 * reduction / total_tokens) if total_tokens > 0 else 0
        #
        #     logger.info(
        #         f"Summarization complete: {total_tokens} -> {new_tokens} tokens "
        #         f"({reduction} saved, {reduction_pct:.1f}% reduction)"
        #     )
        #
        #     return processed_messages
        # except Exception as e:
        #     logger.error(f"Error during summarization: {e}", exc_info=True)
        #     # Fallback: return recent messages
        #     return messages_to_keep if messages_to_keep else messages[-5:]

        # Return the processed messages (with compression applied)
        final_tokens = self._count_total_context_tokens(
            ctx, messages_to_keep, model_name
        )
        reduction = total_tokens - final_tokens
        reduction_pct = (100 * reduction / total_tokens) if total_tokens > 0 else 0

        logger.info(
            f"Message processing complete: {total_tokens} -> {final_tokens} tokens "
            f"({reduction} saved, {reduction_pct:.1f}% reduction)"
        )

        # Get final messages that will be sent to LLM
        final_messages = messages_to_keep if messages_to_keep else messages[-5:]

        # Log output RunContext to debug file (only if DEBUG enabled)
        # Commented out - not needed right now
        # if logger.isEnabledFor(logging.DEBUG):
        #     self._write_tokenizer_debug_file(
        #         f"Output messages count: {len(final_messages)}\n"
        #         f"Output RunContext logged at end of processing\n"
        #         f"Token reduction: {reduction} tokens ({reduction_pct:.1f}%)",
        #         "output_runcontext",
        #         model_name,
        #         final_tokens,
        #         ctx,
        #     )

        # ALWAYS write messages to LLM debug file (unconditional) for debugging errors
        # This helps debug tool_use/tool_result pairing issues even when errors occur
        # Commented out - not needed right now
        # self._write_messages_to_debug_file(
        #     final_messages,
        #     model_name,
        #     final_tokens,
        #     ctx,
        # )

        # CRITICAL: Store compressed output for retrieval in subsequent runs within the same execution
        # Generate a key from the run context to identify this conversation
        history_key = self._get_history_key_from_context(ctx)
        if history_key:
            self._last_compressed_output[history_key] = final_messages
            logger.debug(
                f"Stored compressed output for key '{history_key}': {len(final_messages)} messages"
            )

        return final_messages


def create_history_processor(
    llm_provider,
    token_limit: int = TOKEN_LIMIT_THRESHOLD,
    target_summary_tokens: int = TARGET_SUMMARY_TOKENS,
):
    """
    Factory function to create a history processor function with a summarization agent.

    Args:
        llm_provider: ProviderService instance to create summarization agent
        token_limit: Token threshold for triggering summarization
        target_summary_tokens: Target token count for summarized history

    Returns:
        A function that can be used as a history processor (takes ctx and messages)
    """
    # Create a cheaper model for summarization (use inference config if available)
    # For now, we'll use the same provider but could be optimized to use a cheaper model
    try:
        # Try to get a cheaper model from inference config if available
        if hasattr(llm_provider, "inference_config"):
            summarize_model = llm_provider.get_pydantic_model(
                model=llm_provider.inference_config.model
            )
        else:
            # Fallback to chat model
            summarize_model = llm_provider.get_pydantic_model()

        summarize_agent = Agent(
            model=summarize_model,
            instructions="""
            You are a conversation summarizer. Your task is to condense conversation history
            while preserving critical context, key decisions, important findings, and information
            needed for continuation. Be concise but comprehensive.
            """,
            output_type=str,
        )

        logger.info("Created summarization agent for history processor")
    except Exception as e:
        logger.warning(
            f"Failed to create summarization agent: {e}. History processor will use fallback."
        )
        summarize_agent = None

    # Create the processor instance
    processor = TokenAwareHistoryProcessor(
        summarize_agent=summarize_agent,
        token_limit=token_limit,
        target_summary_tokens=target_summary_tokens,
    )

    # Return a function closure that Pydantic AI can inspect properly
    async def history_processor(
        ctx: RunContext, messages: List[ModelMessage]
    ) -> List[ModelMessage]:
        """History processor function that wraps the TokenAwareHistoryProcessor instance."""
        return await processor(ctx, messages)

    return history_processor
