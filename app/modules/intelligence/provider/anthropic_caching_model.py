"""
Custom Anthropic Model with Prompt Caching Support

This module provides an enhanced AnthropicModel that automatically adds cache_control
breakpoints to tools and system prompts, significantly improving cache hit rates.

Anthropic's prompt caching can reduce:
- Costs by up to 90% (cache reads are 10% of regular input token cost)
- Latency by up to 85% for long prompts

Cache requirements:
- Minimum cacheable tokens: 1024 (most models) or 2048 (Haiku models)
- Cache TTL: 5 minutes (refreshes on each hit)
- Up to 4 cache breakpoints per request

HOW CACHING WORKS:
- Anthropic caches content based on EXACT PREFIX MATCHING
- Cache keys are automatic - no manual key management
- Content is cached UP TO AND INCLUDING the block with cache_control
- If prefix changes, cache misses; if prefix same, cache hits

STRATEGY:
1. Tools are cached (static, same across requests for same agent)
2. System prompt is split at CACHE_BREAKPOINT_MARKER:
   - Content BEFORE marker: cached (static instructions)
   - Content AFTER marker: not cached (dynamic context)

For more info: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic_ai.models.anthropic import (
    AnthropicModel,
    AnthropicModelSettings,
    AnthropicModelName,
    AnthropicStreamedResponse,
)
from pydantic_ai.models import (
    ModelRequestParameters,
    StreamedResponse,
    get_user_agent,
    _utils,
)
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.providers import Provider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.profiles import ModelProfileSpec
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai import ModelHTTPError

try:
    from anthropic import NOT_GIVEN, APIStatusError, AsyncAnthropic, AsyncStream
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaMessage,
        BetaRawMessageStartEvent,
        BetaRawMessageStreamEvent,
        BetaTextBlockParam,
        BetaToolChoiceParam,
        BetaToolParam,
    )
except ImportError as e:
    raise ImportError(
        "Please install `anthropic` to use the Anthropic model, "
        'you can use the `anthropic` optional group — `pip install "pydantic-ai-slim[anthropic]"`'
    ) from e

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


def _sanitize_anthropic_messages(messages: list) -> list:
    """Sanitize Anthropic-format messages to prevent API errors.

    This is the LAST LINE OF DEFENSE before messages are sent to Anthropic.
    It handles:
    1. Duplicate tool_result blocks (same tool_use_id)
    2. Orphaned tool_use blocks (no matching tool_result)
    3. Preserves text content even when stripping tool_use

    Works on Anthropic's native message format:
    - Messages have "role" and "content"
    - Content is a list of blocks with "type" (text, tool_use, tool_result, etc.)
    - tool_use has "id", tool_result has "tool_use_id"
    """
    if not messages:
        return messages

    # Track all tool_use IDs and tool_result IDs
    tool_use_ids: set = set()
    tool_result_ids: set = set()
    seen_tool_result_ids: set = set()  # For detecting duplicates

    # First pass: collect all IDs
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "tool_use":
                        tool_use_ids.add(block.get("id"))
                    elif block_type == "tool_result":
                        tool_result_ids.add(block.get("tool_use_id"))

    # Find orphaned tool_use IDs (no matching tool_result)
    orphaned_tool_use_ids = tool_use_ids - tool_result_ids

    if orphaned_tool_use_ids:
        logger.warning(
            f"[Anthropic Sanitizer] Found {len(orphaned_tool_use_ids)} orphaned tool_use IDs: "
            f"{orphaned_tool_use_ids}"
        )

    # Second pass: filter messages
    sanitized_messages = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", [])

        if isinstance(content, str):
            # Plain text content - keep as-is
            sanitized_messages.append(msg)
            continue

        if not isinstance(content, list):
            sanitized_messages.append(msg)
            continue

        # Filter content blocks
        filtered_content = []
        text_preserved = []

        for block in content:
            if not isinstance(block, dict):
                filtered_content.append(block)
                continue

            block_type = block.get("type")

            if block_type == "text":
                # ALWAYS keep text blocks - this preserves LLM responses
                filtered_content.append(block)
                text_preserved.append(block.get("text", "")[:50])

            elif block_type == "tool_use":
                tool_id = block.get("id")
                # Only keep tool_use if it has a matching tool_result
                if tool_id in orphaned_tool_use_ids:
                    logger.debug(
                        f"[Anthropic Sanitizer] Stripping orphaned tool_use: {tool_id}"
                    )
                    continue
                filtered_content.append(block)

            elif block_type == "tool_result":
                tool_use_id = block.get("tool_use_id")
                # Check for duplicate tool_result
                if tool_use_id in seen_tool_result_ids:
                    logger.warning(
                        f"[Anthropic Sanitizer] Removing duplicate tool_result: {tool_use_id}"
                    )
                    continue
                seen_tool_result_ids.add(tool_use_id)
                filtered_content.append(block)

            else:
                # Keep other block types (thinking, etc.)
                filtered_content.append(block)

        # Only add message if it has content
        if filtered_content:
            sanitized_msg = dict(msg)
            sanitized_msg["content"] = filtered_content
            sanitized_messages.append(sanitized_msg)
        else:
            logger.debug(f"[Anthropic Sanitizer] Removing empty message (role={role})")

    # Final validation: ensure tool_use/tool_result pairing is correct
    # After stripping, we need to check again for orphaned tool_results
    final_tool_use_ids: set = set()
    for msg in sanitized_messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    final_tool_use_ids.add(block.get("id"))

    # Remove tool_results that no longer have matching tool_use
    final_messages = []
    for msg in sanitized_messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            final_messages.append(msg)
            continue

        filtered_content = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                if block.get("tool_use_id") not in final_tool_use_ids:
                    logger.debug(
                        f"[Anthropic Sanitizer] Removing orphaned tool_result: {block.get('tool_use_id')}"
                    )
                    continue
            filtered_content.append(block)

        if filtered_content:
            final_msg = dict(msg)
            final_msg["content"] = filtered_content
            final_messages.append(final_msg)

    return final_messages


# Cache control type for Anthropic
CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}

# Marker to separate static (cacheable) content from dynamic content in system prompts
# Place this marker in your agent instructions where the static content ends
# Content BEFORE this marker will be cached, content AFTER will not be cached
CACHE_BREAKPOINT_MARKER = "<!-- CACHE_BREAKPOINT -->"


# Debug folder for cache metrics
DEBUG_FOLDER = Path(".debug")
CACHE_METRICS_FILE = DEBUG_FOLDER / "anthropic_cache_metrics.jsonl"

# Running totals for session summary
_session_totals = {
    "total_requests": 0,
    "total_cache_read_tokens": 0,
    "total_cache_write_tokens": 0,
    "total_uncached_tokens": 0,
    "total_output_tokens": 0,
    "session_start": None,
}

# Lock for thread-safe access to _session_totals
_session_totals_lock = threading.Lock()


def _write_cache_metrics_to_file(metrics: dict) -> None:
    """Write cache metrics to a JSONL file in .debug folder."""
    try:
        DEBUG_FOLDER.mkdir(exist_ok=True)
        with open(CACHE_METRICS_FILE, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write cache metrics to file: {e}")


def _log_cache_metrics(usage_details: dict[str, int], model_name: str) -> None:
    """
    Log Anthropic cache metrics from usage details and write to .debug folder.

    Args:
        usage_details: The usage.details dict from ModelResponse
        model_name: The model name for logging context
    """
    global _session_totals

    cache_creation = usage_details.get("cache_creation_input_tokens", 0)
    cache_read = usage_details.get("cache_read_input_tokens", 0)
    input_tokens = usage_details.get("input_tokens", 0)
    output_tokens = usage_details.get("output_tokens", 0)

    total_input = input_tokens + cache_creation + cache_read

    # Update session totals with thread-safe locking
    with _session_totals_lock:
        if _session_totals["session_start"] is None:
            _session_totals["session_start"] = datetime.now().isoformat()

        _session_totals["total_requests"] += 1
        _session_totals["total_cache_read_tokens"] += cache_read
        _session_totals["total_cache_write_tokens"] += cache_creation
        _session_totals["total_uncached_tokens"] += input_tokens
        _session_totals["total_output_tokens"] += output_tokens

        # Read current values for metrics record (while holding lock)
        current_totals = {
            "total_requests": _session_totals["total_requests"],
            "total_cache_read_tokens": _session_totals["total_cache_read_tokens"],
            "total_cache_write_tokens": _session_totals["total_cache_write_tokens"],
            "total_uncached_tokens": _session_totals["total_uncached_tokens"],
            "total_output_tokens": _session_totals["total_output_tokens"],
        }

    if total_input > 0:
        # Calculate cache hit rate
        cache_hit_rate = (cache_read / total_input * 100) if total_input > 0 else 0
        cache_write_rate = (
            (cache_creation / total_input * 100) if total_input > 0 else 0
        )
        uncached_rate = (input_tokens / total_input * 100) if total_input > 0 else 0

        # Calculate cost analysis
        # Without cache: all tokens at full price (100%)
        # With cache:
        #   - cache_read at 10% price (90% discount)
        #   - cache_creation at 125% price (25% extra for 5min TTL)
        #   - uncached at 100% price
        tokens_without_cache = total_input
        effective_tokens_with_cache = (
            input_tokens + (cache_creation * 1.25) + (cache_read * 0.1)
        )
        savings_percent = (
            (
                (tokens_without_cache - effective_tokens_with_cache)
                / tokens_without_cache
                * 100
            )
            if tokens_without_cache > 0
            else 0
        )

        # Determine cache status
        if cache_read > 0 and cache_creation == 0:
            cache_status = "HIT"
        elif cache_creation > 0 and cache_read == 0:
            cache_status = "WRITE"
        elif cache_creation > 0 and cache_read > 0:
            cache_status = "PARTIAL_HIT"
        else:
            cache_status = "MISS"

        # Build detailed metrics record
        metrics_record = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "cache_status": cache_status,
            "tokens": {
                "cache_read": cache_read,
                "cache_write": cache_creation,
                "uncached": input_tokens,
                "output": output_tokens,
                "total_input": total_input,
            },
            "percentages": {
                "cache_hit_rate": round(cache_hit_rate, 2),
                "cache_write_rate": round(cache_write_rate, 2),
                "uncached_rate": round(uncached_rate, 2),
            },
            "cost_analysis": {
                "tokens_without_cache": tokens_without_cache,
                "effective_tokens_with_cache": round(effective_tokens_with_cache, 2),
                "savings_percent": round(savings_percent, 2),
                "explanation": {
                    "cache_read_cost": "10% of base (90% discount)",
                    "cache_write_cost": "125% of base (25% extra for 5min TTL)",
                    "uncached_cost": "100% of base",
                },
            },
            "session_totals": {
                "requests": current_totals["total_requests"],
                "cumulative_cache_read": current_totals["total_cache_read_tokens"],
                "cumulative_cache_write": current_totals["total_cache_write_tokens"],
                "cumulative_uncached": current_totals["total_uncached_tokens"],
                "cumulative_output": current_totals["total_output_tokens"],
            },
        }

        # Calculate cumulative session savings
        session_total_input = (
            current_totals["total_cache_read_tokens"]
            + current_totals["total_cache_write_tokens"]
            + current_totals["total_uncached_tokens"]
        )
        if session_total_input > 0:
            session_effective = (
                current_totals["total_uncached_tokens"]
                + (current_totals["total_cache_write_tokens"] * 1.25)
                + (current_totals["total_cache_read_tokens"] * 0.1)
            )
            session_savings = (
                (session_total_input - session_effective) / session_total_input * 100
            )
            metrics_record["session_totals"]["cumulative_savings_percent"] = round(
                session_savings, 2
            )

        # Write to file
        _write_cache_metrics_to_file(metrics_record)

        # Log to console
        logger.info(
            f"📊 Anthropic Cache [{cache_status}] [{model_name}]: "
            f"read={cache_read:,} ({cache_hit_rate:.1f}%), "
            f"write={cache_creation:,} ({cache_write_rate:.1f}%), "
            f"uncached={input_tokens:,} ({uncached_rate:.1f}%), "
            f"output={output_tokens:,}, "
            f"savings≈{savings_percent:.1f}%"
        )

        # Log session summary periodically
        if current_totals["total_requests"] % 5 == 0:
            logger.info(
                f"📈 Session Summary (requests={current_totals['total_requests']}): "
                f"cumulative_savings≈{metrics_record['session_totals'].get('cumulative_savings_percent', 0):.1f}%"
            )
    else:
        # Still write a record for no-cache requests
        metrics_record = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "cache_status": "NO_DATA",
            "tokens": {"total_input": 0, "output": output_tokens},
        }
        _write_cache_metrics_to_file(metrics_record)
        logger.debug(f"No cache metrics available for {model_name}")


@dataclass(init=False)
class CachingAnthropicModel(AnthropicModel):
    """
    An enhanced AnthropicModel that automatically enables prompt caching.

    This model adds cache_control breakpoints to:
    1. The last tool in the tools list (caches all tool definitions)
    2. The system prompt (caches instructions)

    This significantly improves cache hit rates for multi-turn conversations
    and repeated requests with the same tools/prompts.

    Usage:
        model = CachingAnthropicModel(
            model_name="claude-sonnet-4-20250514",
            provider=AnthropicProvider(api_key=api_key)
        )
    """

    _enable_tool_caching: bool = field(default=True, repr=False)
    _enable_system_caching: bool = field(default=True, repr=False)
    _cache_ttl: Literal["5m", "1h"] = field(default="5m", repr=False)

    def __init__(
        self,
        model_name: AnthropicModelName,
        *,
        provider: Literal["anthropic"] | Provider[AsyncAnthropic] = "anthropic",
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
        enable_tool_caching: bool = True,
        enable_system_caching: bool = True,
        cache_ttl: Literal["5m", "1h"] = "5m",
    ):
        """
        Initialize a caching-enabled Anthropic model.

        Args:
            model_name: The name of the Anthropic model to use.
            provider: The provider to use for the Anthropic API.
            profile: The model profile to use.
            settings: Default model settings for this model instance.
            enable_tool_caching: If True, adds cache_control to the last tool.
            enable_system_caching: If True, adds cache_control to the system prompt.
            cache_ttl: Cache time-to-live ("5m" or "1h"). Default is "5m".
        """
        super().__init__(
            model_name=model_name,
            provider=provider,
            profile=profile,
            settings=settings,
        )
        self._enable_tool_caching = enable_tool_caching
        self._enable_system_caching = enable_system_caching
        self._cache_ttl = cache_ttl

    def _process_response(self, response: BetaMessage) -> ModelResponse:
        """
        Override to log cache metrics after processing each response.
        """
        # Call parent implementation to process the response
        model_response = super()._process_response(response)

        # Log cache metrics from usage details
        if model_response.usage and model_response.usage.details:
            _log_cache_metrics(model_response.usage.details, str(self._model_name))

        return model_response

    async def _process_streamed_response(
        self, response: AsyncStream[BetaRawMessageStreamEvent]
    ) -> StreamedResponse:
        """
        Override to log cache metrics from streaming responses.

        The usage data is available in the first event (BetaRawMessageStartEvent).
        We peek at it, extract usage, log it, then process normally.
        """
        from datetime import datetime, timezone

        # Create peekable stream to access first event (parent also does this)
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()

        # Extract usage data from first event if it's a start event
        if not isinstance(first_chunk, _utils.Unset) and isinstance(
            first_chunk, BetaRawMessageStartEvent
        ):
            # Extract usage from the start event
            if hasattr(first_chunk, "message") and hasattr(
                first_chunk.message, "usage"
            ):
                usage_obj = first_chunk.message.usage
                # Convert usage to dict format for logging
                usage_dict = {}
                if hasattr(usage_obj, "model_dump"):
                    usage_dict = usage_obj.model_dump()
                elif hasattr(usage_obj, "dict"):
                    usage_dict = usage_obj.dict()
                elif isinstance(usage_obj, dict):
                    usage_dict = usage_obj

                # Extract integer values for cache metrics
                usage_details = {
                    key: value
                    for key, value in usage_dict.items()
                    if isinstance(value, int)
                }

                if usage_details:
                    _log_cache_metrics(usage_details, str(self._model_name))

        # Process the streamed response normally (same as parent)
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior(
                "Streamed response ended without content or tool calls"
            )

        # Since Anthropic doesn't provide a timestamp in the message, we'll use the current time
        timestamp = datetime.now(tz=timezone.utc)
        return AnthropicStreamedResponse(
            _model_name=self._model_name,
            _response=peekable_response,
            _timestamp=timestamp,
        )

    def _get_tools(
        self,
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> list[BetaToolParam]:
        """
        Override to add cache_control to the last tool, enabling caching of all tool definitions.

        Anthropic caches all content UP TO AND INCLUDING the block with cache_control.
        By placing cache_control on the last tool, we cache all tools.
        """
        # Ensure model_request_parameters is a ModelRequestParameters object, not a dict
        # This can happen if pydantic-ai passes it as a dict in some cases
        if isinstance(model_request_parameters, dict):
            model_request_parameters = ModelRequestParameters(
                **model_request_parameters
            )

        tools = super()._get_tools(model_settings, model_request_parameters)

        if tools and self._enable_tool_caching:
            # Add cache_control to the last tool to cache all tools
            cache_control: BetaCacheControlEphemeralParam = {"type": "ephemeral"}
            if self._cache_ttl != "5m":
                cache_control["ttl"] = self._cache_ttl
            tools[-1]["cache_control"] = cache_control
            logger.debug(
                f"Added cache_control to last tool ({tools[-1]['name']}), "
                f"total tools: {len(tools)}"
            )

        return tools

    async def _messages_create(  # type: ignore[override]
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaMessage | AsyncStream[BetaRawMessageStreamEvent]:
        """
        Override to add cache_control to the system prompt for caching.

        This method adds a cache_control breakpoint to the system prompt,
        enabling Anthropic to cache the instructions across requests.
        """
        # Ensure model_request_parameters is a ModelRequestParameters object, not a dict
        # This can happen if pydantic-ai passes it as a dict in some cases
        if isinstance(model_request_parameters, dict):
            model_request_parameters = ModelRequestParameters(
                **model_request_parameters
            )

        tools = self._get_tools(model_settings, model_request_parameters)
        tool_choice: BetaToolChoiceParam | None

        if not tools:
            tool_choice = None
        else:
            if not model_request_parameters.allow_text_output:
                tool_choice = {"type": "any"}
            else:
                tool_choice = {"type": "auto"}

            if (
                allow_parallel_tool_calls := model_settings.get("parallel_tool_calls")
            ) is not None:
                tool_choice["disable_parallel_tool_use"] = not allow_parallel_tool_calls

        system_prompt, anthropic_messages = await self._map_message(messages)

        # CRITICAL: Sanitize messages before sending to API
        # This is the last line of defense against duplicate tool_results and orphaned tool_use
        anthropic_messages = _sanitize_anthropic_messages(anthropic_messages)

        # Prepare system prompt with intelligent cache_control placement
        system_param: Any = NOT_GIVEN
        if system_prompt:
            if self._enable_system_caching:
                cache_control: BetaCacheControlEphemeralParam = {"type": "ephemeral"}
                if self._cache_ttl != "5m":
                    cache_control["ttl"] = self._cache_ttl

                # Check if system prompt contains the cache breakpoint marker
                if CACHE_BREAKPOINT_MARKER in system_prompt:
                    # Split at marker: cache static part, don't cache dynamic part
                    parts = system_prompt.split(CACHE_BREAKPOINT_MARKER, 1)
                    static_content = parts[0].strip()
                    dynamic_content = parts[1].strip() if len(parts) > 1 else ""

                    system_blocks: list[BetaTextBlockParam] = []

                    if static_content:
                        # Static content gets cache_control
                        static_block: BetaTextBlockParam = {
                            "type": "text",
                            "text": static_content,
                            "cache_control": cache_control,
                        }
                        system_blocks.append(static_block)

                    if dynamic_content:
                        # Dynamic content does NOT get cache_control
                        dynamic_block: BetaTextBlockParam = {
                            "type": "text",
                            "text": dynamic_content,
                        }
                        system_blocks.append(dynamic_block)

                    system_param = system_blocks
                    logger.debug(
                        f"Split system prompt at CACHE_BREAKPOINT_MARKER: "
                        f"static={len(static_content)} chars (cached), "
                        f"dynamic={len(dynamic_content)} chars (not cached)"
                    )
                else:
                    # No marker found - cache entire system prompt (legacy behavior)
                    # This works well for single-conversation multi-turn where prompt is static
                    system_block: BetaTextBlockParam = {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": cache_control,
                    }
                    system_param = [system_block]
                    logger.debug(
                        f"Added cache_control to entire system prompt "
                        f"(length: {len(system_prompt)} chars). "
                        f"Tip: Use CACHE_BREAKPOINT_MARKER to split static/dynamic content."
                    )
            else:
                system_param = system_prompt

        try:
            extra_headers = model_settings.get("extra_headers", {})
            extra_headers.setdefault("User-Agent", get_user_agent())

            # Type ignores match pydantic-ai's base implementation pattern
            # The NOT_GIVEN pattern is standard in anthropic SDK but doesn't type-check cleanly
            return await self.client.beta.messages.create(  # pyright: ignore[reportArgumentType]
                max_tokens=model_settings.get("max_tokens", 4096),
                system=system_param,
                messages=anthropic_messages,
                model=self._model_name,
                tools=tools or NOT_GIVEN,
                tool_choice=tool_choice or NOT_GIVEN,
                stream=stream,
                thinking=model_settings.get("anthropic_thinking", NOT_GIVEN),
                stop_sequences=model_settings.get("stop_sequences", NOT_GIVEN),
                temperature=model_settings.get("temperature", NOT_GIVEN),
                top_p=model_settings.get("top_p", NOT_GIVEN),
                timeout=model_settings.get("timeout", NOT_GIVEN),
                metadata=model_settings.get("anthropic_metadata", NOT_GIVEN),
                extra_headers=extra_headers,
                extra_body=model_settings.get("extra_body"),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(
                    status_code=status_code, model_name=self.model_name, body=e.body
                ) from e
            raise  # pragma: no cover
