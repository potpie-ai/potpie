"""
History processor for managing token usage in Pydantic AI agents.

Always truncates older tool result content (keeping the last RECENT_TOOL_RESULTS_TO_KEEP
in full), preserving system messages, tool schemas, user input, message order,
and all model text responses. Only tool result bodies are replaced with a truncation notice.

Previous conversation turns are passed as plain text via message_history (built by
prepare_multimodal_message_history). This processor only runs WITHIN a single agent
run, truncating older tool results that accumulated during that run.

Stateless: no in-memory caches; config is captured in the closure at factory time.
"""

import json
import logging
import re
from typing import TYPE_CHECKING, Callable, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from app.modules.intelligence.agents.chat_agents.history_summarizer import (
        HistorySummarizer,
    )

from pydantic_ai import Agent, RunContext

from app.modules.intelligence.agents.chat_agents.token_utils import (
    count_tokens as shared_count_tokens,
)
from app.modules.intelligence.agents.chat_agents.message_helpers import (
    extract_system_prompt_from_messages,
    extract_tool_call_ids,
    is_tool_result_message,
    serialize_messages_to_text,
)
from app.modules.intelligence.agents.chat_agents.message_compressor import (
    truncate_tool_result_message,
    validate_and_fix_tool_pairing,
)
from pydantic_ai.messages import ModelMessage

logger = logging.getLogger(__name__)

# Token thresholds
TOKEN_LIMIT_THRESHOLD = 35000
TARGET_SUMMARY_TOKENS = 10000
RECENT_TOOL_RESULTS_TO_KEEP = 7


def _get_model_name_from_context(ctx: RunContext) -> Optional[str]:
    """Extract model name from run context for token encoding; pure, no state."""
    try:
        if hasattr(ctx, "model") and ctx.model:
            model = ctx.model
            if hasattr(model, "model_name"):
                return getattr(model, "model_name", None)
            model_str = str(model)
            match = re.search(
                r"(gpt-[0-9.]+|claude-[0-9.]+|gemini-[0-9.]+)",
                model_str,
                re.IGNORECASE,
            )
            if match:
                return match.group(1)
    except Exception as e:
        logger.debug("Failed to extract model name from context: %s", e)
    return None


def _get_agent_from_context(ctx: RunContext) -> Optional[Agent]:
    """Extract agent from run context; pure, no state."""
    try:
        if hasattr(ctx, "agent"):
            agent = getattr(ctx, "agent", None)
            if agent:
                return agent
        if hasattr(ctx, "_agent"):
            agent = getattr(ctx, "_agent", None)
            if agent:
                return agent
        if hasattr(ctx, "run_state"):
            run_state = getattr(ctx, "run_state", None)
            if run_state and hasattr(run_state, "agent"):
                agent = getattr(run_state, "agent", None)
                if agent:
                    return agent
    except Exception as e:
        logger.debug("Failed to extract agent from context: %s", e)
    return None


def _extract_agent_info(agent: Agent) -> Tuple[str, str]:
    """Extract (instructions, tool_schemas) from agent; pure, no cache."""
    instructions = ""
    tool_schemas = ""
    try:
        if hasattr(agent, "instructions"):
            instructions = str(getattr(agent, "instructions", ""))
        elif hasattr(agent, "_instructions"):
            instructions = str(getattr(agent, "_instructions", ""))
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
        logger.debug("Failed to extract agent info: %s", e)
    return (instructions, tool_schemas)


def _count_total_context_tokens(
    ctx: RunContext,
    messages: List[ModelMessage],
    model_name: Optional[str] = None,
) -> int:
    """Count tokens for system prompt + tool schemas + messages; pure, no state."""
    total = 0
    agent = _get_agent_from_context(ctx)
    system_prompt_text = ""
    tool_schemas_text = ""
    if agent:
        instructions, tool_schemas_text = _extract_agent_info(agent)
        system_prompt_text = instructions
    if not system_prompt_text:
        system_prompt_text = extract_system_prompt_from_messages(messages)
    if system_prompt_text:
        total += shared_count_tokens(system_prompt_text, model_name)
    if tool_schemas_text:
        total += shared_count_tokens(tool_schemas_text, model_name)
    message_text = serialize_messages_to_text(messages)
    total += shared_count_tokens(message_text, model_name)
    return total


def _apply_truncation(messages: List[ModelMessage]) -> List[ModelMessage]:
    """
    Always truncate older tool results: keep last RECENT_TOOL_RESULTS_TO_KEEP in full,
    replace older tool result content with truncation notice. Preserves order and all
    non-tool-result messages.
    """
    # Reverse-scan: collect tool_call_ids of the most recent RECENT_TOOL_RESULTS_TO_KEEP
    protected_tool_call_ids: Set[str] = set()
    seen_result_count = 0
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if is_tool_result_message(msg):
            seen_result_count += 1
            if seen_result_count <= RECENT_TOOL_RESULTS_TO_KEEP:
                protected_tool_call_ids.update(extract_tool_call_ids(msg))
            else:
                break

    # Forward-pass: keep all messages; replace only old tool result content
    rebuilt: List[ModelMessage] = []
    for msg in messages:
        if not is_tool_result_message(msg):
            rebuilt.append(msg)
            continue
        ids_in_msg = extract_tool_call_ids(msg)
        if ids_in_msg <= protected_tool_call_ids:
            rebuilt.append(msg)
        else:
            rebuilt.append(truncate_tool_result_message(msg))

    return validate_and_fix_tool_pairing(rebuilt)


def create_history_processor(
    llm_provider,
    token_limit: int = TOKEN_LIMIT_THRESHOLD,
    target_summary_tokens: int = TARGET_SUMMARY_TOKENS,
):
    """
    Factory to create a stateless history processor (async ctx, messages -> messages).
    Always truncates older tool results; optionally runs Phase 4 summarization when
    still over limit and summarizer is configured.
    """
    from app.modules.intelligence.agents.context_config import (
        CONTEXT_MANAGEMENT_SUMMARIZATION_ENABLED,
        CONTEXT_MANAGEMENT_SUMMARIZATION_MODEL,
        HISTORY_TOKEN_BUDGET_RATIO,
        SUMMARIZATION_HEAD_MESSAGES,
        SUMMARIZATION_TAIL_MESSAGES,
        SUMMARIZATION_TARGET_TOKENS,
        get_history_token_budget,
    )
    from app.modules.intelligence.agents.chat_agents.history_summarizer import (
        NoOpHistorySummarizer,
        get_history_summarizer,
    )
    from app.modules.intelligence.provider.llm_config import get_context_window

    model_string = None
    if hasattr(llm_provider, "chat_config") and llm_provider.chat_config:
        model_string = getattr(llm_provider.chat_config, "model", None)
    effective_token_limit = (
        get_history_token_budget(model_string) if model_string else token_limit
    )
    context_window_resolver: Callable[[str], Optional[int]] = get_context_window
    history_summarizer: "HistorySummarizer" = get_history_summarizer(
        llm_provider,
        summarization_model=CONTEXT_MANAGEMENT_SUMMARIZATION_MODEL,
        target_tokens=target_summary_tokens or SUMMARIZATION_TARGET_TOKENS,
    )
    summarization_head_messages = max(0, SUMMARIZATION_HEAD_MESSAGES)
    summarization_tail_messages = max(0, SUMMARIZATION_TAIL_MESSAGES)
    target_summary_tokens_resolved = target_summary_tokens or SUMMARIZATION_TARGET_TOKENS

    async def history_processor(
        ctx: RunContext, messages: List[ModelMessage]
    ) -> List[ModelMessage]:
        # 1–3: Always truncate older tool results and validate pairing
        rebuilt = _apply_truncation(messages)
        final_messages = rebuilt

        # 4: Only if summarizer is configured and not no-op, count tokens and maybe summarize
        if history_summarizer is None or isinstance(
            history_summarizer, NoOpHistorySummarizer
        ):
            return final_messages
        if not CONTEXT_MANAGEMENT_SUMMARIZATION_ENABLED:
            return final_messages

        model_name = _get_model_name_from_context(ctx)
        effective_limit = effective_token_limit
        if model_name and context_window_resolver:
            context_window = context_window_resolver(model_name)
            if context_window is not None and context_window > 0:
                effective_limit = min(
                    effective_token_limit,
                    int(context_window * HISTORY_TOKEN_BUDGET_RATIO),
                )

        final_tokens = _count_total_context_tokens(ctx, rebuilt, model_name)
        logger.info(
            "[History Processor] After truncation: %s messages, %s tokens%s",
            len(rebuilt),
            final_tokens,
            f", model: {model_name}" if model_name else "",
        )

        if final_tokens <= effective_limit:
            return final_messages

        if len(rebuilt) <= summarization_head_messages + summarization_tail_messages:
            return final_messages

        head = rebuilt[:summarization_head_messages]
        tail = rebuilt[-summarization_tail_messages:]
        middle = rebuilt[summarization_head_messages:-summarization_tail_messages]
        logger.info(
            "Summarization triggered: over limit after truncation (before=%s, limit=%s)",
            final_tokens,
            effective_limit,
        )
        try:
            summarized_middle = await history_summarizer.summarize(
                middle,
                model_name=model_name,
                target_tokens=target_summary_tokens_resolved,
            )
            final_messages = head + summarized_middle + tail
            final_messages = validate_and_fix_tool_pairing(final_messages)
            final_tokens = _count_total_context_tokens(
                ctx, final_messages, model_name
            )
            logger.info(
                "Summarization complete: head=%s, middle=%s, tail=%s, summary_messages=%s, tokens_after=%s",
                len(head),
                len(middle),
                len(tail),
                len(summarized_middle),
                final_tokens,
            )
        except Exception as e:
            logger.warning(
                "Summarization failed, using truncated history: %s", e
            )
            final_messages = rebuilt

        return final_messages

    return history_processor
