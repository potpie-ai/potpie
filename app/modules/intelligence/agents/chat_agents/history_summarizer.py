"""
Phase 4: Optional LLM summarization tier for history.

When context is still over the token limit after compaction, the history processor
can replace a middle segment of messages with an LLM-generated summary (head and
tail preserved). This module provides the summarizer abstraction and implementations.
"""

import json
import logging
from typing import List, Optional, Protocol

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

logger = logging.getLogger(__name__)

# Default prompt template for summarization (preserves key decisions, tool names/outcomes)
DEFAULT_SUMMARY_PROMPT_TEMPLATE = """Summarize the following conversation history, preserving:
1. Key decisions and conclusions
2. Important context and findings
3. Critical information needed for continuation
4. Any unresolved questions or pending tasks
5. Tool calls that were made and their outcomes (tool name + key finding, not full output)

Omit:
- Repetitive content
- Verbose explanations that can be condensed
- Full tool result content (summarize what tools did, not their full output)
- Unnecessary details

For tool calls/results: include tool name, what it was used for, key findings; exclude full file contents and long snippets.

Keep the summary concise but comprehensive enough to maintain context. Target length: approximately {target_tokens} tokens.

Conversation history to summarize:

{history_text}
"""


def _serialize_messages_to_text(messages: List[ModelMessage]) -> str:
    """Serialize messages to readable text for the summarization prompt.

    Mirrors the logic used in history_processor for consistency. Does not mutate input.
    """
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


class HistorySummarizer(Protocol):
    """Protocol for history summarization.

    Implementations may use an LLM (LLMHistorySummarizer) or be a no-op (NoOpHistorySummarizer).
    The history processor invokes the summarizer only when compaction is insufficient and
    summarization is enabled.
    """

    async def summarize(
        self,
        messages: List[ModelMessage],
        model_name: Optional[str] = None,
        target_tokens: Optional[int] = None,
    ) -> List[ModelMessage]:
        """Return a short list of messages (typically one) that summarize the input.

        Must not mutate the input list. Caller may pass model_name and target_tokens
        for logging or model-specific behavior.

        Args:
            messages: The middle segment of history to summarize.
            model_name: Optional model identifier (e.g. for tokenizer or logging).
            target_tokens: Optional target length for the summary in tokens.

        Returns:
            List of messages representing the summary (usually a single ModelRequest).
        """
        ...


class NoOpHistorySummarizer:
    """Summarizer that returns the input unchanged.

    Used when summarization is disabled or as a safe fallback. Does not call the LLM.
    """

    async def summarize(
        self,
        messages: List[ModelMessage],
        model_name: Optional[str] = None,
        target_tokens: Optional[int] = None,
    ) -> List[ModelMessage]:
        """Return a copy of the input list (no summarization)."""
        return list(messages)


class LLMHistorySummarizer:
    """Summarizer that uses an LLM (via Pydantic AI Agent) to condense the middle segment.

    Serializes messages to text, builds a prompt from a template, calls the agent,
    and wraps the result in a single ModelRequest (system + user part) for insertion
    into the history as head + [summary] + tail.
    """

    def __init__(
        self,
        agent: Agent,
        prompt_template: str = DEFAULT_SUMMARY_PROMPT_TEMPLATE,
        target_tokens: int = 10_000,
    ):
        """
        Args:
            agent: Pydantic AI Agent used for summarization (e.g. same or cheaper model).
            prompt_template: Format string with placeholders {history_text} and {target_tokens}.
            target_tokens: Default target length for the summary in tokens.
        """
        self._agent = agent
        self._prompt_template = prompt_template
        self._target_tokens = target_tokens

    async def summarize(
        self,
        messages: List[ModelMessage],
        model_name: Optional[str] = None,
        target_tokens: Optional[int] = None,
    ) -> List[ModelMessage]:
        """Summarize the given messages via the LLM and return a single summary message."""
        if not messages:
            return []

        target = target_tokens if target_tokens is not None else self._target_tokens
        history_text = _serialize_messages_to_text(messages)

        prompt = self._prompt_template.format(
            history_text=history_text,
            target_tokens=target,
        )

        try:
            result = await self._agent.run(prompt)
            summary_content = (
                result.output if hasattr(result, "output") else str(result)
            )
        except Exception as e:
            logger.warning(
                "LLM summarization run failed: %s. Re-raising for processor fallback.",
                e,
            )
            raise

        summary_message = ModelRequest(
            parts=[
                SystemPromptPart(
                    content="Summary of previous conversation history to preserve context while reducing token usage."
                ),
                UserPromptPart(content=f"## Conversation Summary\n\n{summary_content}"),
            ]
        )
        return [summary_message]


def get_history_summarizer(
    llm_provider,
    summarization_model: Optional[str] = None,
    target_tokens: int = 10_000,
) -> HistorySummarizer:
    """Build a HistorySummarizer for the history processor.

    When CONTEXT_MANAGEMENT_SUMMARIZATION_ENABLED is True and a model is available,
    returns an LLMHistorySummarizer. Otherwise returns a NoOpHistorySummarizer so the
    processor can always call summarize() without checking for None.

    Args:
        llm_provider: ProviderService (or similar) used to create the summarization agent.
        summarization_model: Optional model string; when unset, factory uses same/cheaper model.
        target_tokens: Target token count for the summary (default from config).

    Returns:
        A HistorySummarizer instance (never None).
    """
    from app.modules.intelligence.agents.context_config import (
        CONTEXT_MANAGEMENT_SUMMARIZATION_ENABLED,
    )

    if not CONTEXT_MANAGEMENT_SUMMARIZATION_ENABLED:
        logger.debug(
            "Summarization disabled (CONTEXT_MANAGEMENT_SUMMARIZATION_ENABLED=false). "
            "Using NoOpHistorySummarizer."
        )
        return NoOpHistorySummarizer()

    try:
        if summarization_model and hasattr(llm_provider, "get_pydantic_model"):
            model = llm_provider.get_pydantic_model(model=summarization_model)
        elif (
            hasattr(llm_provider, "inference_config") and llm_provider.inference_config
        ):
            model = llm_provider.get_pydantic_model(
                model=llm_provider.inference_config.model
            )
        elif hasattr(llm_provider, "get_pydantic_model"):
            model = llm_provider.get_pydantic_model()
        else:
            logger.warning(
                "Cannot get model for summarization; using NoOpHistorySummarizer."
            )
            return NoOpHistorySummarizer()

        agent = Agent(
            model=model,
            instructions=(
                "You are a conversation summarizer. Condense conversation history "
                "while preserving critical context, key decisions, important findings, "
                "and information needed for continuation. Be concise but comprehensive."
            ),
            output_type=str,
        )
        summarizer = LLMHistorySummarizer(
            agent=agent,
            prompt_template=DEFAULT_SUMMARY_PROMPT_TEMPLATE,
            target_tokens=target_tokens,
        )
        logger.info(
            "Created LLMHistorySummarizer for history processor (model=%s)",
            summarization_model or getattr(model, "model_id", "default"),
        )
        return summarizer
    except Exception as e:
        logger.warning(
            "Failed to create LLM summarizer: %s. Using NoOpHistorySummarizer.",
            e,
        )
        return NoOpHistorySummarizer()
