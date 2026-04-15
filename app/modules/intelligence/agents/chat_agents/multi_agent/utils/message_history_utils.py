"""Message history utility functions for multi-agent system"""

from typing import List, Optional, Any
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from app.modules.intelligence.agents.chat_agents.message_compressor import (
    sanitize_message_history_for_pydantic_ai,
)
from app.modules.intelligence.agents.chat_agents.token_utils import count_tokens
from app.modules.intelligence.agents.context_config import (
    get_history_token_budget,
)
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


def validate_and_fix_message_history(
    messages: List[ModelMessage],
) -> List[ModelMessage]:
    """Validate message history for PydanticAI: tool pairing and ends with ModelRequest.

    Uses shared sanitize (message_compressor). Applies to initial history passed to iter().
    """
    return sanitize_message_history_for_pydantic_ai(messages)


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
