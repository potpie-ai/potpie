from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.modules.intelligence.agents.agents.prompts.anthropic_agent_prompts import (
    AnthropicAgentPrompts,
)
from app.modules.intelligence.agents.agents.prompts.openai_agent_prompts import (
    OpenAIAgentPrompts,
)
from app.modules.intelligence.llm_provider.llm_provider_service import (
    LLMProviderService,
)
from app.modules.intelligence.prompts_provider.agent_types import AgentRuntimeLLMType


class AgentPromptsProvider:
    @classmethod
    def get_agent_prompt(
        cls, agent_id: str, user_id, db: Session, **kwargs: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """Get agent prompt based on agent ID and LLM type."""
        llm_provider_service = LLMProviderService.create(db, user_id)
        preferred_llm, model_type = llm_provider_service.get_preferred_llm(user_id)
        if preferred_llm == AgentRuntimeLLMType.ANTHROPIC.value.lower():
            prompt = AnthropicAgentPrompts.get_anthropic_agent_prompt(agent_id)
        elif preferred_llm == AgentRuntimeLLMType.OPENAI.value.lower():
            prompt = OpenAIAgentPrompts.get_openai_agent_prompt(agent_id)
        else:
            return None
        if prompt and kwargs:
            # Format any template strings if kwargs provided
            backstory = prompt.get("backstory", "")
            if backstory:
                backstory = f"""{backstory}"""
                prompt["backstory"] = backstory.format(**kwargs)
        return prompt

    @classmethod
    def get_task_prompt(
        cls, task_id: str, user_id, db: Session, **kwargs: Dict[str, Any]
    ) -> Optional[str]:
        """Get task prompt based on task ID and LLM type."""
        llm_provider_service = LLMProviderService.create(db, user_id)
        preferred_llm, model_type = llm_provider_service.get_preferred_llm(user_id)
        if preferred_llm == AgentRuntimeLLMType.ANTHROPIC.value.lower():
            description = AnthropicAgentPrompts.get_anthropic_task_prompt(task_id)
        elif preferred_llm == AgentRuntimeLLMType.OPENAI.value.lower():
            description = OpenAIAgentPrompts.get_openai_task_prompt(task_id)
        else:
            return None

        if description and kwargs:
            try:
                # Format the description string using the processed kwargs\
                description = description.format(**kwargs)
            except KeyError as e:
                raise ValueError(f"Missing key in kwargs for formatting: {e}")
            except IndexError as e:
                raise ValueError(f"Positional argument index out of range: {e}")
        return description
