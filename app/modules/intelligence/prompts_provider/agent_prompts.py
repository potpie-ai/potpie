from enum import Enum
import re
from typing import Optional, Dict, Any

from app.modules.intelligence.agents.agents.prompts.anthropic_agent_prompts import AnthropicAgentPrompts
from app.modules.intelligence.agents.agents.prompts.openai_agent_prompts import OpenAIAgentPrompts
from app.modules.intelligence.llm_provider.llm_provider_service import AgentLLMType


class AgentPromptsProvider:
    @classmethod
    def get_agent_prompt(cls, agent_id: str, agent_type: AgentLLMType, **kwargs: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Get agent prompt based on agent ID and LLM type."""
        if agent_type == AgentLLMType.CREWAI:
            prompt = AnthropicAgentPrompts.get_anthropic_agent_prompt(agent_id)
        elif agent_type == AgentLLMType.LANGCHAIN:
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
    def get_task_prompt(cls, task_id: str, agent_type: AgentLLMType, **kwargs: Dict[str, Any]) -> Optional[str]:
        """Get task prompt based on task ID and LLM type."""
        if agent_type == AgentLLMType.CREWAI:
            description = AnthropicAgentPrompts.get_anthropic_task_prompt(task_id)
        elif agent_type == AgentLLMType.LANGCHAIN:
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
