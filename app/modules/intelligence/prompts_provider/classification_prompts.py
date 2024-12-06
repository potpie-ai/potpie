from enum import Enum

from pydantic import BaseModel

from app.modules.intelligence.agents.chat_agents.classification_prompts.anthropic_classification_prompts import (
    AnthropicClassificationPrompts,
)
from app.modules.intelligence.agents.chat_agents.classification_prompts.openai_classification_prompts import (
    OpenAIClassificationPrompts,
)
from app.modules.intelligence.llm_provider.llm_provider_service import AgentLLMType
from app.modules.intelligence.prompts_provider.agent_types import SystemAgentType


class ClassificationResult(Enum):
    LLM_SUFFICIENT = "LLM_SUFFICIENT"
    AGENT_REQUIRED = "AGENT_REQUIRED"


class ClassificationResponse(BaseModel):
    classification: ClassificationResult


class ClassificationPromptsProvider:
    @classmethod
    def get_classification_prompt(
        cls, agent_type: AgentLLMType, system_agent_type: SystemAgentType
    ) -> str:
        if agent_type == AgentLLMType.CREWAI:
            return AnthropicClassificationPrompts.get_anthropic_classification_prompt(
                system_agent_type
            )
        elif agent_type == AgentLLMType.LANGCHAIN:
            return OpenAIClassificationPrompts.get_openai_classification_prompt(
                system_agent_type
            )
        return ""
