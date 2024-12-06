from enum import Enum
from typing import Dict

from pydantic import BaseModel
from app.modules.intelligence.llm_provider.llm_provider_service import AgentType
from app.modules.intelligence.prompts.classification_prompts.anthropic_classification_prompts import AnthropicClassificationPrompts
from app.modules.intelligence.prompts.classification_prompts.openai_classification_prompts import OpenAIClassificationPrompts

class SystemAgentType(Enum):
    QNA = "QNA_AGENT"
    DEBUGGING = "DEBUGGING_AGENT"
    UNIT_TEST = "UNIT_TEST_AGENT"
    INTEGRATION_TEST = "INTEGRATION_TEST_AGENT"
    CODE_CHANGES = "CODE_CHANGES_AGENT"
    LLD = "LLD_AGENT"


class ClassificationResult(Enum):
    LLM_SUFFICIENT = "LLM_SUFFICIENT"
    AGENT_REQUIRED = "AGENT_REQUIRED"


class ClassificationResponse(BaseModel):
    classification: ClassificationResult


class ClassificationPrompts:

    @classmethod
    def get_classification_prompt(cls, agent_type: AgentType, system_agent_type: SystemAgentType) -> str:
        if agent_type == AgentType.CREWAI:
            return AnthropicClassificationPrompts.get_anthropic_classification_prompt(system_agent_type)
        elif agent_type == AgentType.LANGCHAIN:
            return OpenAIClassificationPrompts.get_openai_classification_prompt(system_agent_type)
        return ""
