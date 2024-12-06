from enum import Enum


class SystemAgentType(Enum):
    QNA = "QNA_AGENT"
    DEBUGGING = "DEBUGGING_AGENT"
    UNIT_TEST = "UNIT_TEST_AGENT"
    INTEGRATION_TEST = "INTEGRATION_TEST_AGENT"
    CODE_CHANGES = "CODE_CHANGES_AGENT"
    LLD = "LLD_AGENT"


class AgentLLMType(Enum):
    CREWAI = "CREWAI"
    LANGCHAIN = "LANGCHAIN"


class AgentRuntimeLLMType(Enum):
    ANTHROPIC = "ANTHROPIC"
    OPENAI = "OPENAI"
