""" Delete this file, this is temporary planning """

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List
from pydantic import BaseModel

"""
- classification will move to agents module
- add providers for llms, repository access, prompts, chat history etc (pass them on startup)
- pass config at startup (Don't access from env for each request in place)
- agents repo for db operations, have some abstraction
"""


""" ------------------------------------------------------------- """
""" Chat Agent Response """
""" ------------------------------------------------------------- """


class ChatAgentResponse(BaseModel):
    message: str
    citations: List[str]


# This will be used in conversation APIs aswell as wrappers. All agents will implement this interface
# including supervisors and custom agents
class ChatAgent(ABC):
    """Interface for chat agents. Chat agents will be used in conversation APIs"""

    @abstractmethod
    def run(self, user_id: str, project_id: str) -> ChatAgentResponse:
        """Run synchronously in a blocking manner, return entire response at once"""
        pass

    @abstractmethod
    def run_stream(
        self, user_id: str, project_id: str
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Run asynchronously, yield response piece by piece"""
        pass


""" ------------------------------------------------------------- """
""" Agents Services """
""" ------------------------------------------------------------- """


# this can be reused in many places (including conversations etc)
class AgentsPostgresRepo:
    """Repository for agents, get list create update delete etc"""

    def __init__(self, db: Session):
        self.db = db


class ConversationMessagesPostgresRepo:
    """Repository for conversations messages, get history create update delete etc"""

    def __init__(self, db: Session):
        self.db = db


class AgentsService:
    """Service for managing system and custom agents (CRUD Operations)"""

    def __init__(self, db: Session):
        self.db = db

    def list_agents(self) -> List[ChatAgent]:
        """List all agents"""
        return []

    def get_agent(self, agent_id: str) -> ChatAgent:
        """Get an agent by id"""
        pass

    def create_agent(self, agent: ChatAgent) -> ChatAgent:
        """Create an agent"""
        pass

    def update_agent(self, agent_id: str, agent: ChatAgent) -> ChatAgent:
        """Update an agent"""
        pass

    def delete_agent(self, agent_id: str) -> None:
        """Delete an agent"""
        pass


class AgentContext(BaseModel):
    """Context for building an agent"""

    message: str
    llm_config: LLMConfig
    agent_type: AgentType


class AgentBuilderService:
    """Build agent for current context. Context will include current message, users llm config,
    current selected agent type, system/custom etc. Agent builder will respond with a ChatAgent for
    given context.
    """

    def __init__(self, db: Session):  # pass all the providers
        self.db = db

    def build_agent(self, context: AgentContext) -> ChatAgent:
        """Build an agent for the given context"""
        pass


""" ------------------------------------------------------------- """
""" Agents Controller """
""" ------------------------------------------------------------- """

""" When initializing the controller, we will pass all the services. On request we build an agent depending
on the current context and run the query """


""" ------------------------------------------------------------- """
""" ChatAgent Implementations """
""" ------------------------------------------------------------- """

"""
1. AgentBuilderService -> returns a ChatAgent for context
2. SupervisorAgent(list of chat agents with info) + AutoClassifierAgent (simple llm or supervisor):
    - Classifies the query into agent type and uses it
    - Langgraph agent
3. SimpleLLMAgent:
    - uses simple llm directly
4. CrewAIRAGAgent: (Can build langgraph agent or any other alternative later)
    - has context of codebase, full agent
    - build from runtime_agent
5. BlastRadiusAgent: (CrewAIRagAgent)
6. CodeGenerationAgent: (CrewAIRagAgent)
7. IntegrationTestAgent: (CrewAIRagAgent)
8. DebugRAGAgent: (CrewAIRagAgent)
9. LowLevelDesignAgent: (CrewAIRagAgent)
10. QueryAndAnalysisAgent: (CrewAIRagAgent)
11. CustomAgent: (CrewAIRagAgent)
"""

"""
Refer QnA agent to split work. Use Crew ai to fetch context and some llm to generate responses
"""
