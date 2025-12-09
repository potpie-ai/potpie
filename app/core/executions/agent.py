from abc import ABC, abstractmethod
from pydantic import BaseModel


class AgentResponse(BaseModel):
    text: str
    conversation_id: str


class AgentQueryExecutor(ABC):
    """Abstract adapter for potpie agent."""

    @abstractmethod
    async def run_agent(
        self, user_id: str, repo_name: str, branch: str, agent_id: str, query: str
    ) -> AgentResponse:
        """Run the query for given agent."""
        pass


class InternalAgent(ABC):
    """Abstract adapter for internal llm calls/ agentic tasks"""

    @abstractmethod
    async def run_agent(self, query: str) -> str:
        """Run the query for given agent."""
        pass
