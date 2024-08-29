from typing import List

from app.modules.intelligence.agents.agents_schema import AgentInfo


class AgentsService:
    def __init__(self, db):
        self.db = db

    @classmethod
    def create(cls, db):
        return cls(db)

    async def list_available_agents(self) -> List[AgentInfo]:
        return [
            AgentInfo(
                id="debugging_agent",
                name="Debugging with Knowledge Graph Agent",
                description="An agent specialized in debugging using knowledge graphs.",
            ),
            AgentInfo(
                id="chat_llm_orchestrator",
                name="Intelligent Tool-Using Orchestrator",
                description="A versatile chat LLM that can use various tools to assist with a wide range of tasks.",
            ),
            AgentInfo(
                id="codebase_qna_agent",
                name="Codebase Q&A Agent",
                description="An agent specialized in answering questions about the codebase using the knowledge graph and code analysis tools.",
            ),
        ]
