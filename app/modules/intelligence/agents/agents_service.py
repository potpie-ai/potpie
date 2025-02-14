import os
from typing import List

from app.modules.intelligence.agents.agents_schema import AgentInfo
from app.modules.intelligence.agents.custom_agents.custom_agents_service import (
    CustomAgentService,
)
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

class AgentsService:
    def __init__(self, db):
        self.project_path = os.getenv("PROJECT_PATH", "projects/")
        self.db = db
        self.prompt_service = PromptService(db)

    async def list_available_agents(
        self, current_user: dict, list_system_agents: bool
    ) -> List[AgentInfo]:
        system_agents = [
            AgentInfo(
                id="codebase_qna_agent",
                name="Codebase Q&A Agent",
                description="An agent specialized in answering questions about the codebase using the knowledge graph and code analysis tools.",
                status="SYSTEM",
            ),
            AgentInfo(
                id="debugging_agent",
                name="Debugging with Knowledge Graph Agent",
                description="An agent specialized in debugging using knowledge graphs.",
                status="SYSTEM",
            ),
            AgentInfo(
                id="unit_test_agent",
                name="Unit Test Agent",
                description="An agent specialized in generating unit tests for code snippets for given function names",
                status="SYSTEM",
            ),
            AgentInfo(
                id="integration_test_agent",
                name="Integration Test Agent",
                description="An agent specialized in generating integration tests for code snippets from the knowledge graph based on given function names of entry points. Works best with Py, JS, TS",
                status="SYSTEM",
            ),
            AgentInfo(
                id="LLD_agent",
                name="Low-Level Design Agent",
                description="An agent specialized in generating a low-level design plan for implementing a new feature.",
                status="SYSTEM",
            ),
            AgentInfo(
                id="code_changes_agent",
                name="Code Changes Agent",
                description="An agent specialized in generating blast radius of the code changes in your current branch compared to default branch. Use this for functional review of your code changes. Works best with Py, JS, TS",
                status="SYSTEM",
            ),
            AgentInfo(
                id="code_generation_agent",
                name="Code Generation Agent",
                description="An agent specialized in generating code for new features or fixing bugs.",
                status="SYSTEM",
            ),
        ]

        try:
            custom_agents = CustomAgentService(self.db).list_agents(
                current_user["user_id"]
            )
        except Exception as e:
            logger.error(f"Failed to fetch custom agents for user {current_user['user_id']}: {e}")
            custom_agents = []
        agent_info_list = [
            AgentInfo(
                id=agent.id,
                name=agent.role,
                description=agent.goal,
                status=agent.deployment_status,
            )
            for agent in custom_agents
        ]

        if list_system_agents:
            return system_agents + agent_info_list
        else:
            return agent_info_list

    def format_citations(self, citations: List[str]) -> List[str]:
        cleaned_citations = []
        for citation in citations:
            cleaned_citations.append(
                citation.split(self.project_path, 1)[-1].split("/", 2)[-1]
                if self.project_path in citation
                else citation
            )
        return cleaned_citations
