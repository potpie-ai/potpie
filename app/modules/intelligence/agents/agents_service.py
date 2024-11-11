from typing import List
import os
import aiohttp

from app.modules.auth.auth_service import AuthService
from app.modules.intelligence.agents.agents_schema import AgentInfo
from app.modules.intelligence.prompts.prompt_service import PromptService


class AgentsService:

    def __init__(self, db):
        self.project_path = os.getenv("PROJECT_PATH", "projects/")
        self.db = db
        self.prompt_service = PromptService(db)
        self.base_url = os.getenv("POTPIE_PLUS_BASE_URL","http://localhost:8000")

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
                description="An agent specialized in generating detailed analysis of code changes in your current branch compared to default branch. Works best with Py, JS, TS",
                status="SYSTEM",
            ),
        ]

        try:
            custom_agents = await self.fetch_custom_agents(current_user["user_id"])
        except Exception as e:
            custom_agents = []

        if list_system_agents:
            return system_agents + custom_agents
        else:
            return custom_agents

    async def fetch_custom_agents(self, user_id: str) -> List[AgentInfo]:
        custom_agents = []
        skip = 0
        limit = 10
        hmac_signature = AuthService.generate_hmac_signature(f"user_id={user_id}")
        headers = {"X-HMAC-Signature": hmac_signature}

        async with aiohttp.ClientSession(headers=headers) as session:
            while True:
                url = f"{self.base_url}/custom-agents/agents/?user_id={user_id}&skip={skip}&limit={limit}"
                async with session.get(url) as response:
                    if response.status != 200:
                        break
                    data = await response.json()
                    if not data:
                        break

                    for agent in data:
                        custom_agents.append(
                            AgentInfo(
                                id=agent["id"],
                                name=agent["role"],
                                description=agent["goal"],
                                status=agent["deployment_status"],
                            )
                        )

                    skip += limit
                    if len(data) < limit:
                        break

        return custom_agents


    def format_citations(self, citations: List[str]) -> List[str]:
        cleaned_citations = []
        for citation in citations:
            cleaned_citations.append(
                citation.split(self.project_path, 1)[-1].split("/", 2)[-1]
                if self.project_path in citation
                else citation
            )
        return cleaned_citations
