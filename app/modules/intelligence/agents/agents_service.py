import hashlib
import hmac
from typing import List
import os
import aiohttp
from sqlalchemy.orm import Session

from app.modules.intelligence.agents.agents_schema import AgentInfo
from app.modules.intelligence.prompts.prompt_service import PromptService


class AgentsService:

    def __init__(self, db):
        self.project_path = os.getenv("PROJECT_PATH", "projects/")
        self.db = db
        self.prompt_service = PromptService(db)
        self.base_url = os.getenv("POTPIE_PLUS_BASE_URL")
        self.hmac_secret = os.getenv("POTPIE_PLUS_HMAC_SECRET")

    async def list_available_agents(
        self, current_user: dict, list_system_agents: bool
    ) -> List[AgentInfo]:
        system_agents = [
            AgentInfo(
                id="codebase_qna_agent",
                name="Codebase Q&A Agent",
                description="An agent specialized in answering questions about the codebase using the knowledge graph and code analysis tools.",
            ),
            AgentInfo(
                id="debugging_agent",
                name="Debugging with Knowledge Graph Agent",
                description="An agent specialized in debugging using knowledge graphs.",
            ),
            AgentInfo(
                id="unit_test_agent",
                name="Unit Test Agent",
                description="An agent specialized in generating unit tests for code snippets for given function names",
            ),
            AgentInfo(
                id="integration_test_agent",
                name="Integration Test Agent",
                description="An agent specialized in generating integration tests for code snippets from the knowledge graph based on given function names of entry points. Works best with Py, JS, TS",
            ),
            AgentInfo(
                id="LLD_agent",
                name="Low-Level Design Agent",
                description="An agent specialized in generating a low-level design plan for implementing a new feature.",
            ),
            AgentInfo(
                id="code_changes_agent",
                name="Code Changes Agent",
                description="An agent specialized in generating detailed analysis of code changes in your current branch compared to default branch. Works best with Py, JS, TS",
            ),
        ]

        custom_agents = await self.fetch_custom_agents(current_user)

        if list_system_agents:
            return system_agents + custom_agents
        else:
            return custom_agents

    async def fetch_custom_agents(self, current_user: dict) -> List[AgentInfo]:
        custom_agents = []
        skip = 0
        limit = 10
        print("current_user", current_user)
        user_id = current_user["user_id"]
        hmac_signature = self.generate_hmac_signature(f"user_id={user_id}")
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
                            )
                        )

                    skip += limit
                    if len(data) < limit:
                        break

        return custom_agents

    def generate_hmac_signature(self, message: str) -> str:
        secret_key = "1234"
        return hmac.new(
            secret_key.encode(), message.encode(), hashlib.sha256
        ).hexdigest()

    def generate_hmac_token(self, user_id: str) -> str:
        message = user_id.encode("utf-8")
        signature = hmac.new(self.hmac_secret.encode("utf-8"), message, hashlib.sha256)
        return f"{user_id}:{signature.hexdigest()}"

    def format_citations(self, citations: List[str]) -> List[str]:
        cleaned_citations = []
        for citation in citations:
            cleaned_citations.append(
                citation.split(self.project_path, 1)[-1].split("/", 2)[-1]
                if self.project_path in citation
                else citation
            )
        return cleaned_citations
