from typing import List

from sqlalchemy.orm import Session
from uuid6 import uuid7

from app.modules.intelligence.agents.custom_agents.custom_agents_model import CustomAgent
from app.modules.intelligence.agents.agents_schema import (
    Agent,
    AgentDetailsAsACrewAIAgent,
    AgentInfo,
)
from app.modules.intelligence.prompts.prompt_schema import PromptType
from app.modules.intelligence.prompts.prompt_service import PromptService


class AgentsService:
    def __init__(self, db: Session):
        self.db = db
        self.prompt_service = PromptService(db)

    async def list_available_agents(self) -> List[AgentInfo]:
        return [
            AgentInfo(
                id="debugging_agent",
                name="Debugging with Knowledge Graph Agent",
                description="An agent specialized in debugging using knowledge graphs.",
            ),
            AgentInfo(
                id="codebase_qna_agent",
                name="Codebase Q&A Agent",
                description="An agent specialized in answering questions about the codebase using the knowledge graph and code analysis tools.",
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
                id="code_changes_agent",
                name="Code Changes Agent",
                description="An agent specialized in generating detailed analysis of code changes in your current branch compared to default branch. Works best with Py, JS, TS",
            ),
        ]

    def format_citations(self, citations: List[str]) -> List[str]:
        cleaned_citations = []
        for citation in citations:
            cleaned_citations.append(
                citation.split("/projects/", 1)[-1].split("/", 1)[-1]
                if "/projects/" in citation
                else citation
            )
        return cleaned_citations

    async def get_agent_details(self, agent_id: str) -> AgentDetailsAsACrewAIAgent:
        # Fetch prompts and other components related to the agent
        prompts = await self.prompt_service.get_prompts_by_agent_id_and_types(
            agent_id, [PromptType.SYSTEM, PromptType.HUMAN]
        )

        # Extract backstory and goals from prompts
        backstory = next(
            (prompt.text for prompt in prompts if prompt.type == PromptType.SYSTEM), ""
        )
        goals = [prompt.text for prompt in prompts if prompt.type == PromptType.HUMAN]

        # Construct the agent details
        agent_details = AgentDetailsAsACrewAIAgent(
            id=agent_id,
            name=f"{agent_id.replace('_', ' ').title()}",
            description=f"Details for {agent_id.replace('_', ' ').title()}",
            backstory=backstory,
            goals=goals,
            tools=[],  # Assuming no tools for now, you may need to fetch these separately
        )
        return agent_details

    async def create_or_update_agent(
        self,
        user_id: str,
        role: str,
        goal: str,
        backstory: str,
        tool_ids: List[str],
        tasks: List[dict],
    ) -> Agent:
        agent_id = str(uuid7())
        custom_agent = CustomAgent(
            id=agent_id,
            user_id=user_id,
            role=role,
            goal=goal,
            backstory=backstory,
            tool_ids=tool_ids,
            tasks=tasks,
        )
        self.db.add(custom_agent)

        self.db.commit()
        self.db.refresh(custom_agent)
        return Agent(**custom_agent.__dict__)
