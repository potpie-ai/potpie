import os
from typing import List, Optional

from app.modules.intelligence.agents.chat_agents.system_agents.general_purpose_agent import (
    GeneralPurposeAgent,
)
from app.modules.intelligence.agents.custom_agents.custom_agent_schema import (
    AgentVisibility,
)
from app.modules.intelligence.agents.custom_agents.custom_agents_service import (
    CustomAgentService,
)
from app.modules.utils.logger import setup_logger
from .chat_agent import AgentWithInfo, ChatContext
from .chat_agents.system_agents import (
    blast_radius_agent,
    code_gen_agent,
    debug_agent,
    integration_test_agent,
    low_level_design_agent,
    qna_agent,
    unit_test_agent,
)
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.agents.chat_agents.adaptive_agent import (
    PromptService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.intelligence.agents.chat_agents.supervisor_agent import (
    SupervisorAgent,
)
from pydantic import BaseModel

logger = setup_logger(__name__)


class AgentInfo(BaseModel):
    id: str
    name: str
    description: str
    status: str
    visibility: Optional[AgentVisibility] = None


class AgentsService:
    def __init__(
        self,
        db,
        llm_provider: ProviderService,
        prompt_provider: PromptService,
        tools_provider: ToolService,
    ):
        self.project_path = os.getenv("PROJECT_PATH", "projects/")
        self.db = db
        self.prompt_service = PromptService(db)
        self.system_agents = self._system_agents(
            llm_provider, prompt_provider, tools_provider
        )
        self.supervisor_agent = SupervisorAgent(llm_provider, self.system_agents)
        self.custom_agent_service = CustomAgentService(self.db)

    def _system_agents(
        self,
        llm_provider: ProviderService,
        prompt_provider: PromptService,
        tools_provider: ToolService,
    ):
        return {
            "codebase_qna_agent": AgentWithInfo(
                id="codebase_qna_agent",
                name="Codebase Q&A Agent",
                description="An agent specialized in answering questions about the codebase using the knowledge graph and code analysis tools.",
                agent=qna_agent.QnAAgent(llm_provider, tools_provider, prompt_provider),
            ),
            "debugging_agent": AgentWithInfo(
                id="debugging_agent",
                name="Debugging with Knowledge Graph Agent",
                description="An agent specialized in debugging using knowledge graphs.",
                agent=debug_agent.DebugAgent(
                    llm_provider, tools_provider, prompt_provider
                ),
            ),
            "unit_test_agent": AgentWithInfo(
                id="unit_test_agent",
                name="Unit Test Agent",
                description="An agent specialized in generating unit tests for code snippets for given function names",
                agent=unit_test_agent.UnitTestAgent(
                    llm_provider, tools_provider, prompt_provider
                ),
            ),
            "integration_test_agent": AgentWithInfo(
                id="integration_test_agent",
                name="Integration Test Agent",
                description="An agent specialized in generating integration tests for code snippets from the knowledge graph based on given function names of entry points. Works best with Py, JS, TS",
                agent=integration_test_agent.IntegrationTestAgent(
                    llm_provider, tools_provider, prompt_provider
                ),
            ),
            "LLD_agent": AgentWithInfo(
                id="LLD_agent",
                name="Low-Level Design Agent",
                description="An agent specialized in generating a low-level design plan for implementing a new feature.",
                agent=low_level_design_agent.LowLevelDesignAgent(
                    llm_provider, tools_provider, prompt_provider
                ),
            ),
            "code_changes_agent": AgentWithInfo(
                id="code_changes_agent",
                name="Code Changes Agent",
                description="An agent specialized in generating blast radius of the code changes in your current branch compared to default branch. Use this for functional review of your code changes. Works best with Py, JS, TS",
                agent=blast_radius_agent.BlastRadiusAgent(
                    llm_provider, tools_provider, prompt_provider
                ),
            ),
            "code_generation_agent": AgentWithInfo(
                id="code_generation_agent",
                name="Code Generation Agent",
                description="An agent specialized in generating code for new features or fixing bugs.",
                agent=code_gen_agent.CodeGenAgent(
                    llm_provider, tools_provider, prompt_provider
                ),
            ),
            "general_purpose_agent": AgentWithInfo(
                id="general_purpose_agent",
                name="General Purpose Agent",
                description="Agent for queries not needing understanding of or access to the users code repository",
                agent=GeneralPurposeAgent(
                    llm_provider, tools_provider, prompt_provider
                ),
            ),
        }

    async def execute(self, ctx: ChatContext):
        return await self.supervisor_agent.run(ctx)

    async def execute_stream(self, ctx: ChatContext):
        async for chunk in self.supervisor_agent.run_stream(ctx):
            yield chunk

    async def list_available_agents(
        self, current_user: dict, list_system_agents: bool
    ) -> List[AgentInfo]:
        system_agents = [
            AgentInfo(
                id=id,
                name=self.system_agents[id].name,
                description=self.system_agents[id].description,
                status="SYSTEM",
            )
            for (id) in self.system_agents
        ]

        try:
            custom_agents = await CustomAgentService(self.db).list_agents(
                current_user["user_id"]
            )
        except Exception as e:
            logger.error(
                f"Failed to fetch custom agents for user {current_user['user_id']}: {e}"
            )
            custom_agents = []
        agent_info_list = [
            AgentInfo(
                id=agent.id,
                name=agent.role,
                description=agent.goal,
                status=agent.deployment_status,
                visibility=agent.visibility,
            )
            for agent in custom_agents
        ]

        if list_system_agents:
            return system_agents + agent_info_list
        else:
            return agent_info_list

    async def validate_agent_id(self, user_id: str, agent_id: str) -> str | None:
        """Validate if an agent ID is valid"""
        if agent_id in self.system_agents:
            return "SYSTEM_AGENT"
        if await self.custom_agent_service.get_custom_agent(self.db, user_id, agent_id):
            return "CUSTOM_AGENT"
        return None
