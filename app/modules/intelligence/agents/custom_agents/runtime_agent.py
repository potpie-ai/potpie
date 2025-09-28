from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from pydantic import BaseModel

from app.modules.intelligence.agents.chat_agents.pydantic_agent import (
    PydanticRagAgent,
)
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from ..chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from ..chat_agents.crewai_agent import AgentConfig, CrewAIAgent, TaskConfig
from app.modules.utils.logger import setup_logger


logger = setup_logger(__name__)


class CustomTaskConfig(BaseModel):
    """Model for task configuration from agent_config.json"""

    tools: List[str]
    description: str
    expected_output: Union[Dict[str, Any], str]
    mcp_servers: List[Dict[str, Any]] = []


class CustomAgentConfig(BaseModel):
    """Model for agent configuration from agent_config.json"""

    user_id: str
    role: str
    goal: str
    backstory: str
    system_prompt: str
    tasks: List[CustomTaskConfig]
    project_id: str = ""


class RuntimeCustomAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        agent_config: Dict[str, Any],
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.agent_config = CustomAgentConfig(**agent_config)
        self._project_context_cache: Dict[str, str] = {}
        self._pydantic_agent: Optional[PydanticRagAgent] = None

    def _build_agent(self) -> ChatAgent:
        if not self.agent_config.tasks:
            raise ValueError("Agent configuration must include at least one task")

        first_task = self.agent_config.tasks[0]
        agent_config = AgentConfig(
            role=self.agent_config.role,
            goal=self.agent_config.goal,
            backstory=self.agent_config.backstory + "\n\n" + how_to_prompt,
            tasks=[
                TaskConfig(
                    description=first_task.description,
                    expected_output=f"{first_task.expected_output}",
                )
            ],
        )

        tools = self.tools_provider.get_tools(first_task.tools)

        mcp_servers: List[Dict[str, Any]] = []
        try:
            if hasattr(first_task, "mcp_servers") and first_task.mcp_servers:
                mcp_servers = first_task.mcp_servers
                logger.info(
                    "Found %d MCP servers in task configuration", len(mcp_servers)
                )
        except Exception as exc:
            logger.warning(
                "Failed to extract MCP servers from task configuration: %s. Continuing without MCP servers.",
                exc,
            )
            mcp_servers = []

        if self.llm_provider.is_current_model_supported_by_pydanticai(
            config_type="chat"
        ):
            agent = PydanticRagAgent(
                self.llm_provider, agent_config, tools, mcp_servers
            )
            self._pydantic_agent = agent
            return agent

        return CrewAIAgent(self.llm_provider, agent_config, tools)

    async def _get_large_pr_context(self, project_id: str) -> Optional[str]:
        if not project_id:
            return None

        if project_id in self._project_context_cache:
            return self._project_context_cache[project_id]

        tool = getattr(self.tools_provider, "process_large_pr_tool", None)
        if not tool:
            return None

        payload = {"project_id": project_id}
        try:
            logger.info("Fetching large PR context for project %s", project_id)
            context = await tool.arun(payload)
        except Exception as exc:
            logger.warning("Failed to fetch large PR context: %s", exc)
            return None

        if isinstance(context, str) and context.strip():
            self._project_context_cache[project_id] = context
            return context

        return None

    async def _enriched_context(self, ctx: ChatContext) -> ChatContext:
        project_id = ctx.project_id or self.agent_config.project_id
        pr_context = await self._get_large_pr_context(project_id)
        if pr_context:
            project_header = (
                f"Project Name (github: {ctx.project_name})\n"
                if ctx.project_name
                else ""
            )
            context_block = f"{project_header}{pr_context}".strip()
            if context_block:
                if ctx.additional_context:
                    ctx.additional_context += "\n"
                ctx.additional_context += f"PR Analysis Summary:\n{context_block}"

        if ctx.node_ids:
            try:
                code_results = await (
                    self.tools_provider.get_code_from_multiple_node_ids_tool.run_multiple(
                        ctx.project_id, ctx.node_ids
                    )
                )
                if code_results:
                    if ctx.additional_context:
                        ctx.additional_context += "\n"
                    ctx.additional_context += (
                        f"Code referred to in the query:\n{code_results}\n"
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch code snippets for node ids %s: %s",
                    ctx.node_ids,
                    exc,
                )

        return ctx

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        enriched_ctx = await self._enriched_context(ctx)
        return await self._build_agent().run(enriched_ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        enriched_ctx = await self._enriched_context(ctx)
        async for chunk in self._build_agent().run_stream(enriched_ctx):
            yield chunk


how_to_prompt = """
    IMPORTANT: Use the following guide to accomplish tasks within the current context of execution
    HOW TO GUIDE:

    IMPORATANT: steps on HOW TO traverse the codebase:
    1. You can use websearch, docstrings, readme to understand current feature/code you are working with better. Understand how to use current feature in context of codebase
    2. Use AskKnowledgeGraphQueries tool to understand where perticular feature or functionality resides or to fetch specific code related to some keywords. Fetch file structure to understand the codebase better, Use FetchFile tool to fetch code from a file
    3. Use GetcodefromProbableNodeIDs tool to fetch code for perticular class or function in a file, Use analyze_code_structure to get all the class/function/nodes in a file
    4. Use GetcodeFromMultipleNodeIDs to fetch code for nodeIDs fetched from tools before
    5. Use GetNodeNeighboursFromNodeIDs to fetch all the code referencing current code or code referenced in the current node (code snippet)
    6. Above tools and steps can help you figure out full context about the current code in question
    7. Figure out how all the code ties together to implement current functionality
    8. Fetch Dir structure of the repo and use fetch file tool to fetch entire files, if file is too big the tool will throw error, then use code analysis tool to target proper line numbers (feel free to use set startline and endline such that few extra context lines are also fetched, tool won't throw out of bounds exception and return lines if they exist)
    9. Use above mentioned tools to fetch imported code, referenced code, helper functions, classes etc to understand the control flow
"""
