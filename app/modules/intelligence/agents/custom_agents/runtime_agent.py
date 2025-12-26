from pydantic import BaseModel
from app.modules.intelligence.agents.chat_agents.pydantic_multi_agent import (
    PydanticMultiAgent,
)
from app.modules.intelligence.agents.multi_agent_config import MultiAgentConfig
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.agents.chat_agents.agent_config import (
    AgentConfig,
    TaskConfig,
)
from app.modules.intelligence.provider.exceptions import UnsupportedProviderError
from app.modules.intelligence.tools.tool_service import ToolService
from ..chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from app.modules.utils.logger import setup_logger
from typing import Any, AsyncGenerator, Dict, List, Union

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
    use_multi_agent: bool = True  # Multi-agent mode is now default


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

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role=self.agent_config.role,
            goal=self.agent_config.goal,
            backstory=self.agent_config.backstory + "\n\n" + how_to_prompt,
            tasks=[
                TaskConfig(
                    description=self.agent_config.tasks[0].description,
                    expected_output=f"{self.agent_config.tasks[0].expected_output}",
                )
            ],
        )

        tools = self.tools_provider.get_tools(self.agent_config.tasks[0].tools)

        # Extract MCP servers from the first task with graceful error handling
        mcp_servers = []
        try:
            if (
                hasattr(self.agent_config.tasks[0], "mcp_servers")
                and self.agent_config.tasks[0].mcp_servers
            ):
                mcp_servers = self.agent_config.tasks[0].mcp_servers
                logger.info(
                    f"Found {len(mcp_servers)} MCP servers in task configuration"
                )
        except Exception as e:
            logger.warning(
                f"Failed to extract MCP servers from task configuration: {e}. Continuing without MCP servers."
            )
            mcp_servers = []

        supports_pydantic = self.llm_provider.supports_pydantic("chat")
        should_use_multi = MultiAgentConfig.should_use_multi_agent("custom_agent")

        logger.info(
            f"RuntimeCustomAgent: supports_pydantic={supports_pydantic}, should_use_multi_agent={should_use_multi}"
        )
        logger.info(f"Current model: {self.llm_provider.chat_config.model}")
        logger.info(f"Model capabilities: {self.llm_provider.chat_config.capabilities}")

        if supports_pydantic:
            if should_use_multi:
                logger.info(
                    "✅ Using PydanticMultiAgent (multi-agent system) for custom agent"
                )
                agent = PydanticMultiAgent(
                    self.llm_provider, agent_config, tools, mcp_servers
                )
                self._pydantic_agent = agent  # Store reference for status access
                return agent
            else:
                logger.info(
                    "❌ Multi-agent disabled by config for custom agent, using PydanticRagAgent"
                )
                from app.modules.intelligence.agents.chat_agents.pydantic_agent import (
                    PydanticRagAgent,
                )

                agent = PydanticRagAgent(self.llm_provider, agent_config, tools)
                self._pydantic_agent = agent
                return agent
        else:
            logger.error(
                f"❌ Model '{self.llm_provider.chat_config.model}' does not support Pydantic - cannot create custom agent"
            )
            raise UnsupportedProviderError(
                f"Model '{self.llm_provider.chat_config.model}' does not support Pydantic-based agents."
            )

    async def _enriched_context(self, ctx: ChatContext) -> ChatContext:
        if ctx.node_ids and len(ctx.node_ids) > 0:
            code_results = await self.tools_provider.get_code_from_multiple_node_ids_tool.run_multiple(
                ctx.project_id, ctx.node_ids
            )
            ctx.additional_context += (
                f"Code referred to in the query:\n {code_results}\n"
            )
        return ctx

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(await self._enriched_context(ctx))

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        ctx = await self._enriched_context(ctx)
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


how_to_prompt = """
    IMPORTANT: Use the following guide to accomplish tasks within the current context of execution

    🧠 **YOU HAVE ACCESS TO PAST MEMORIES (IF SEARCH TOOL IS SELECTED):**
    If you have access to the `search_user_memories` tool, use it actively throughout your work. This tool provides access to user's past interactions, preferences, decisions, and any relevant context from previous conversations.

    HOW TO GUIDE:

    🧠 **SEARCH MEMORIES FOR CONTEXT (USE WHENEVER NEEDED):**
    - **If `search_user_memories` tool is available**, use it proactively throughout your work
    - **BEFORE making decisions**, search for user preferences and past context
    - Query for: coding style, naming conventions, framework preferences, architectural patterns, past decisions, "[specific topic]"
    - Apply discovered preferences to all your outputs
    - **Search memories whenever you need guidance** - they contain valuable personalized context
    - This ensures responses aligned with user's established preferences and style

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
