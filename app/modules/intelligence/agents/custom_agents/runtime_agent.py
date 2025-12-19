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
from typing import Any, AsyncGenerator, Dict, List, Union, Optional
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

logger = setup_logger(__name__)

# Maximum runtime delegation depth to prevent infinite loops
MAX_RUNTIME_DELEGATION_DEPTH = 3


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
    sub_agents: List[str] = []
    project_id: str = ""
    use_multi_agent: bool = True  # Multi-agent mode is now default


class RuntimeCustomAgent(ChatAgent):
    """
    Runtime custom agent that can operate in two modes:
    - Supervisor mode (default): Full planning, delegation tools, orchestration capabilities
    - Delegate mode: Focused task execution with just tools, no planning overhead
    
    Sub-agents always run in delegate mode for efficiency.
    """
    
    # Delegate agent instructions - simple, focused on task execution
    DELEGATE_INSTRUCTIONS = """You are a focused task execution agent. Execute the assigned task efficiently using your available tools.

**EXECUTION RULES:**
- Execute tasks completely without asking for clarification unless critical info is missing
- Make reasonable assumptions and mention them
- Use tools to gather information and perform actions
- Return focused, actionable results
- Start your response with "## Task Result"

**CODE MANAGEMENT:**
- Use code changes tools (add_file_to_changes, update_file_lines, etc.) instead of including code in response text
- Use show_updated_file and show_diff to display changes
- Keep responses concise"""
    
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        agent_config: Dict[str, Any],
        db: Optional[Session] = None,
        delegation_depth: int = 0,  # Track runtime delegation depth
        is_delegate: bool = False,  # Run in delegate mode (no planning/delegation overhead)
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.agent_config = CustomAgentConfig(**agent_config)
        self.db = db
        self.delegation_depth = delegation_depth
        self.is_delegate = is_delegate
        self._current_context: Optional[ChatContext] = None

    def _build_agent(self) -> ChatAgent:
        # Delegate mode: Create a lightweight focused agent (no planning/delegation overhead)
        if self.is_delegate:
            return self._build_delegate_agent()
        
        # Supervisor mode: Full multi-agent with planning and delegation
        return self._build_supervisor_agent()
    
    def _build_delegate_agent(self) -> ChatAgent:
        """Build a lightweight delegate agent focused on task execution.
        
        Delegate agents (sub-agents) are tool-like executors:
        - They use their configured tools directly
        - NO sub-sub-agents (no delegation to other sub-agents)
        - NO planning overhead
        - Just execute and return results
        """
        from pydantic_ai import Agent, Tool
        import functools
        
        # Get only the sub-agent's configured tools - NO delegation tools
        tools = self.tools_provider.get_tools(self.agent_config.tasks[0].tools)
        
        logger.info(
            f"🎯 Building delegate agent '{self.agent_config.role}' with {len(tools)} tools "
            f"(NO sub-agent delegation)"
        )
        
        # Wrap tools for pydantic-ai
        def handle_exception(tool_func):
            @functools.wraps(tool_func)
            def wrapper(*args, **kwargs):
                try:
                    return tool_func(*args, **kwargs)
                except Exception:
                    logger.exception("Exception in tool function", tool_name=tool_func.__name__)
                    return "An internal error occurred. Please try again later."
            return wrapper
        
        pydantic_tools = [
            Tool(
                name=tool.name.replace(" ", ""),
                description=tool.description,
                function=handle_exception(tool.func),
            )
            for tool in tools
        ]
        
        # Create a simple pydantic-ai agent - no PydanticMultiAgent overhead
        delegate_agent = Agent(
            model=self.llm_provider.get_pydantic_model(),
            tools=pydantic_tools,
            instructions=f"""{self.DELEGATE_INSTRUCTIONS}

**YOUR IDENTITY:**
Role: {self.agent_config.role}
Goal: {self.agent_config.goal}

{self.agent_config.backstory}
""",
            output_retries=3,
            output_type=str,
            defer_model_check=True,
            end_strategy="exhaustive",
            instrument=True,
        )
        
        # Wrap in a simple ChatAgent adapter
        return _DelegateAgentWrapper(delegate_agent)
    
    def _build_supervisor_agent(self) -> ChatAgent:
        """Build a full supervisor agent with planning and delegation capabilities."""
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

        # Add sub-agent delegation tools
        sub_agent_tools = self._create_sub_agent_delegation_tools()
        tools = tools + sub_agent_tools

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
        self._current_context = ctx
        ctx = await self._enriched_context(ctx)
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk

    def _create_sub_agent_delegation_tools(self) -> List[StructuredTool]:
        """Create delegation tools for configured sub-agents"""
        from app.modules.intelligence.agents.custom_agents.custom_agent_model import (
            CustomAgent as CustomAgentModel,
        )

        delegation_tools = []

        if not self.db or not self.agent_config.sub_agents:
            return delegation_tools

        for sub_agent_id in self.agent_config.sub_agents:
            sub_agent_model = (
                self.db.query(CustomAgentModel)
                .filter(CustomAgentModel.id == sub_agent_id)
                .first()
            )

            if not sub_agent_model:
                logger.warning(f"Sub-agent {sub_agent_id} not found, skipping")
                continue

            safe_name = sub_agent_model.role.lower().replace(" ", "_").replace("-", "_")
            safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
            tool_name = f"delegate_to_{safe_name}"

            delegation_tool = StructuredTool.from_function(
                func=self._create_sub_agent_executor(sub_agent_model),
                name=tool_name,
                description=f"""🤖 DELEGATE TO {sub_agent_model.role.upper()} - 
Goal: {sub_agent_model.goal}

Use this to delegate specific tasks to the {sub_agent_model.role} sub-agent. 
The sub-agent will execute the task using its specialized tools and return results.

Provide clear, specific task instructions. The sub-agent will NOT do exploratory work.

Args:
    task_description: Clear description of the specific task to execute
    context: Any relevant context from your current work (file paths, code snippets, findings)
""",
            )
            delegation_tools.append(delegation_tool)
            logger.info(
                f"Created delegation tool '{tool_name}' for sub-agent {sub_agent_id}"
            )

        return delegation_tools

    def _create_sub_agent_executor(self, sub_agent_model):
        """Create an executor function for a sub-agent"""

        def execute_sub_agent(task_description: str, context: str = "") -> str:
            import asyncio

            # Check runtime depth to prevent infinite loops
            new_depth = self.delegation_depth + 1
            if new_depth > MAX_RUNTIME_DELEGATION_DEPTH:
                logger.warning(
                    f"⚠️ Maximum runtime delegation depth ({MAX_RUNTIME_DELEGATION_DEPTH}) exceeded. "
                    f"Stopping delegation to prevent infinite loop."
                )
                return f"""## Task Result

⚠️ **Delegation depth limit reached**

The task could not be delegated because the maximum delegation depth ({MAX_RUNTIME_DELEGATION_DEPTH}) was exceeded. 
This prevents infinite delegation loops.

**Original Task:** {task_description[:200]}{'...' if len(task_description) > 200 else ''}

**Recommendation:** The parent agent should attempt to complete this task directly or break it into simpler steps.
"""

            try:
                logger.info(
                    f"Executing custom sub-agent '{sub_agent_model.role}' (id: {sub_agent_model.id}) "
                    f"at delegation depth {new_depth}/{MAX_RUNTIME_DELEGATION_DEPTH}"
                )

                # Create tool service for the sub-agent
                sub_tool_service = ToolService(self.db, sub_agent_model.user_id)

                # Build sub-agent config - sub-agents are tool-like executors (NO sub-sub-agents)
                sub_agent_config = {
                    "user_id": sub_agent_model.user_id,
                    "role": sub_agent_model.role,
                    "goal": sub_agent_model.goal,
                    "backstory": sub_agent_model.backstory,
                    "system_prompt": sub_agent_model.system_prompt,
                    "tasks": sub_agent_model.tasks,
                    "sub_agents": [],  # NO sub-sub-agents - sub-agents just execute
                }

                # Create RuntimeCustomAgent in DELEGATE MODE (tool-like executor)
                sub_runtime = RuntimeCustomAgent(
                    self.llm_provider,
                    sub_tool_service,
                    sub_agent_config,
                    db=None,  # No db needed - sub-agents can't have their own sub-agents
                    delegation_depth=new_depth,
                    is_delegate=True,  # Run in delegate mode - just execute and return
                )

                # Build comprehensive context for the sub-agent
                project_context = self._create_project_context_info()

                full_task = f"""Execute this focused task:

**TASK:**
{task_description}

**PROJECT CONTEXT:**
{project_context}

**CONTEXT FROM PARENT AGENT:**
{context if context else "No additional context provided"}

**YOUR MISSION:**
Execute the task above and return ONLY the specific, actionable result.
Start your response with "## Task Result" and then provide the focused answer.

**CRITICAL:**
1. Execute exactly what is asked - no exploratory work
2. Use your tools and capabilities to complete the task
3. Be specific and concise
4. Return actionable results
"""

                # Check if we're in an async context with a running loop
                try:
                    asyncio.get_running_loop()
                    import concurrent.futures

                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(
                                self._run_sub_agent_async(sub_runtime, full_task, sub_agent_model.id)
                            )
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        result = future.result()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            self._run_sub_agent_async(sub_runtime, full_task, sub_agent_model.id)
                        )
                    finally:
                        loop.close()

                logger.info(
                    f"✅ Custom sub-agent '{sub_agent_model.role}' (id: {sub_agent_model.id}) "
                    f"completed task successfully at depth {new_depth}"
                )

                return result

            except Exception as e:
                logger.exception(f"Error executing sub-agent {sub_agent_model.id}")
                return f"## Task Result\n\n❌ Error executing sub-agent: {str(e)}"

        return execute_sub_agent

    def _create_project_context_info(self) -> str:
        """Create project context information for sub-agents"""
        if not self._current_context:
            return "No project context available"

        ctx = self._current_context
        context_parts = []

        if ctx.project_id:
            context_parts.append(f"Project: {ctx.project_name} (ID: {ctx.project_id})")

        if ctx.node_ids:
            node_ids = ctx.node_ids if isinstance(ctx.node_ids, list) else [ctx.node_ids]
            if node_ids:
                context_parts.append(f"Nodes: {', '.join(node_ids)}")

        if ctx.additional_context:
            context_parts.append(f"Additional Context: {ctx.additional_context}")

        return "\n".join(context_parts) if context_parts else "No additional context"

    async def _run_sub_agent_async(
        self, sub_runtime: "RuntimeCustomAgent", task: str, sub_agent_id: str
    ) -> str:
        """Run a sub-agent asynchronously and return the full response"""
        ctx = ChatContext(
            query=task,
            project_id=self._current_context.project_id if self._current_context else "",
            node_ids=self._current_context.node_ids if self._current_context else [],
            project_name=self._current_context.project_name if self._current_context else "",
            curr_agent_id=sub_agent_id,
            history=[],
            additional_context="",
        )

        full_response = ""
        async for chunk in sub_runtime.run_stream(ctx):
            full_response += chunk.response

        return full_response


class _DelegateAgentWrapper(ChatAgent):
    """Simple wrapper to make pydantic-ai Agent compatible with ChatAgent interface."""
    
    def __init__(self, agent):
        self._agent = agent
    
    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        from pydantic_ai.usage import UsageLimits
        
        result = await self._agent.run(
            user_prompt=ctx.query,
            usage_limits=UsageLimits(request_limit=None),
        )
        return ChatAgentResponse(
            response=result.output,
            tool_calls=[],
            citations=[],
        )
    
    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        from pydantic_ai import Agent
        from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, TextPart, TextPartDelta
        from pydantic_ai.usage import UsageLimits
        
        async with self._agent.iter(
            user_prompt=ctx.query,
            usage_limits=UsageLimits(request_limit=None),
        ) as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                                yield ChatAgentResponse(
                                    response=event.part.content,
                                    tool_calls=[],
                                    citations=[],
                                )
                            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                yield ChatAgentResponse(
                                    response=event.delta.content_delta,
                                    tool_calls=[],
                                    citations=[],
                                )
                elif Agent.is_end_node(node):
                    break


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
