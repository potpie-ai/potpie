import functools
import re
from typing import List, AsyncGenerator, Sequence, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

import anyio

from pydantic_ai import Agent, Tool, RunContext
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
    ModelResponse,
    TextPart,
    ImageUrl,
    UserContent,
    ModelMessage,
)
from pydantic_ai.exceptions import ModelRetry, AgentRunError, UserError
from langchain_core.tools import StructuredTool

from .tool_helpers import (
    get_tool_call_info_content,
    get_tool_response_message,
    get_tool_result_info_content,
    get_tool_run_message,
    get_delegation_call_message,
    get_delegation_response_message,
    get_delegation_info_content,
    get_delegation_result_content,
)
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from .crewai_agent import AgentConfig, TaskConfig
from app.modules.utils.logger import setup_logger

from ..chat_agent import (
    ChatAgent,
    ChatAgentResponse,
    ChatContext,
    ToolCallEventType,
    ToolCallResponse,
)

logger = setup_logger(__name__)


class AgentType(Enum):
    """Types of specialized agents in the multi-agent system"""

    SUPERVISOR = "supervisor"
    CAB = "codebase_analyzer"  # Codebase Analyzer
    CBL = "codebase_locator"  # Codebase Locator
    THINK_EXECUTE = "think_execute"  # Generic Think and Execute Agent


@dataclass
class AgentDelegationResult:
    """Result from agent delegation"""

    agent_type: AgentType
    result: str
    success: bool
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


def handle_exception(tool_func):
    @functools.wraps(tool_func)
    def wrapper(*args, **kwargs):
        try:
            return tool_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in tool function: {e}")
            return "An internal error occurred. Please try again later."

    return wrapper


def is_delegation_tool(tool_name: str) -> bool:
    """Check if a tool call is a delegation to a subagent"""
    return tool_name.startswith("delegate_to_")


def extract_agent_type_from_delegation_tool(tool_name: str) -> str:
    """Extract agent type from delegation tool name"""
    if tool_name.startswith("delegate_to_"):
        return tool_name[12:]  # Remove "delegate_to_" prefix
    return tool_name


def create_tool_call_response(event: FunctionToolCallEvent) -> ToolCallResponse:
    """Create appropriate tool call response for regular or delegation tools"""
    tool_name = event.part.tool_name

    if is_delegation_tool(tool_name):
        agent_type = extract_agent_type_from_delegation_tool(tool_name)
        args_dict = event.part.args_as_dict()
        task_description = args_dict.get("task_description", "")
        context = args_dict.get("context", "")

        return ToolCallResponse(
            call_id=event.part.tool_call_id or "",
            event_type=ToolCallEventType.DELEGATION_CALL,
            tool_name=tool_name,
            tool_response=get_delegation_call_message(agent_type),
            tool_call_details={
                "summary": get_delegation_info_content(
                    agent_type, task_description, context
                )
            },
        )
    else:
        return ToolCallResponse(
            call_id=event.part.tool_call_id or "",
            event_type=ToolCallEventType.CALL,
            tool_name=tool_name,
            tool_response=get_tool_run_message(tool_name),
            tool_call_details={
                "summary": get_tool_call_info_content(
                    tool_name, event.part.args_as_dict()
                )
            },
        )


def create_tool_result_response(event: FunctionToolResultEvent) -> ToolCallResponse:
    """Create appropriate tool result response for regular or delegation tools"""
    tool_name = event.result.tool_name or "unknown tool"

    if is_delegation_tool(tool_name):
        agent_type = extract_agent_type_from_delegation_tool(tool_name)
        result_content = str(event.result.content) if event.result.content else ""

        return ToolCallResponse(
            call_id=event.result.tool_call_id or "",
            event_type=ToolCallEventType.DELEGATION_RESULT,
            tool_name=tool_name,
            tool_response=get_delegation_response_message(agent_type),
            tool_call_details={
                "summary": get_delegation_result_content(agent_type, result_content)
            },
        )
    else:
        return ToolCallResponse(
            call_id=event.result.tool_call_id or "",
            event_type=ToolCallEventType.RESULT,
            tool_name=tool_name,
            tool_response=get_tool_response_message(tool_name),
            tool_call_details={
                "summary": get_tool_result_info_content(tool_name, event.result.content)
            },
        )


class PydanticMultiAgent(ChatAgent):
    """
    Multi-agent system using Pydantic AI with agent delegation patterns.

    This system consists of:
    1. A supervisor agent that coordinates tasks
    2. Specialized delegate agents for specific tasks
    3. Context passing between agents
    4. Usage tracking across all agents
    """

    def __init__(
        self,
        llm_provider: ProviderService,
        config: AgentConfig,
        tools: List[StructuredTool],
        mcp_servers: List[dict] | None = None,
        delegate_agents: Optional[Dict[AgentType, AgentConfig]] = None,
    ):
        """Initialize the multi-agent system with configuration and tools"""

        self.tasks = config.tasks
        self.max_iter = config.max_iter
        self.llm_provider = llm_provider
        self.tools = tools
        self.config = config
        self.mcp_servers = mcp_servers or []

        # Clean tool names (no spaces for pydantic agents)
        for i, tool in enumerate(tools):
            tools[i].name = re.sub(r" ", "", tool.name)

        # Initialize delegate agents
        self.delegate_agents = delegate_agents or self._create_default_delegate_agents()

        # Store agent instances
        self._agent_instances: Dict[AgentType, Agent] = {}
        self._supervisor_agent: Optional[Agent] = None

    def _create_default_delegate_agents(self) -> Dict[AgentType, AgentConfig]:
        """Create default specialized agents if none provided"""
        return {
            AgentType.CAB: AgentConfig(
                role="Codebase Analyzer Specialist",
                goal="Analyze and document how code works by examining implementation details, tracing data flow, and explaining technical workings",
                backstory="""You are a specialist at understanding HOW code works. Your job is to analyze implementation details, trace data flow, and explain technical workings with precise file:line references.

CRITICAL: YOUR ONLY JOB IS TO DOCUMENT AND EXPLAIN THE CODEBASE AS IT EXISTS TODAY
- DO NOT suggest improvements or changes unless the user explicitly asks for them
- DO NOT perform root cause analysis unless the user explicitly asks for them
- DO NOT propose future enhancements unless the user explicitly asks for them
- DO NOT critique the implementation or identify "problems"
- DO NOT comment on code quality, performance issues, or security concerns
- DO NOT suggest refactoring, optimization, or better approaches
- ONLY describe what exists, how it works, and how components interact""",
                tasks=[
                    TaskConfig(
                        description="""Analyze the codebase to understand implementation details using available tools:

AVAILABLE TOOLS FOR ANALYSIS:
- fetch_file: Read complete file contents
- get_code_file_structure: Get directory structure and file organization
- analyze_code_structure: Analyze code elements (classes, functions) in a file
- get_code_from_probable_node_name: Get code for specific functions/classes by name
- get_code_from_node_id: Get code for specific node IDs
- get_code_from_multiple_node_ids: Get code for multiple node IDs at once
- ask_knowledge_graph_queries: Query the knowledge graph for code relationships
- get_nodes_from_tags: Find code nodes using semantic tags
- get_node_neighbours_from_node_id: Find related/connected code
- get_code_graph_from_node_id: Get code graph relationships
- intelligent_code_graph: Advanced code graph analysis

ANALYSIS STRATEGY:
1. Start with get_code_file_structure to understand project layout
2. Use fetch_file to read key files identified in the request
3. Use analyze_code_structure to understand code elements in files
4. Use get_code_from_probable_node_name to find specific functions/classes
5. Use ask_knowledge_graph_queries to understand relationships
6. Use get_node_neighbours_from_node_id to trace connections
7. Provide precise file:line references for all claims

Output Structure:
- Overview: 2-3 sentence summary
- Entry Points: List with file:line references
- Core Implementation: Detailed breakdown by component with file:line references
- Data Flow: Step-by-step flow
- Key Patterns: Architectural patterns in use
- Configuration: Config settings used
- Error Handling: How errors are handled

REMEMBER: You are a documentarian, not a critic. Your sole purpose is to explain HOW the code currently works.""",
                        expected_output="Detailed codebase analysis with precise file:line references showing how the code works",
                    )
                ],
                max_iter=15,
            ),
            AgentType.CBL: AgentConfig(
                role="Codebase Locator Specialist",
                goal="Locate files, directories, and components relevant to a feature or task - a 'Super Grep/Glob/LS tool'",
                backstory="""You are a specialist at finding WHERE code lives in a codebase. Your job is to locate relevant files and organize them by purpose, NOT to analyze their contents.

CRITICAL: YOUR ONLY JOB IS TO LOCATE AND DOCUMENT WHERE FILES EXIST
- DO NOT suggest improvements or changes unless the user explicitly asks for them
- DO NOT perform root cause analysis unless the user explicitly asks for them
- DO NOT propose future enhancements unless the user explicitly asks for them
- DO NOT critique the implementation
- DO NOT comment on code quality, architecture decisions, or best practices
- ONLY describe what exists, where it exists, and how components are organized

Core Responsibilities:
1. Find Files by Topic/Feature - Search for files containing relevant keywords
2. Categorize Findings - Group by implementation, tests, config, docs, types
3. Return Structured Results - Provide full paths, group by purpose, note directory clusters""",
                tasks=[
                    TaskConfig(
                        description="""Locate files and directories for a feature or topic using available tools:

AVAILABLE TOOLS FOR LOCATING:
- get_code_file_structure: Get directory structure and file organization
- get_nodes_from_tags: Find code nodes using semantic tags/keywords
- ask_knowledge_graph_queries: Query for files containing specific functionality
- fetch_file: Read file contents to verify relevance (use sparingly)
- analyze_code_structure: Get overview of what's in a file without reading full content

SEARCH STRATEGY:
1. Start with get_code_file_structure to understand project layout
2. Use get_nodes_from_tags to find files containing relevant keywords
3. Use ask_knowledge_graph_queries to find files by functionality
4. Consider language/framework conventions (src/, lib/, components/, etc.)
5. Look for naming patterns (*service*, *handler*, *test*, etc.)
6. Use analyze_code_structure to understand file contents without reading full files

Output Structure:
- Implementation Files: Core logic files with paths
- Test Files: Unit, integration, e2e tests
- Configuration: Config files
- Type Definitions: TypeScript types, interfaces
- Related Directories: Directory clusters with file counts
- Entry Points: Where features are imported/registered

Important:
- Don't read full file contents unless necessary - just report locations
- Group files logically by purpose
- Include directory file counts
- Check multiple extensions (.js/.ts, .py, .go, etc.)
- Use semantic search tools to find relevant files efficiently

REMEMBER: You are a file finder and organizer, documenting WHERE everything is located.""",
                        expected_output="Structured list of file locations organized by type (implementation, tests, config, docs, types)",
                    )
                ],
                max_iter=10,
            ),
            AgentType.THINK_EXECUTE: AgentConfig(
                role="Think and Execute Specialist",
                goal="Think through problems systematically and execute tasks using all available tools",
                backstory="""You are a versatile problem-solver with access to all available tools. Your job is to think through problems step-by-step and execute tasks efficiently.

You have access to the complete toolkit including:
- Code analysis and retrieval tools
- File system operations
- Knowledge graph queries
- Web search and content extraction
- Linear issue management
- GitHub integration
- Intelligent reasoning tools

Your approach should be methodical:
1. Understand the task requirements
2. Plan your approach using available tools
3. Execute the plan step by step
4. Verify results and iterate if needed""",
                tasks=[
                    TaskConfig(
                        description="""Execute assigned tasks using systematic thinking and all available tools:

AVAILABLE TOOLS (ALL TOOLS FROM TOOL SERVICE):
- fetch_file: Read complete file contents
- get_code_file_structure: Get directory structure and file organization
- analyze_code_structure: Analyze code elements (classes, functions) in a file
- get_code_from_probable_node_name: Get code for specific functions/classes by name
- get_code_from_node_id: Get code for specific node IDs
- get_code_from_multiple_node_ids: Get code for multiple node IDs at once
- ask_knowledge_graph_queries: Query the knowledge graph for code relationships
- get_nodes_from_tags: Find code nodes using semantic tags
- get_node_neighbours_from_node_id: Find related/connected code
- get_code_graph_from_node_id: Get code graph relationships
- intelligent_code_graph: Advanced code graph analysis
- change_detection: Detect changes in codebase
- get_linear_issue: Get Linear issue details
- update_linear_issue: Update Linear issues
- think: Use reasoning and thinking tools
- webpage_extractor: Extract content from web pages (if available)
- github_tool: GitHub integration tools (if available)
- web_search_tool: Search the web (if available)

EXECUTION STRATEGY:
1. Use 'think' tool to plan your approach
2. Break down complex tasks into smaller steps
3. Use appropriate tools for each step
4. Verify results and iterate as needed
5. Provide comprehensive results with evidence

TASK EXECUTION PRINCIPLES:
- Think before acting - use the 'think' tool to plan
- Use the most appropriate tools for each subtask
- Provide detailed explanations of your reasoning
- Include evidence and references for your conclusions
- Be thorough but efficient in tool usage""",
                        expected_output="Comprehensive task completion with detailed reasoning, evidence, and results",
                    )
                ],
                max_iter=20,
            ),
        }

    def _create_agent(self, agent_type: AgentType, ctx: ChatContext) -> Agent:
        """Create a specialized agent instance"""
        if agent_type in self._agent_instances:
            return self._agent_instances[agent_type]

        config = self.delegate_agents[agent_type]

        # Prepare multimodal instructions if images are present
        multimodal_instructions = self._prepare_multimodal_instructions(ctx)

        # Create MCP servers
        mcp_toolsets: List[MCPServerStreamableHTTP] = []
        for mcp_server in self.mcp_servers:
            try:
                mcp_server_instance = MCPServerStreamableHTTP(
                    url=mcp_server["link"], timeout=10.0
                )
                mcp_toolsets.append(mcp_server_instance)
                logger.info(
                    f"Successfully created MCP server: {mcp_server.get('name', 'unknown')}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create MCP server {mcp_server.get('name', 'unknown')}: {e}"
                )
                continue

        agent = Agent(
            model=self.llm_provider.get_pydantic_model(),
            tools=[
                Tool(
                    name=tool.name,
                    description=tool.description,
                    function=handle_exception(tool.func),  # type: ignore
                )
                for tool in self.tools
            ],
            mcp_servers=mcp_toolsets,
            instructions=f"""
            Role: {config.role}
            Goal: {config.goal}
            Backstory: {config.backstory}

            {multimodal_instructions}

            CURRENT CONTEXT AND AGENT TASK OVERVIEW:
            {self._create_task_description(task_config=config.tasks[0], ctx=ctx, agent_type=agent_type)}

            SPECIALIZATION: You are a {agent_type.value} specialist. Focus on your area of expertise and provide detailed, professional results.
            """,
            result_type=str,
            output_retries=3,
            output_type=str,
            defer_model_check=True,
            end_strategy="exhaustive",
            model_settings={"max_tokens": 14000},
        )

        self._agent_instances[agent_type] = agent
        return agent

    def _create_supervisor_agent(self, ctx: ChatContext) -> Agent:
        """Create the supervisor agent that coordinates other agents"""
        if self._supervisor_agent:
            return self._supervisor_agent

        # Prepare multimodal instructions if images are present
        multimodal_instructions = self._prepare_multimodal_instructions(ctx)

        # Create MCP servers
        mcp_toolsets: List[MCPServerStreamableHTTP] = []
        for mcp_server in self.mcp_servers:
            try:
                mcp_server_instance = MCPServerStreamableHTTP(
                    url=mcp_server["link"], timeout=10.0
                )
                mcp_toolsets.append(mcp_server_instance)
                logger.info(
                    f"Successfully created MCP server: {mcp_server.get('name', 'unknown')}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create MCP server {mcp_server.get('name', 'unknown')}: {e}"
                )
                continue

        # Create delegation tools for each agent type
        delegation_tools = []
        for agent_type in self.delegate_agents.keys():
            # Create highly attractive descriptions for each delegation tool
            if agent_type == AgentType.CAB:
                description = "🔍 DELEGATE TO CODEBASE ANALYZER - MANDATORY for understanding how code works! Analyzes implementation details, traces data flow, documents technical workings with precise file:line references. Use for ANY 'how does X work' questions."
            elif agent_type == AgentType.CBL:
                description = "📍 DELEGATE TO CODEBASE LOCATOR - MANDATORY for finding files/components! Acts as a Super Grep/Glob/LS tool to locate where code lives. Use for ANY 'where is X located' questions."
            elif agent_type == AgentType.THINK_EXECUTE:
                description = "🧠 DELEGATE TO THINK & EXECUTE AGENT - MANDATORY for ALL IMPLEMENTATION WORK! Builds, implements, fixes, and creates working solutions. Has access to all available tools and can write code, create files, and deliver working implementations. USE THIS TO GET THINGS BUILT AND WORKING - NOT JUST ANALYZED!"
            else:
                description = f"🤖 DELEGATE TO {agent_type.value.upper()} SPECIALIST - Use this to delegate tasks to the specialist agent"

            delegation_tools.append(
                Tool(
                    name=f"delegate_to_{agent_type.value}",
                    description=description,
                    function=self._create_delegation_function(agent_type),
                )
            )

        # Create todo management tools for supervisor
        from app.modules.intelligence.tools.todo_management_tool import (
            create_todo_management_tools,
        )

        todo_tools = create_todo_management_tools()

        supervisor_agent = Agent(
            model=self.llm_provider.get_pydantic_model(),
            tools=[
                Tool(
                    name=tool.name,
                    description=tool.description,
                    function=handle_exception(tool.func),  # type: ignore
                )
                for tool in self.tools
            ]
            + delegation_tools
            + [
                Tool(
                    name=todo_tool.name,
                    description=todo_tool.description,
                    function=handle_exception(todo_tool.func),  # type: ignore
                )
                for todo_tool in todo_tools
            ],
            mcp_servers=mcp_toolsets,
            instructions=f"""
            🚨🚨🚨 STOP! READ THIS FIRST! 🚨🚨🚨
            YOU ARE A SUPERVISOR AGENT. YOU SOLVE PROBLEMS BY ORCHESTRATING SPECIALISTS.
            YOU ARE THE PROBLEM SOLVER - BREAK DOWN WORK AND COORDINATE THE SOLUTION!

            Role: {self.config.role} - SUPERVISOR AGENT
            Goal: {self.config.goal}
            Backstory: {self.config.backstory}

            {multimodal_instructions}

            CURRENT CONTEXT AND SUPERVISOR TASK OVERVIEW:
            {self._create_supervisor_task_description(ctx)}

            🚨 CRITICAL: YOU ARE THE PROBLEM-SOLVING ORCHESTRATOR 🚨

            YOUR ROLE AS PROBLEM-SOLVING SUPERVISOR:
            - YOU are responsible for solving the user's problem
            - YOU break down complex problems into smaller, well-defined tasks
            - YOU use todo management tools to track and coordinate the solution
            - YOU delegate specific, focused subtasks to appropriate specialists
            - YOU synthesize specialist results into the final solution
            - YOU ensure the problem gets fully solved

            PROBLEM-SOLVING WORKFLOW (MANDATORY STEPS):
            1. Understand the user's problem/request
            2. Break it down into smaller, well-defined tasks using create_todo
            3. IMMEDIATELY start delegating: Mark first task as in_progress and delegate to specialist
            4. When specialist completes, add results to todo notes and mark completed
            5. Continue delegating remaining tasks one by one
            6. Synthesize all specialist results into final solution
            7. Present the complete solution to the user

            REQUIRED BEHAVIOR (MUST DO BOTH):
            ✅ Use create_todo to break down complex problems into manageable tasks
            ✅ IMMEDIATELY delegate each task using delegate_to_[agent_type] tools
            ✅ Use update_todo_status to track each delegation (pending → in_progress → completed)
            ✅ Use add_todo_note to record specialist results
            ✅ Continue delegating until all todos are completed
            ✅ Synthesize all specialist results into a comprehensive solution

            CRITICAL COMMUNICATION REQUIREMENT:
            🗣️ ALWAYS explain what you're doing before each tool call with detailed context:

            BEFORE EVERY TOOL CALL, PROVIDE CLEAR EXPLANATION:
            - Before create_todo: "I'm breaking down this problem into manageable tasks. Let me create a todo for [specific task] because [reason]..."
            - Before update_todo_status: "I'm updating task [ID/title] from [old status] to [new status] because [reason]..."
            - Before delegate_to_codebase_analyzer: "I'm delegating codebase analysis to the specialist because we need to understand [specific aspect]..."
            - Before delegate_to_codebase_locator: "I'm delegating file location to the specialist to find [specific files/components] for [purpose]..."
            - Before delegate_to_think_execute: "I'm delegating implementation work to the Think & Execute agent to [specific action] because [reason]..."
            - Before add_todo_note: "I'm recording the results from [agent] for task [ID/title]. The specialist found/completed [summary]..."
            - Before list_todos: "Let me check the current progress on all tasks to see what's completed and what needs attention next..."

            COMMUNICATION STYLE:
            ✅ Always explain the "why" behind each action
            ✅ Reference specific task IDs/titles when updating todos
            ✅ Mention which specialist you're delegating to and why they're the right choice
            ✅ Summarize key findings when recording results
            ✅ Keep the user informed of your orchestration strategy

            CRITICAL: You MUST use both todo management AND delegation tools together!

            AVAILABLE SPECIALIST AGENTS (USE THESE DELEGATION TOOLS IMMEDIATELY):
            - delegate_to_codebase_analyzer: For analyzing codebase implementation details and documenting how code works
            - delegate_to_codebase_locator: For locating files, directories, and components relevant to a feature or task
            - delegate_to_think_execute: For general problem-solving, implementation, debugging, and complex analysis tasks

            TODO MANAGEMENT TOOLS (FOR LONG-RUNNING TASKS):
            - create_todo: Create todo items to break down complex requests into manageable tasks
            - update_todo_status: Mark todos as in_progress, completed, etc.
            - list_todos: See current task state and progress
            - get_todo_summary: Get overview of all todos and their status
            - add_todo_note: Add progress notes to todos

            ORCHESTRATION WORKFLOW - COMBINE TODO MANAGEMENT + DELEGATION:

            FOR SIMPLE REQUESTS:
            1. Understand the specific task needed
            2. IMMEDIATELY delegate to the appropriate specialist using delegate_to_[agent_type]
            3. Present the result to user

            FOR COMPLEX PROBLEMS (YOUR MAIN ROLE - MUST USE BOTH TODOS + DELEGATION):
            1. Analyze the user's problem and what needs to be solved
            2. Break it down into specific, well-defined subtasks using create_todo
            3. Start working through todos systematically:
               - update_todo_status(todo_id, "in_progress")
               - IMMEDIATELY delegate_to_[appropriate_specialist](task_description)
               - add_todo_note(todo_id, specialist_results)
               - update_todo_status(todo_id, "completed")
            4. Continue until all todos are completed
            5. Synthesize all specialist results into a complete solution
            6. Present the final comprehensive solution to user

            EXAMPLE ORCHESTRATION - "Review and improve CLI implementation":
            1. create_todo("Locate CLI files", "Find all CLI-related files", "high", "codebase_locator")
            2. create_todo("Analyze CLI structure", "Examine current implementation", "high", "codebase_analyzer")
            3. create_todo("Identify improvements", "Find issues and improvements", "high", "think_execute")
            4. create_todo("Implement fixes", "Apply improvements to code", "high", "think_execute")
            5. For EACH todo: update_todo_status → delegate_to_specialist → add_todo_note → mark completed
            6. Synthesize all specialist results into final improved CLI implementation

            MANDATORY: You MUST use BOTH create_todo AND delegate_to_[agent_type] tools together!

            YOUR COORDINATION RESPONSIBILITIES:
            - Break down problems into actionable subtasks
            - Choose the right specialist for each subtask
            - Track progress and coordinate between specialists
            - Ensure nothing is missed in the solution
            - Synthesize all specialist results into the final comprehensive solution
            """,
            result_type=str,
            output_retries=3,
            output_type=str,
            defer_model_check=True,
            end_strategy="exhaustive",
            model_settings={"max_tokens": 14000},
        )

        self._supervisor_agent = supervisor_agent
        return supervisor_agent

    def _create_delegation_function(self, agent_type: AgentType):
        """Create a delegation function for a specific agent type"""

        async def delegate_function(
            ctx: RunContext[None], task_description: str, context: str = ""
        ) -> str:
            """Delegate a task to a specialized agent"""
            try:
                logger.info(
                    f"Delegating task to {agent_type.value} agent: {task_description}"
                )

                # Create the delegate agent
                delegate_agent = self._create_agent(agent_type, self._current_context)

                # Prepare the task with context
                full_task = f"""
                DELEGATED TASK: {task_description}

                CONTEXT: {context}

                Please complete this task using your specialized knowledge and available tools.
                Provide a detailed, professional response.
                """

                # Run the delegate agent
                result = await delegate_agent.run(
                    user_prompt=full_task,
                    usage=ctx.usage,  # Pass usage to track across agents
                )

                logger.info(f"Task completed by {agent_type.value} agent")
                return result.output

            except Exception as e:
                logger.error(f"Error in delegation to {agent_type.value}: {e}")
                return f"Error occurred while delegating to {agent_type.value} agent: {str(e)}"

        return delegate_function

    def _prepare_multimodal_instructions(self, ctx: ChatContext) -> str:
        """Prepare multimodal-specific instructions when images are present"""
        if not ctx.has_images():
            return ""

        all_images = ctx.get_all_images()
        current_images = ctx.get_current_images_only()
        context_images = ctx.get_context_images_only()

        return f"""
        MULTIMODAL ANALYSIS INSTRUCTIONS:
        You have access to {len(all_images)} image(s) - {len(current_images)} from the current message and {len(context_images)} from conversation history.

        CRITICAL GUIDELINES FOR ACCURATE ANALYSIS:
        1. **ONLY analyze what you can clearly see** - Do not infer or guess about unclear details
        2. **Distinguish between current and historical images** - Focus primarily on current message images
        3. **State uncertainty** - If you cannot clearly see something, say "I cannot clearly see..." instead of guessing
        4. **Be specific** - Reference exact text, colors, shapes, or elements you observe
        5. **Avoid assumptions** - Do not assume context beyond what's explicitly visible

        ANALYSIS APPROACH:
        - **Current Images**: These are directly related to the user's current query
        - **Historical Images**: These provide context but may not be directly relevant
        - **Text Recognition**: Only transcribe text that is clearly legible
        - **UI Elements**: Only describe elements that are clearly visible and identifiable
        - **Error Messages**: Only report errors that are clearly displayed and readable

        IMPORTANT: If an image is unclear, blurry, or doesn't contain the type of content the user is asking about, explicitly state this rather than making assumptions.
        """

    def _create_task_description(
        self,
        task_config: TaskConfig,
        ctx: ChatContext,
        agent_type: AgentType,
    ) -> str:
        """Create a task description from task configuration for delegate agents"""
        if ctx.node_ids is None:
            ctx.node_ids = []
        if isinstance(ctx.node_ids, str):
            ctx.node_ids = [ctx.node_ids]

        # Add image context information
        image_context = ""
        if ctx.has_images():
            all_images = ctx.get_all_images()
            image_details = []
            for attachment_id, image_data in all_images.items():
                file_name = image_data.get("file_name", "unknown")
                file_size = image_data.get("file_size", 0)
                image_details.append(f"- {file_name} ({file_size} bytes)")

            image_context = f"""
            ATTACHED IMAGES:
            {chr(10).join(image_details)}

            Image Analysis Notes:
            - These images are provided for visual analysis and debugging
            - Reference specific details from the images in your response
            - Correlate visual evidence with the user's query
            """

        return f"""
                CONTEXT:
                Project ID: {ctx.project_id}
                Node IDs: {" ,".join(ctx.node_ids)}
                Project Name (this is name from github. i.e. owner/repo): {ctx.project_name}

                {image_context}

                Additional Context:
                {ctx.additional_context if ctx.additional_context != "" else "no additional context"}

                TASK HANDLING: (follow the method below if the user asks you to execute your task for your role and goal)
                {task_config.description}

                INSTRUCTIONS:
                1. Use the available tools to gather information
                2. {"Analyze the provided images in detail and " if ctx.has_images() else ""}Process and synthesize the gathered information
                3. Format your response in markdown unless explicitly asked to output in a different format, make sure it's well formatted
                4. Include relevant code snippets and file references
                5. {"Reference specific details from the images when relevant" if ctx.has_images() else "Provide clear explanations"}
                6. Verify your output before submitting

                IMPORTANT:
                - Use tools efficiently and avoid unnecessary API calls
                - Only use the tools listed below
                - You have access to tools in MCP Servers too, use them effectively. These mcp servers provide you with tools user might ask you to perform tasks on
                {"- Provide detailed image analysis when images are present" if ctx.has_images() else ""}
            """

    def _create_supervisor_task_description(self, ctx: ChatContext) -> str:
        """Create a task description for the supervisor agent"""
        if ctx.node_ids is None:
            ctx.node_ids = []
        if isinstance(ctx.node_ids, str):
            ctx.node_ids = [ctx.node_ids]

        # Add image context information
        image_context = ""
        if ctx.has_images():
            all_images = ctx.get_all_images()
            image_details = []
            for attachment_id, image_data in all_images.items():
                file_name = image_data.get("file_name", "unknown")
                file_size = image_data.get("file_size", 0)
                image_details.append(f"- {file_name} ({file_size} bytes)")

            image_context = f"""
            ATTACHED IMAGES:
            {chr(10).join(image_details)}

            Image Analysis Notes:
            - These images are provided for visual analysis and debugging
            - Reference specific details from the images in your response
            - Correlate visual evidence with the user's query
            """

        return f"""
                CONTEXT:
                Project ID: {ctx.project_id}
                Node IDs: {" ,".join(ctx.node_ids)}
                Project Name (this is name from github. i.e. owner/repo): {ctx.project_name}

                {image_context}

                Additional Context:
                {ctx.additional_context if ctx.additional_context != "" else "no additional context"}

                SUPERVISOR TASK HANDLING:
                {self.config.tasks[0].description if self.config.tasks else "Coordinate and delegate tasks to appropriate specialist agents"}

                SUPERVISOR INSTRUCTIONS (ORCHESTRATION-FOCUSED):
                1. Understand the user's problem and what needs to be solved
                2. Break complex problems into specific, well-defined subtasks using create_todo
                3. Delegate each subtask to the appropriate specialist
                4. Track progress and coordinate results using todo management tools
                5. Synthesize all specialist results into a complete solution
                6. Format response in markdown unless explicitly asked otherwise

                ORCHESTRATION DELEGATION RULES (MUST USE BOTH TOOLS):
                - Break down complex problems using create_todo
                - IMMEDIATELY delegate each todo using delegate_to_[agent_type] tools
                - File/component location tasks → delegate_to_codebase_locator
                - Code analysis tasks → delegate_to_codebase_analyzer
                - Implementation/problem-solving tasks → delegate_to_think_execute
                - Track each delegation using update_todo_status

                MANDATORY WORKFLOW FOR EACH TODO:
                1. create_todo(title, description, priority, assigned_agent)
                2. update_todo_status(todo_id, "in_progress")
                3. delegate_to_[appropriate_agent](task_description)
                4. add_todo_note(todo_id, specialist_results)
                5. update_todo_status(todo_id, "completed")
                6. Move to next todo

                TODO + DELEGATION INTEGRATION:
                - Every todo MUST be followed by immediate delegation
                - Every delegation result MUST be recorded in todo notes
                - Track progress through both todo status and delegation completion
                - Use list_todos to see what needs to be delegated next

                CRITICAL RULES:
                - NEVER create todos without delegating them
                - NEVER delegate without tracking in todos
                - You MUST use BOTH todo management AND delegation tools
                - Each subtask gets both a todo AND a delegation
                - Synthesize all specialist results into comprehensive solutions
                {"- Provide detailed image analysis when images are present" if ctx.has_images() else ""}
            """

    def _create_multimodal_user_content(
        self, ctx: ChatContext
    ) -> Sequence[UserContent]:
        """Create multimodal user content with images using PydanticAI's ImageUrl"""
        content: List[UserContent] = [ctx.query]

        # Add current images to the content
        current_images = ctx.get_current_images_only()
        logger.info(
            f"Processing {len(current_images)} current images for multimodal content"
        )

        for attachment_id, image_data in current_images.items():
            try:
                # Validate image data structure
                if not isinstance(image_data, dict):
                    logger.error(
                        f"Invalid image data structure for {attachment_id}: {type(image_data)}"
                    )
                    continue

                # Validate required fields
                if "base64" not in image_data:
                    logger.error(f"Missing base64 data for image {attachment_id}")
                    continue

                base64_data = image_data["base64"]
                if not isinstance(base64_data, str) or not base64_data:
                    logger.error(f"Invalid base64 data for image {attachment_id}")
                    continue

                # Validate base64 format
                try:
                    import base64

                    base64.b64decode(base64_data)
                except Exception as e:
                    logger.error(
                        f"Invalid base64 format for image {attachment_id}: {str(e)}"
                    )
                    continue

                # Get mime type with better fallback
                mime_type = image_data.get("mime_type", "image/jpeg")
                if (
                    not mime_type
                    or not isinstance(mime_type, str)
                    or not mime_type.startswith("image/")
                ):
                    logger.warning(
                        f"Invalid mime type for image {attachment_id}: {mime_type}, defaulting to image/jpeg"
                    )
                    mime_type = "image/jpeg"

                # Create data URL
                data_url = f"data:{mime_type};base64,{base64_data}"

                # Log image details for debugging
                file_name = image_data.get("file_name", "unknown")
                file_size = image_data.get("file_size", 0)
                logger.info(
                    f"Adding image {attachment_id} ({file_name}, {file_size} bytes, {mime_type}) to multimodal content"
                )

                content.append(ImageUrl(url=data_url))
                logger.info(
                    f"Successfully added image {attachment_id} to multimodal content"
                )

            except Exception as e:
                logger.error(
                    f"Failed to add image {attachment_id} to content: {str(e)}",
                    exc_info=True,
                )
                continue

        # If no current images, add context images as fallback
        if not current_images:
            context_images = ctx.get_context_images_only()
            logger.info(
                f"No current images found, processing {len(context_images)} context images as fallback"
            )

            for attachment_id, image_data in context_images.items():
                try:
                    # Apply same validation as above
                    if not isinstance(image_data, dict) or "base64" not in image_data:
                        logger.error(f"Invalid context image data for {attachment_id}")
                        continue

                    base64_data = image_data["base64"]
                    if not isinstance(base64_data, str) or not base64_data:
                        logger.error(
                            f"Invalid base64 data for context image {attachment_id}"
                        )
                        continue

                    # Validate base64 format
                    try:
                        import base64

                        base64.b64decode(base64_data)
                    except Exception as e:
                        logger.error(
                            f"Invalid base64 format for context image {attachment_id}: {str(e)}"
                        )
                        continue

                    mime_type = image_data.get("mime_type", "image/jpeg")
                    if (
                        not mime_type
                        or not isinstance(mime_type, str)
                        or not mime_type.startswith("image/")
                    ):
                        mime_type = "image/jpeg"

                    data_url = f"data:{mime_type};base64,{base64_data}"

                    file_name = image_data.get("file_name", "unknown")
                    file_size = image_data.get("file_size", 0)
                    logger.info(
                        f"Adding context image {attachment_id} ({file_name}, {file_size} bytes, {mime_type}) to multimodal content"
                    )

                    content.append(ImageUrl(url=data_url))
                    logger.info(
                        f"Successfully added context image {attachment_id} to multimodal content"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to add context image {attachment_id} to content: {str(e)}",
                        exc_info=True,
                    )
                    continue

        logger.info(
            f"Final multimodal content has {len(content)} elements: 1 text + {len(content)-1} images"
        )
        return content

    async def _prepare_multimodal_message_history(
        self, ctx: ChatContext
    ) -> List[ModelMessage]:
        """Prepare message history with multimodal support"""
        history_messages = []

        for msg in ctx.history:
            # For now, keep history as text-only to avoid token bloat
            # Images are only added to the current query
            history_messages.append(ModelResponse([TextPart(content=str(msg))]))

        return history_messages

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Main execution flow with multi-agent coordination"""
        logger.info(
            f"Running pydantic multi-agent system {'with multimodal support' if ctx.has_images() else ''}"
        )

        # Store context for delegation functions
        self._current_context = ctx

        # Check if we have images and if the model supports vision
        if ctx.has_images() and self.llm_provider.is_vision_model():
            logger.info(
                f"Processing {len(ctx.get_all_images())} images with PydanticAI multimodal multi-agent"
            )
            return await self._run_multimodal(ctx)
        else:
            if ctx.has_images() and not self.llm_provider.is_vision_model():
                logger.warning(
                    "Images provided but current model doesn't support vision, proceeding with text-only"
                )
            # Use standard PydanticAI multi-agent for text-only
            return await self._run_standard(ctx)

    async def _run_standard(self, ctx: ChatContext) -> ChatAgentResponse:
        """Standard text-only multi-agent execution"""
        try:
            # Prepare message history
            message_history = await self._prepare_multimodal_message_history(ctx)

            # Create and run supervisor agent
            supervisor_agent = self._create_supervisor_agent(ctx)

            # Try to initialize MCP servers with timeout handling
            try:
                async with supervisor_agent.run_mcp_servers():
                    resp = await supervisor_agent.run(
                        user_prompt=ctx.query,
                        message_history=message_history,
                    )
            except (TimeoutError, anyio.WouldBlock, Exception) as mcp_error:
                logger.warning(f"MCP server initialization failed: {mcp_error}")
                logger.info("Continuing without MCP servers...")

                # Fallback: run without MCP servers
                resp = await supervisor_agent.run(
                    user_prompt=ctx.query,
                    message_history=message_history,
                )

            return ChatAgentResponse(
                response=resp.output,
                tool_calls=[],
                citations=[],
            )

        except Exception as e:
            logger.error(
                f"Error in standard multi-agent run method: {str(e)}", exc_info=True
            )
            return ChatAgentResponse(
                response=f"An error occurred while processing your request: {str(e)}",
                tool_calls=[],
                citations=[],
            )

    async def _run_multimodal(self, ctx: ChatContext) -> ChatAgentResponse:
        """Multimodal multi-agent execution using PydanticAI's native multimodal capabilities"""
        try:
            # Create multimodal user content with images
            multimodal_content = self._create_multimodal_user_content(ctx)

            # Prepare message history (text-only for now to avoid token bloat)
            message_history = await self._prepare_multimodal_message_history(ctx)

            # Create and run supervisor agent
            supervisor_agent = self._create_supervisor_agent(ctx)

            resp = await supervisor_agent.run(
                user_prompt=multimodal_content,
                message_history=message_history,
            )

            return ChatAgentResponse(
                response=resp.output,
                tool_calls=[],
                citations=[],
            )

        except Exception as e:
            logger.error(
                f"Error in multimodal multi-agent run method: {str(e)}", exc_info=True
            )
            # Fallback to standard execution
            logger.info("Falling back to standard text-only execution")
            return await self._run_standard(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Stream multi-agent response with delegation support"""
        logger.info(
            f"Running pydantic multi-agent stream {'with multimodal support' if ctx.has_images() else ''}"
        )

        # Store context for delegation functions
        self._current_context = ctx

        # Check if we have images and if the model supports vision
        if ctx.has_images() and self.llm_provider.is_vision_model():
            logger.info(
                f"Processing {len(ctx.get_all_images())} images with PydanticAI multimodal multi-agent streaming"
            )
            async for chunk in self._run_multimodal_stream(ctx):
                yield chunk
        else:
            if ctx.has_images() and not self.llm_provider.is_vision_model():
                logger.warning(
                    "Images provided but current model doesn't support vision, proceeding with text-only streaming"
                )
            # Use standard PydanticAI multi-agent streaming for text-only
            async for chunk in self._run_standard_stream(ctx):
                yield chunk

    async def _run_multimodal_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Stream multimodal multi-agent response using PydanticAI's native capabilities"""
        try:
            # Create multimodal user content with images
            multimodal_content = self._create_multimodal_user_content(ctx)

            # Prepare message history (text-only for now to avoid token bloat)
            message_history = await self._prepare_multimodal_message_history(ctx)

            # Create supervisor agent
            supervisor_agent = self._create_supervisor_agent(ctx)

            # Stream the response
            async with supervisor_agent.iter(
                user_prompt=multimodal_content,
                message_history=message_history,
            ) as run:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        # A model request node => We can stream tokens from the model's request
                        async with node.stream(run.ctx) as request_stream:
                            async for event in request_stream:
                                if isinstance(event, PartStartEvent) and isinstance(
                                    event.part, TextPart
                                ):
                                    yield ChatAgentResponse(
                                        response=event.part.content,
                                        tool_calls=[],
                                        citations=[],
                                    )
                                if isinstance(event, PartDeltaEvent) and isinstance(
                                    event.delta, TextPartDelta
                                ):
                                    yield ChatAgentResponse(
                                        response=event.delta.content_delta,
                                        tool_calls=[],
                                        citations=[],
                                    )

                    elif Agent.is_call_tools_node(node):
                        async with node.stream(run.ctx) as handle_stream:
                            async for event in handle_stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    yield ChatAgentResponse(
                                        response="",
                                        tool_calls=[create_tool_call_response(event)],
                                        citations=[],
                                    )
                                if isinstance(event, FunctionToolResultEvent):
                                    yield ChatAgentResponse(
                                        response="",
                                        tool_calls=[create_tool_result_response(event)],
                                        citations=[],
                                    )

                    elif Agent.is_end_node(node):
                        logger.info(
                            "multimodal multi-agent result streamed successfully!!"
                        )

        except Exception as e:
            logger.error(
                f"Error in multimodal multi-agent stream: {str(e)}", exc_info=True
            )
            # Fallback to standard streaming
            async for chunk in self._run_standard_stream(ctx):
                yield chunk

    async def _run_standard_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Standard multi-agent streaming execution with MCP server support"""
        # Create supervisor agent directly
        supervisor_agent = self._create_supervisor_agent(ctx)

        try:
            # Try to initialize MCP servers with timeout handling
            try:
                async with supervisor_agent.run_mcp_servers():
                    async with supervisor_agent.iter(
                        user_prompt=ctx.query,
                        message_history=[
                            ModelResponse([TextPart(content=msg)])
                            for msg in ctx.history
                        ],
                    ) as run:
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                # A model request node => We can stream tokens from the model's request
                                try:
                                    async with node.stream(run.ctx) as request_stream:
                                        async for event in request_stream:
                                            if isinstance(
                                                event, PartStartEvent
                                            ) and isinstance(event.part, TextPart):
                                                yield ChatAgentResponse(
                                                    response=event.part.content,
                                                    tool_calls=[],
                                                    citations=[],
                                                )
                                            if isinstance(
                                                event, PartDeltaEvent
                                            ) and isinstance(
                                                event.delta, TextPartDelta
                                            ):
                                                yield ChatAgentResponse(
                                                    response=event.delta.content_delta,
                                                    tool_calls=[],
                                                    citations=[],
                                                )
                                except (
                                    ModelRetry,
                                    AgentRunError,
                                    UserError,
                                ) as pydantic_error:
                                    logger.warning(
                                        f"Pydantic-ai error in model request stream: {pydantic_error}"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*Encountered an issue while processing your request. Trying to recover...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue
                                except anyio.WouldBlock:
                                    logger.warning(
                                        "Model request stream would block - continuing..."
                                    )
                                    continue
                                except Exception as e:
                                    logger.error(
                                        f"Unexpected error in model request stream: {e}"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*An unexpected error occurred. Continuing...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue

                            elif Agent.is_call_tools_node(node):
                                try:
                                    async with node.stream(run.ctx) as handle_stream:
                                        async for event in handle_stream:
                                            if isinstance(event, FunctionToolCallEvent):
                                                yield ChatAgentResponse(
                                                    response="",
                                                    tool_calls=[
                                                        create_tool_call_response(event)
                                                    ],
                                                    citations=[],
                                                )
                                            if isinstance(
                                                event, FunctionToolResultEvent
                                            ):
                                                yield ChatAgentResponse(
                                                    response="",
                                                    tool_calls=[
                                                        create_tool_result_response(
                                                            event
                                                        )
                                                    ],
                                                    citations=[],
                                                )
                                except (
                                    ModelRetry,
                                    AgentRunError,
                                    UserError,
                                ) as pydantic_error:
                                    logger.warning(
                                        f"Pydantic-ai error in tool call stream: {pydantic_error}"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*Encountered an issue while calling tools. Trying to recover...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue
                                except anyio.WouldBlock:
                                    logger.warning(
                                        "Tool call stream would block - continuing..."
                                    )
                                    continue
                                except Exception as e:
                                    logger.error(
                                        f"Unexpected error in tool call stream: {e}"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*An unexpected error occurred during tool execution. Continuing...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue

                            elif Agent.is_end_node(node):
                                logger.info(
                                    "multi-agent result streamed successfully!!"
                                )

            except (TimeoutError, anyio.WouldBlock, Exception) as mcp_error:
                logger.warning(f"MCP server initialization failed: {mcp_error}")
                logger.info("Continuing without MCP servers...")

                # Fallback: run without MCP servers
                try:
                    async with supervisor_agent.iter(
                        user_prompt=ctx.query,
                        message_history=[
                            ModelResponse([TextPart(content=msg)])
                            for msg in ctx.history
                        ],
                    ) as run:
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                try:
                                    async with node.stream(run.ctx) as request_stream:
                                        async for event in request_stream:
                                            if isinstance(
                                                event, PartStartEvent
                                            ) and isinstance(event.part, TextPart):
                                                yield ChatAgentResponse(
                                                    response=event.part.content,
                                                    tool_calls=[],
                                                    citations=[],
                                                )
                                            if isinstance(
                                                event, PartDeltaEvent
                                            ) and isinstance(
                                                event.delta, TextPartDelta
                                            ):
                                                yield ChatAgentResponse(
                                                    response=event.delta.content_delta,
                                                    tool_calls=[],
                                                    citations=[],
                                                )
                                except (
                                    ModelRetry,
                                    AgentRunError,
                                    UserError,
                                ) as pydantic_error:
                                    logger.warning(
                                        f"Pydantic-ai error in fallback model request stream: {pydantic_error}"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*Encountered an issue while processing your request. Trying to recover...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue
                                except anyio.WouldBlock:
                                    logger.warning(
                                        "Model request stream would block - continuing..."
                                    )
                                    continue
                                except Exception as e:
                                    logger.error(
                                        f"Unexpected error in fallback model request stream: {e}"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*An unexpected error occurred. Continuing...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue

                            elif Agent.is_call_tools_node(node):
                                try:
                                    async with node.stream(run.ctx) as handle_stream:
                                        async for event in handle_stream:
                                            if isinstance(event, FunctionToolCallEvent):
                                                yield ChatAgentResponse(
                                                    response="",
                                                    tool_calls=[
                                                        create_tool_call_response(event)
                                                    ],
                                                    citations=[],
                                                )
                                            if isinstance(
                                                event, FunctionToolResultEvent
                                            ):
                                                yield ChatAgentResponse(
                                                    response="",
                                                    tool_calls=[
                                                        create_tool_result_response(
                                                            event
                                                        )
                                                    ],
                                                    citations=[],
                                                )
                                except (
                                    ModelRetry,
                                    AgentRunError,
                                    UserError,
                                ) as pydantic_error:
                                    logger.warning(
                                        f"Pydantic-ai error in fallback tool call stream: {pydantic_error}"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*Encountered an issue while calling tools. Trying to recover...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue
                                except anyio.WouldBlock:
                                    logger.warning(
                                        "Tool call stream would block - continuing..."
                                    )
                                    continue
                                except Exception as e:
                                    logger.error(
                                        f"Unexpected error in fallback tool call stream: {e}"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*An unexpected error occurred during tool execution. Continuing...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue

                            elif Agent.is_end_node(node):
                                logger.info(
                                    "fallback multi-agent result streamed successfully!!"
                                )

                except (ModelRetry, AgentRunError, UserError) as pydantic_error:
                    logger.error(
                        f"Pydantic-ai error in fallback agent iteration: {pydantic_error}"
                    )
                    yield ChatAgentResponse(
                        response=f"\n\n*The multi-agent system encountered an error while processing your request: {str(pydantic_error)}*\n\n",
                        tool_calls=[],
                        citations=[],
                    )
                except Exception as e:
                    logger.error(f"Unexpected error in fallback agent iteration: {e}")
                    yield ChatAgentResponse(
                        response=f"\n\n*An unexpected error occurred: {str(e)}*\n\n",
                        tool_calls=[],
                        citations=[],
                    )

        except (ModelRetry, AgentRunError, UserError) as pydantic_error:
            logger.error(
                f"Pydantic-ai error in multi-agent run_stream method: {str(pydantic_error)}",
                exc_info=True,
            )
            yield ChatAgentResponse(
                response=f"\n\n*The multi-agent system encountered an error: {str(pydantic_error)}*\n\n",
                tool_calls=[],
                citations=[],
            )
        except Exception as e:
            logger.error(
                f"Error in multi-agent run_stream method: {str(e)}", exc_info=True
            )
            yield ChatAgentResponse(
                response="\n\n*An error occurred during multi-agent streaming*\n\n",
                tool_calls=[],
                citations=[],
            )
