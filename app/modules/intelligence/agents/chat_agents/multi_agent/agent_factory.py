"""Agent factory for creating supervisor and delegate agents"""

from typing import List, Dict, Callable, Any
from pydantic_ai import Agent, Tool
from pydantic_ai.mcp import MCPServerStreamableHTTP
from langchain_core.tools import StructuredTool

from .utils.delegation_utils import AgentType
from .utils.tool_utils import wrap_structured_tools, deduplicate_tools_by_name
from .agent_instructions import (
    DELEGATE_AGENT_INSTRUCTIONS,
    get_integration_agent_instructions,
    get_supervisor_instructions,
    prepare_multimodal_instructions,
)
from .utils.context_utils import create_supervisor_task_description
from app.modules.intelligence.agents.chat_agent import ChatContext
from ..agent_config import AgentConfig, TaskConfig
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class AgentFactory:
    """Factory for creating supervisor and delegate agents"""

    def __init__(
        self,
        llm_provider: ProviderService,
        tools: List[StructuredTool],
        mcp_servers: List[dict] | None,
        delegate_agents: Dict[AgentType, AgentConfig],
        history_processor: Any,
        create_delegation_function: Callable[[AgentType], Callable],
    ):
        """Initialize the agent factory"""
        self.llm_provider = llm_provider
        self.tools = tools
        self.mcp_servers = mcp_servers or []
        self.delegate_agents = delegate_agents
        self.history_processor = history_processor
        self.create_delegation_function = create_delegation_function

        # Clean tool names (no spaces for pydantic agents)
        import re

        for i, tool in enumerate(tools):
            tools[i].name = re.sub(r" ", "", tool.name)

        # Cache for agent instances - keyed by (agent_type, conversation_id) to avoid stale context
        self._agent_instances: Dict[tuple[AgentType, str], Agent] = {}
        # Cache for supervisor agents - keyed by conversation_id to avoid stale context
        self._supervisor_agents: Dict[str, Agent] = {}

    def create_mcp_servers(self) -> List[MCPServerStreamableHTTP]:
        """Create MCP server instances from configuration"""
        mcp_toolsets: List[MCPServerStreamableHTTP] = []
        for mcp_server in self.mcp_servers:
            try:
                mcp_server_instance = MCPServerStreamableHTTP(
                    url=mcp_server["link"], timeout=10.0
                )
                mcp_toolsets.append(mcp_server_instance)
            except Exception as e:
                logger.warning(
                    f"Failed to create MCP server {mcp_server.get('name', 'unknown')}: {e}"
                )
                continue
        return mcp_toolsets

    def _filter_tools_by_names(self, tool_names: List[str]) -> List[StructuredTool]:
        """Filter tools by a list of tool names. Uses keyword matching since tool names may vary."""
        filtered = []
        seen_tool_ids = set()  # Track tool IDs already added to avoid duplicates

        # Normalize tool names to keywords (lowercase, split by underscore)
        tool_keywords = {}
        for name in tool_names:
            # Convert "get_jira_issue" -> ["get", "jira", "issue"]
            keywords = [k for k in name.lower().split("_") if k]  # Filter empty strings
            tool_keywords[name] = keywords

        for tool in self.tools:
            tool_id = id(tool)
            if tool_id in seen_tool_ids:
                continue  # Skip if already added

            # Tool names are already cleaned (spaces removed) in __init__
            clean_name = tool.name.lower()

            # Check if this tool matches any of the requested tool names
            for requested_name, keywords in tool_keywords.items():
                # Match if all keywords from requested name appear in cleaned tool name
                if keywords and all(keyword in clean_name for keyword in keywords):
                    filtered.append(tool)
                    seen_tool_ids.add(tool_id)
                    break  # Don't check other patterns for this tool
        return filtered

    def build_integration_agent_tools(self, agent_type: AgentType) -> List[Tool]:
        """Build tool list for integration-specific agents

        Integration agents receive ONLY their domain-specific tools - no code changes tools,
        no todo tools, no requirement tools. They focus exclusively on their integration domain.
        """
        # Define integration-specific tool names
        integration_tools_map = {
            AgentType.JIRA: [
                "get_jira_issue",
                "search_jira_issues",
                "create_jira_issue",
                "update_jira_issue",
                "add_jira_comment",
                "transition_jira_issue",
                "get_jira_projects",
                "get_jira_project_details",
                "get_jira_project_users",
                "link_jira_issues",
            ],
            AgentType.GITHUB: [
                "github_tool",
                "code_provider_tool",  # Also available as github_tool
                "github_create_branch",
                "code_provider_create_branch",  # Also available as github_create_branch
                "github_create_pull_request",
                "code_provider_create_pr",  # Also available as github_create_pull_request
                "github_add_pr_comments",
                "code_provider_add_pr_comments",  # Also available as github_add_pr_comments
                "github_update_branch",
                "code_provider_update_file",  # Also available as github_update_branch
            ],
            AgentType.CONFLUENCE: [
                "get_confluence_spaces",
                "get_confluence_page",
                "search_confluence_pages",
                "get_confluence_space_pages",
                "create_confluence_page",
                "update_confluence_page",
                "add_confluence_comment",
            ],
            AgentType.LINEAR: [
                "get_linear_issue",
                "update_linear_issue",
            ],
        }

        # Integration agents get ONLY their domain-specific tools - no common tools, no code changes tools
        integration_tool_names = integration_tools_map.get(agent_type, [])
        integration_tools = self._filter_tools_by_names(integration_tool_names)

        # Return ONLY integration-specific tools (no code changes, no todo, no requirements)
        return wrap_structured_tools(integration_tools)

    def build_delegate_agent_tools(self) -> List[Tool]:
        """Build the tool list for delegate agents - includes supervisor tools EXCEPT delegation, todo, and requirement tools.

        Subagents get code execution tools and code changes tools, but NOT:
        - Delegation tools (they don't delegate)
        - Todo management tools (supervisor-only for coordination)
        - Requirement verification tools (supervisor-only for verification)

        This ensures subagents focus on execution while the supervisor handles coordination and verification.
        """
        # Import tools here to avoid circular imports
        from app.modules.intelligence.tools.code_changes_manager import (
            create_code_changes_management_tools,
        )

        code_changes_tools = create_code_changes_management_tools()

        # Filter out todo and requirement tools from self.tools (supervisor-only)
        # These tool names should not be available to subagents
        supervisor_only_tool_names = {
            # Todo management tools
            "create_todo",
            "update_todo_status",
            "add_todo_note",
            "get_todo",
            "list_todos",
            "get_todo_summary",
            # Requirement verification tools
            "add_requirements",
            "delete_requirements",
            "get_requirements",
        }

        # Filter tools to exclude supervisor-only tools
        filtered_tools = [
            tool for tool in self.tools if tool.name not in supervisor_only_tool_names
        ]

        # Subagents get execution tools and code changes, but NOT todo/requirement tools
        all_tools = wrap_structured_tools(filtered_tools) + wrap_structured_tools(
            code_changes_tools
        )
        return deduplicate_tools_by_name(all_tools)

    def _get_delegation_tool_description(self, agent_type: AgentType) -> str:
        """Get the description for a delegation tool based on agent type.

        These tools delegate tasks to specialized subagents that can execute
        the work in isolated contexts.
        """
        if agent_type == AgentType.THINK_EXECUTE:
            description = """🔨 DELEGATE TO SUBAGENT - Spin up a focused worker with ALL your tools to execute specific tasks or reason through problems.

**WHAT IS A SUBAGENT:**
- A subagent is an isolated execution context with access to ALL your tools (except delegation)
- It receives ONLY what you explicitly provide: task_description and context
- It does NOT inherit your conversation history - it starts fresh with just your input
- Its work streams back to the user in real-time, then you get a summary result

**WHY DELEGATE:**
1. **Context Efficiency**: Your context stays clean - subagent's tool calls don't bloat your history
2. **Token Savings**: Heavy tool usage happens in subagent's context, not yours
3. **Parallelization**: Spin up MULTIPLE subagents simultaneously for independent tasks
4. **Focus**: Subagents work on one specific task without distraction
5. **Reasoning Tool**: Use delegation to pause and think through complex problems in isolation

**WHEN TO DELEGATE:**
- ✅ **Reasoning & Thinking**: When you need to pause, recollect thoughts, and figure out a problem - delegate reasoning tasks to think through the situation
- ✅ Any task requiring multiple tool calls (searches, file reads, analysis)
- ✅ Code implementations in specific files/modules
- ✅ Debugging and investigation tasks
- ✅ Code analysis and understanding tasks
- ✅ Research and comparison tasks
- ✅ ANY task you want executed in isolation

**CRITICAL - CONTEXT PARAMETER:**
Since subagents DON'T get your history, YOU MUST provide comprehensive context:
- File paths and line numbers you've already identified
- Code snippets relevant to the task
- Analysis results or findings from your previous work
- Configuration values, error messages, or specific details
- Everything the subagent needs to execute WITHOUT re-fetching

**PARALLELIZATION:**
Call this tool MULTIPLE TIMES in parallel for independent tasks:
- E.g., "Analyze module A" and "Analyze module B" simultaneously
- Each subagent works independently with its own context
- Results stream back and you coordinate the final synthesis

**OUTPUT:**
- Subagent streams its work in real-time (visible to user)
- You receive a "## Task Result" summary for coordination
- Use the result to inform your next steps"""
        elif agent_type == AgentType.JIRA:
            description = """🎫 DELEGATE TO JIRA AGENT - Specialized subagent for ALL Jira operations.

**CRITICAL - USE FOR ANY JIRA-RELATED TASK:**
This agent handles ALL Jira operations. Use it whenever the task involves:
- Jira, issues, tickets, projects, workflows, boards
- Creating or managing Jira content
- ANY mention of Jira in the user's request

**WHAT IS THE JIRA AGENT:**
- An isolated execution context with Jira-specific tools (issue management, search, comments, transitions)
- Receives ONLY what you provide: task_description + context
- Does NOT inherit conversation history - starts fresh
- Streams work to user, returns summary to you
- Will tell you if it CANNOT complete the task - listen to its feedback

**WHEN TO USE (use liberally for Jira tasks):**
- ✅ Creating, updating, or searching Jira issues
- ✅ Adding comments to issues
- ✅ Transitioning issues between statuses (To Do → In Progress → Done)
- ✅ Finding projects, users, or issue details
- ✅ Linking issues together
- ✅ ANY task involving Jira, issues, tickets, projects, or workflows
- ✅ When user mentions "Jira", "issue", "ticket", "project", "workflow"

**AVAILABLE OPERATIONS:**
- Search issues with JQL queries
- Get/create/update issues
- Add comments and transition statuses
- Get projects, users, and project details
- Link related issues

**CRITICAL - CONTEXT PARAMETER:**
Provide comprehensive context:
- Issue keys (e.g., PROJ-123) if known
- Project names or keys
- JQL queries or search criteria if needed
- Field values for creation/updates
- Status names for transitions

**NOTE**: The Jira agent will inform you if it cannot complete a task. If it says it cannot do something, adjust your approach or use a different subagent.

**PARALLELIZATION:** Call multiple times in parallel for independent Jira operations."""
        elif agent_type == AgentType.GITHUB:
            description = """🐙 DELEGATE TO GITHUB AGENT - Specialized subagent for ALL GitHub repository operations.

**CRITICAL - USE FOR ANY GITHUB-RELATED TASK:**
This agent handles ALL GitHub operations. Use it whenever the task involves:
- **GitHub, pull requests, PRs, branches, commits, repository operations**
- **Fetching GitHub issues** (open issues, closed issues, specific issues)
- **Creating or managing GitHub content** (PRs, branches, comments)
- **ANY mention of GitHub, repository, issues, PRs, branches, or commits in the user's request**

**WHAT IS THE GITHUB AGENT:**
- An isolated execution context with **ONLY GitHub-specific tools** (github_tool, github_create_branch, github_create_pull_request, etc.)
- **Does NOT have code changes tools, todo tools, or any other tools** - only GitHub operations
- Receives ONLY what you provide: task_description + context
- Does NOT inherit conversation history - starts fresh
- Streams work to user, returns summary to you
- Will tell you if it CANNOT complete the task - listen to its feedback

**WHEN TO USE (use liberally for GitHub tasks):**
- ✅ **Fetching GitHub issues** (all open issues, specific issues, issue details) - **PRIMARY USE CASE**
- ✅ **Fetching pull requests** (PRs, PR details, PR diffs)
- ✅ Creating pull requests and branches
- ✅ Updating files in branches
- ✅ Adding PR review comments with code references
- ✅ Managing repository operations
- ✅ **ANY task involving GitHub, repository, issues, PRs, branches, or commits**
- ✅ When user asks to "list issues", "fetch issues", "get issues", "show PRs", "create PR", etc.

**CRITICAL - GITHUB ISSUE FETCHING:**
- The GitHub agent uses `github_tool` to fetch issues and PRs
- To fetch all open issues: Provide `repo_name="owner/repo"` in context
- Example task: "Fetch and list all open issues in the nndn/coin_game repository"
- The agent will use `github_tool(repo_name="nndn/coin_game", is_pull_request=False, issue_number=None)`

**AVAILABLE OPERATIONS:**
- **Fetch issues/PRs** using github_tool (PRIMARY tool for GitHub data)
- Create branches and pull requests
- Update files in branches (commit changes)
- Add PR comments with code snippet references

**CRITICAL - CONTEXT PARAMETER:**
Provide comprehensive context:
- **Repository name** (e.g., "owner/repo" or "nndn/coin_game") - **REQUIRED for issue/PR fetching**
- Branch names (source and target)
- PR numbers or issue numbers if known
- File paths and content for updates
- Commit messages and descriptions

**NOTE**: The GitHub agent will inform you if it cannot complete a task. If it says it cannot do something, adjust your approach or use a different subagent.

**PARALLELIZATION:** Call multiple times in parallel for independent GitHub operations."""
        elif agent_type == AgentType.CONFLUENCE:
            description = """📄 DELEGATE TO CONFLUENCE AGENT - Specialized subagent for ALL Confluence documentation operations.

**CRITICAL - USE FOR ANY CONFLUENCE-RELATED TASK:**
This agent handles ALL Confluence operations. Use it whenever the task involves:
- Confluence, pages, spaces, documentation, wiki
- Creating or managing Confluence content
- ANY mention of Confluence in the user's request

**WHAT IS THE CONFLUENCE AGENT:**
- An isolated execution context with Confluence-specific tools (pages, spaces, search)
- Receives ONLY what you provide: task_description + context
- Does NOT inherit conversation history - starts fresh
- Streams work to user, returns summary to you
- Will tell you if it CANNOT complete the task - listen to its feedback

**WHEN TO USE (use liberally for Confluence tasks):**
- ✅ Creating or updating Confluence pages
- ✅ Searching for pages and spaces
- ✅ Fetching page content and structure
- ✅ Adding comments to pages
- ✅ Managing documentation in Confluence
- ✅ ANY task involving Confluence, pages, spaces, or documentation
- ✅ When user mentions "Confluence", "page", "space", "documentation", "wiki"

**AVAILABLE OPERATIONS:**
- Search pages and list spaces
- Get/create/update pages
- Add comments to pages
- List pages in spaces

**CRITICAL - CONTEXT PARAMETER:**
Provide comprehensive context:
- Space keys or names
- Page IDs or titles if known
- Search queries or criteria
- Page content for creation/updates
- Parent page IDs if creating child pages

**NOTE**: The Confluence agent will inform you if it cannot complete a task. If it says it cannot do something, adjust your approach or use a different subagent.

**PARALLELIZATION:** Call multiple times in parallel for independent Confluence operations."""
        elif agent_type == AgentType.LINEAR:
            description = """📋 DELEGATE TO LINEAR AGENT - Specialized subagent for ALL Linear issue management.

**CRITICAL - USE FOR ANY LINEAR-RELATED TASK:**
This agent handles ALL Linear operations. Use it whenever the task involves:
- Linear, Linear issues, Linear tasks
- Creating or managing Linear content
- ANY mention of Linear in the user's request

**WHAT IS THE LINEAR AGENT:**
- An isolated execution context with Linear-specific tools (issue get/update)
- Receives ONLY what you provide: task_description + context
- Does NOT inherit conversation history - starts fresh
- Streams work to user, returns summary to you
- Will tell you if it CANNOT complete the task - listen to its feedback

**WHEN TO USE (use liberally for Linear tasks):**
- ✅ Fetching Linear issue details
- ✅ Updating issue fields (title, description, status, assignee, etc.)
- ✅ Managing Linear issues and their properties
- ✅ ANY task involving Linear or Linear issues
- ✅ When user mentions "Linear", "Linear issue", "Linear task"

**AVAILABLE OPERATIONS:**
- Get issue details by ID
- Update issue fields and properties

**CRITICAL - CONTEXT PARAMETER:**
Provide comprehensive context:
- Issue IDs if known
- Field values for updates (status, assignee, labels, etc.)
- Update descriptions or titles
- Team or project context if needed

**NOTE**: The Linear agent will inform you if it cannot complete a task. If it says it cannot do something, adjust your approach or use a different subagent.

**PARALLELIZATION:** Call multiple times in parallel for independent Linear operations."""
        else:
            description = f"""🤖 DELEGATE TO {agent_type.value.upper()} SUBAGENT - Isolated execution with specialized tools.

**WHAT IS A SUBAGENT:**
- An isolated execution context with specialized tools
- Receives ONLY what you provide: task_description + context
- Does NOT inherit conversation history - starts fresh
- Streams work to user, returns summary to you

**WHY DELEGATE:**
- Context Efficiency: Keeps your context clean
- Token Savings: Heavy work happens in subagent context
- Parallelization: Run multiple subagents simultaneously
- Focus: Dedicated execution for specific tasks

**CRITICAL - CONTEXT PARAMETER:**
Subagents DON'T get your history. Provide comprehensive context:
- Relevant identifiers, keys, or IDs
- Previous findings and analysis results
- Everything needed for autonomous execution

**PARALLELIZATION:** Call multiple times in parallel for independent tasks."""

        return description

    def build_delegation_tools(self) -> List[Tool]:
        """Build delegation tools for all configured subagents.

        These tools allow the supervisor to delegate tasks to specialized
        subagents that can execute work in isolated contexts.

        Returns:
            List of Tool instances, one for each configured subagent type
        """
        delegation_tools = []
        for agent_type in self.delegate_agents.keys():
            description = self._get_delegation_tool_description(agent_type)
            delegation_tools.append(
                Tool(
                    name=f"delegate_to_{agent_type.value}_agent",
                    description=description,
                    function=self.create_delegation_function(agent_type),
                )
            )
        return delegation_tools

    def build_supervisor_agent_tools(self) -> List[Tool]:
        """Build the tool list for supervisor agent including delegation, todo, code changes, and requirement verification tools"""
        # Import tools here to avoid circular imports
        from app.modules.intelligence.tools.todo_management_tool import (
            create_todo_management_tools,
        )
        from app.modules.intelligence.tools.code_changes_manager import (
            create_code_changes_management_tools,
        )
        from app.modules.intelligence.tools.requirement_verification_tool import (
            create_requirement_verification_tools,
        )

        todo_tools = create_todo_management_tools()
        code_changes_tools = create_code_changes_management_tools()
        requirement_tools = create_requirement_verification_tools()

        # Create delegation tools - these are subagent tools that can execute tasks
        delegation_tools = self.build_delegation_tools()

        all_tools = (
            wrap_structured_tools(self.tools)
            + delegation_tools
            + wrap_structured_tools(todo_tools)
            + wrap_structured_tools(code_changes_tools)
            + wrap_structured_tools(requirement_tools)
        )
        # Deduplicate tools by name before returning
        return deduplicate_tools_by_name(all_tools)

    def create_delegate_agent(self, agent_type: AgentType, ctx: ChatContext) -> Agent:
        """Create a delegate agent - either generic (THINK_EXECUTE) or integration-specific"""
        # Cache key includes conversation_id to ensure context-specific agents are not reused
        cache_key = (agent_type, ctx.curr_agent_id)
        if cache_key in self._agent_instances:
            return self._agent_instances[cache_key]

        # Determine if this is an integration agent
        integration_agents = {
            AgentType.JIRA,
            AgentType.GITHUB,
            AgentType.CONFLUENCE,
            AgentType.LINEAR,
        }

        if agent_type in integration_agents:
            # Use integration-specific tools and instructions
            tools = self.build_integration_agent_tools(agent_type)
            instructions = get_integration_agent_instructions(agent_type.value)
        else:
            # Use generic tools and instructions (THINK_EXECUTE)
            tools = self.build_delegate_agent_tools()
            instructions = DELEGATE_AGENT_INSTRUCTIONS

        agent = Agent(
            model=self.llm_provider.get_pydantic_model(),
            tools=tools,
            # NOTE: Subagents don't use MCP servers - they're focused workers that don't need them
            # Instructions are specialized for integration agents, generic for THINK_EXECUTE
            instructions=instructions,
            output_retries=3,
            output_type=str,
            defer_model_check=True,
            end_strategy="early",
            # NOTE: No history_processors for delegate agents - they start fresh with empty
            # history and don't need token management. The history processor's tool pairing
            # logic can also break OpenAI's message format requirements.
            instrument=True,
        )
        self._agent_instances[cache_key] = agent
        return agent

    def create_supervisor_agent(self, ctx: ChatContext, config: AgentConfig) -> Agent:
        """Create the supervisor agent that coordinates other agents"""
        # Cache key includes conversation_id to ensure context-specific instructions are not reused
        conversation_id = ctx.curr_agent_id
        if conversation_id in self._supervisor_agents:
            return self._supervisor_agents[conversation_id]

        # Prepare multimodal instructions if images are present
        multimodal_instructions = prepare_multimodal_instructions(ctx)

        # Get supervisor task description
        supervisor_task_description = create_supervisor_task_description(ctx)

        # Generate supervisor instructions
        instructions = get_supervisor_instructions(
            config_role=config.role,
            config_goal=config.goal,
            task_description=config.tasks[0].description if config.tasks else "",
            multimodal_instructions=multimodal_instructions,
            supervisor_task_description=supervisor_task_description,
        )

        supervisor_agent = Agent(
            model=self.llm_provider.get_pydantic_model(),
            tools=self.build_supervisor_agent_tools(),
            mcp_servers=self.create_mcp_servers(),
            instrument=True,
            instructions=instructions,
            output_retries=3,
            output_type=str,
            defer_model_check=True,
            end_strategy="exhaustive",
            history_processors=[self.history_processor],
        )
        self._supervisor_agents[conversation_id] = supervisor_agent
        return supervisor_agent


def create_integration_agents() -> Dict[AgentType, AgentConfig]:
    """Create integration-specific agents (Jira, GitHub, Confluence, Linear)"""
    return {
        AgentType.JIRA: AgentConfig(
            role="Jira Integration Specialist",
            goal="Handle all Jira operations including issue management, search, and workflow transitions",
            backstory="""You are a specialized agent for Jira operations. You handle issue creation, updates, searches, comments, and status transitions efficiently in an isolated context.""",
            tasks=[
                TaskConfig(
                    description="""Execute Jira operations as requested by the supervisor. Use Jira tools to search, create, update, comment on, and transition issues. Return results in "## Task Result" format with issue keys and links.""",
                    expected_output="Completed Jira operations with issue keys, URLs, and relevant details",
                )
            ],
            max_iter=15,
        ),
        AgentType.GITHUB: AgentConfig(
            role="GitHub Integration Specialist",
            goal="Handle all GitHub repository operations including PRs, branches, and commits",
            backstory="""You are a specialized agent for GitHub operations. You handle pull requests, branches, file updates, and PR comments efficiently in an isolated context.""",
            tasks=[
                TaskConfig(
                    description="""Execute GitHub operations as requested by the supervisor. Use GitHub tools to create branches, PRs, update files, and add comments. Return results in "## Task Result" format with PR numbers, branch names, and GitHub URLs.""",
                    expected_output="Completed GitHub operations with PR numbers, branch names, commit SHAs, and GitHub URLs",
                )
            ],
            max_iter=15,
        ),
        AgentType.CONFLUENCE: AgentConfig(
            role="Confluence Integration Specialist",
            goal="Handle all Confluence documentation operations including pages, spaces, and search",
            backstory="""You are a specialized agent for Confluence operations. You handle page creation, updates, searches, and comments efficiently in an isolated context.""",
            tasks=[
                TaskConfig(
                    description="""Execute Confluence operations as requested by the supervisor. Use Confluence tools to search spaces, get/create/update pages, and add comments. Return results in "## Task Result" format with page IDs, titles, and Confluence URLs.""",
                    expected_output="Completed Confluence operations with space keys, page IDs, titles, and Confluence URLs",
                )
            ],
            max_iter=15,
        ),
        AgentType.LINEAR: AgentConfig(
            role="Linear Integration Specialist",
            goal="Handle Linear issue management operations including fetching and updating issues",
            backstory="""You are a specialized agent for Linear operations. You handle issue fetching and updates efficiently in an isolated context.""",
            tasks=[
                TaskConfig(
                    description="""Execute Linear operations as requested by the supervisor. Use Linear tools to get issue details and update issue fields. Return results in "## Task Result" format with issue IDs, titles, and Linear URLs.""",
                    expected_output="Completed Linear operations with issue IDs, titles, statuses, and Linear URLs",
                )
            ],
            max_iter=15,
        ),
    }


def create_default_delegate_agents() -> Dict[AgentType, AgentConfig]:
    """Create default specialized agents if none provided"""
    integration_agents = create_integration_agents()
    return {
        AgentType.THINK_EXECUTE: AgentConfig(
            role="Task Execution Specialist",
            goal="Execute specific tasks with clear, actionable results",
            backstory="""You are a focused task executor that works in isolated context. Execute specific tasks completely and return only the final result to keep the supervisor's context clean.""",
            tasks=[
                TaskConfig(
                    description="""Execute the specific task assigned by the supervisor. Do all work in your isolated context and return only the final result in "## Task Result" format.""",
                    expected_output="Specific task completion with concrete execution results and deliverables",
                )
            ],
            max_iter=20,
        ),
        **integration_agents,
    }
