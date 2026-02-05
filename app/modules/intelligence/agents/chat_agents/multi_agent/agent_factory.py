"""Agent factory for creating supervisor and delegate agents"""

from typing import List, Dict, Callable, Any
from pydantic_ai import Agent, Tool
from pydantic_ai.mcp import MCPServerStreamableHTTP
from langchain_core.tools import StructuredTool

from .utils.delegation_utils import AgentType
from .utils.tool_utils import (
    wrap_structured_tools,
    deduplicate_tools_by_name,
    sanitize_tool_name_for_api,
)
from .agent_instructions import (
    DELEGATE_AGENT_INSTRUCTIONS,
    SEARCH_FLOW_INSTRUCTIONS,
    get_delegate_agent_instructions,
    get_integration_agent_instructions,
    get_supervisor_instructions,
    prepare_multimodal_instructions,
)
from .utils.context_utils import create_supervisor_task_description
from app.modules.intelligence.agents.chat_agent import ChatContext
from ..agent_config import AgentConfig, TaskConfig
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.utils.logger import setup_logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.modules.intelligence.tools.tool_service import ToolService
    from app.modules.intelligence.tools.registry.resolver import ToolResolver

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
        tools_provider: "ToolService | None" = None,
        tool_resolver: "ToolResolver | None" = None,
    ):
        """Initialize the agent factory

        Args:
            llm_provider: The LLM provider service
            tools: List of tools passed from the agent config
            mcp_servers: Optional MCP servers configuration
            delegate_agents: Delegate agent configurations
            history_processor: History processor for managing conversation history
            create_delegation_function: Function to create delegation functions
            tools_provider: Optional ToolService for integration agents to get their tools directly.
                           If provided, integration agents (GitHub, Jira, etc.) will get their tools
                           from this service instead of filtering from the passed tools list.
            tool_resolver: Optional ToolResolver for registry-driven tool sets (Phase 2).
                          When set, supervisor/delegate/integration tools are built from registry
                          allow-lists (supervisor, execute, integration_*) instead of self.tools.
        """
        self.llm_provider = llm_provider
        self.tools = tools
        self.mcp_servers = mcp_servers or []
        self.delegate_agents = delegate_agents
        self.history_processor = history_processor
        self.create_delegation_function = create_delegation_function
        self.tools_provider = tools_provider
        self.tool_resolver = tool_resolver

        # Sanitize tool names for OpenAI-compatible API: ^[a-zA-Z0-9_-]+$
        for i, tool in enumerate(tools):
            tools[i].name = sanitize_tool_name_for_api(tool.name)

        # Cache for agent instances - keyed by (agent_type, conversation_id, use_tool_search_flow)
        # Cache key: (agent_type, curr_agent_id, local_mode, use_tool_search_flow)
        self._agent_instances: Dict[tuple[AgentType, str, bool, bool], Agent] = {}
        # Cache for supervisor agents - keyed by (conversation_id, local_mode, use_tool_search_flow)
        self._supervisor_agents: Dict[tuple[str, bool, bool], Agent] = {}

    def create_mcp_servers(self) -> List[MCPServerStreamableHTTP]:
        """Create MCP server instances from configuration.

        Each MCP server is given a tool_prefix so its tools do not conflict with
        the agent's built-in tools (e.g. read_todos from the todo toolset).
        """
        mcp_toolsets: List[MCPServerStreamableHTTP] = []
        for i, mcp_server in enumerate(self.mcp_servers):
            try:
                name = mcp_server.get("name") or f"mcp{i}"
                prefix = "".join(
                    c if c.isalnum() or c in "_-" else "_" for c in str(name).lower()
                )
                if not prefix:
                    prefix = f"mcp{i}"
                tool_prefix = f"{prefix}_"
                mcp_server_instance = MCPServerStreamableHTTP(
                    url=mcp_server["link"],
                    timeout=10.0,
                    tool_prefix=tool_prefix,
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

    # Phase 2: AgentType → registry allow-list id for integration agents
    _INTEGRATION_ALLOW_LIST_MAP = {
        AgentType.JIRA: "integration_jira",
        AgentType.GITHUB: "integration_github",
        AgentType.CONFLUENCE: "integration_confluence",
        AgentType.LINEAR: "integration_linear",
    }

    def build_integration_agent_tools(
        self,
        agent_type: AgentType,
        local_mode: bool = False,
        log_tool_annotations: bool = True,
    ) -> List[Tool]:
        """Build tool list for integration-specific agents.

        When tool_resolver is set (Phase 2), integration tool names are resolved from
        registry allow-lists (integration_jira, integration_github, etc.). Otherwise
        uses hardcoded integration_tools_map for backward compatibility.

        GitHub agent also receives code changes tools for committing tracked changes.
        """
        # Import code changes tools here to avoid circular imports
        from app.modules.intelligence.tools.code_changes_manager import (
            create_code_changes_management_tools,
        )

        if self.tool_resolver:
            allow_list_id = self._INTEGRATION_ALLOW_LIST_MAP.get(agent_type)
            if not allow_list_id:
                integration_tools = []
                logger.warning(
                    "AgentFactory: no registry allow-list for agent_type=%s",
                    agent_type.value,
                )
            else:
                integration_tools = self.tool_resolver.get_tools_for_agent(
                    allow_list_id,
                    local_mode=local_mode,
                    exclude_embedding_tools=False,
                    log_tool_annotations=log_tool_annotations,
                )
                logger.info(
                    "AgentFactory: integration tools from registry (allow_list=%s), "
                    "tool_count=%s for agent_type=%s",
                    allow_list_id,
                    len(integration_tools),
                    agent_type.value,
                )
        else:
            # Backward compatibility: hardcoded map
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
                    "code_provider_tool",
                    "github_create_branch",
                    "code_provider_create_branch",
                    "github_create_pull_request",
                    "code_provider_create_pr",
                    "github_add_pr_comments",
                    "code_provider_add_pr_comments",
                    "github_update_branch",
                    "code_provider_update_file",
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
            integration_tool_names = integration_tools_map.get(agent_type, [])
            if self.tools_provider:
                integration_tools = self.tools_provider.get_tools(
                    integration_tool_names
                )
                logger.info(
                    "Got %s tools from tools_provider for %s: %s",
                    len(integration_tools),
                    agent_type.value,
                    [t.name for t in integration_tools],
                )
            else:
                integration_tools = self._filter_tools_by_names(integration_tool_names)
                logger.info(
                    "Filtered %s tools from passed tools for %s: %s",
                    len(integration_tools),
                    agent_type.value,
                    [t.name for t in integration_tools],
                )

        wrapped_tools = wrap_structured_tools(integration_tools)

        if agent_type == AgentType.GITHUB:
            code_changes_tools = create_code_changes_management_tools()
            if local_mode:
                code_changes_tools = [
                    t for t in code_changes_tools if t.name != "show_diff"
                ]
            wrapped_tools = wrapped_tools + wrap_structured_tools(code_changes_tools)
            wrapped_tools = deduplicate_tools_by_name(wrapped_tools)

        return wrapped_tools

    def build_delegate_agent_tools(
        self,
        local_mode: bool = False,
        use_tool_search_flow: bool = False,
        log_tool_annotations: bool = True,
    ) -> List[Tool]:
        """Build the tool list for delegate agents (THINK_EXECUTE).

        When tool_resolver is set (Phase 2), delegate gets full execution set from
        registry allow-list "execute". When use_tool_search_flow=True (Phase 3),
        delegate gets the three discovery meta-tools instead of the full list.

        Subagents get code execution tools and code changes tools, but NOT:
        - Delegation tools (they don't delegate)
        - Todo management tools (supervisor-only for coordination)
        - Requirement verification tools (supervisor-only for verification)
        """
        # Import tools here to avoid circular imports
        from app.modules.intelligence.tools.code_changes_manager import (
            create_code_changes_management_tools,
        )

        code_changes_tools = create_code_changes_management_tools()
        if local_mode:
            code_changes_tools = [
                t for t in code_changes_tools if t.name != "show_diff"
            ]

        if self.tool_resolver:
            if use_tool_search_flow:
                # Phase 3: delegate gets discovery meta-tools instead of full list
                execute_tools = self.tool_resolver.get_discovery_tools_for_agent(
                    "execute",
                    local_mode=local_mode,
                    exclude_embedding_tools=False,
                    log_tool_annotations=log_tool_annotations,
                )
                all_tools = wrap_structured_tools(
                    execute_tools
                ) + wrap_structured_tools(code_changes_tools)
                logger.info(
                    "AgentFactory: delegate (execute) discovery tools from registry, "
                    "discovery_tool_count=3, total_with_code_changes=%s",
                    len(all_tools),
                )
            else:
                # Phase 2: delegate gets full execution set from registry
                execute_tools = self.tool_resolver.get_tools_for_agent(
                    "execute",
                    local_mode=local_mode,
                    exclude_embedding_tools=False,
                    log_tool_annotations=log_tool_annotations,
                )
                all_tools = wrap_structured_tools(
                    execute_tools
                ) + wrap_structured_tools(code_changes_tools)
                logger.info(
                    "AgentFactory: delegate (execute) tools from registry (allow_list=execute), "
                    "execute_tool_count=%s, total_with_code_changes=%s",
                    len(execute_tools),
                    len(all_tools),
                )
        else:
            supervisor_only_tool_names = {
                "read_todos",
                "write_todos",
                "add_todo",
                "update_todo_status",
                "remove_todo",
                "add_subtask",
                "set_dependency",
                "get_available_tasks",
                "add_requirements",
                "delete_requirements",
                "get_requirements",
            }
            filtered_tools = [
                t for t in self.tools if t.name not in supervisor_only_tool_names
            ]
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
- **Committing code changes** tracked in the conversation to GitHub
- **ANY mention of GitHub, repository, issues, PRs, branches, or commits in the user's request**

**WHAT IS THE GITHUB AGENT:**
- An isolated execution context with **GitHub-specific tools** (github_tool, github_create_branch, github_create_pull_request, etc.)
- **HAS ACCESS to code changes tools** - can read and use tracked code changes from the conversation
- Receives ONLY what you provide: task_description + context
- Does NOT inherit conversation history - starts fresh
- Streams work to user, returns summary to you
- Will tell you if it CANNOT complete the task - listen to its feedback

**WHEN TO USE (use liberally for GitHub tasks):**
- ✅ **Fetching GitHub issues** (all open issues, specific issues, issue details) - **PRIMARY USE CASE**
- ✅ **Fetching pull requests** (PRs, PR details, PR diffs)
- ✅ Creating pull requests and branches
- ✅ Updating files in branches
- ✅ **Committing tracked code changes** to a branch or PR
- ✅ Adding PR review comments with code references
- ✅ Managing repository operations
- ✅ **ANY task involving GitHub, repository, issues, PRs, branches, or commits**
- ✅ When user asks to "list issues", "fetch issues", "get issues", "show PRs", "create PR", "commit changes", etc.

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
- **Access tracked code changes** (list_files_in_changes, get_file_from_changes, export_changes, show_diff, etc.)

**CRITICAL - CONTEXT PARAMETER:**
Provide comprehensive context:
- **Repository name** (e.g., "owner/repo" or "nndn/coin_game") - **REQUIRED for issue/PR fetching**
- Branch names (source and target)
- PR numbers or issue numbers if known
- File paths and content for updates
- Commit messages and descriptions
- **For committing changes**: Include project_id and branch name

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
                    name=sanitize_tool_name_for_api(
                        f"delegate_to_{agent_type.value}_agent"
                    ),
                    description=description,
                    function=self.create_delegation_function(agent_type),
                )
            )
        return delegation_tools

    def build_supervisor_agent_tools(
        self,
        local_mode: bool = False,
        use_tool_search_flow: bool = False,
        log_tool_annotations: bool = True,
    ) -> List[Tool]:
        """Build the tool list for supervisor agent including delegation, todo, code changes, and requirement verification tools.

        When tool_resolver is set (Phase 2), supervisor gets a curated set from registry
        allow-list "supervisor" (includes todo/requirement tools). When use_tool_search_flow=True (Phase 3),
        supervisor gets the three discovery meta-tools instead of the full tool list.

        Note: Todo/requirement tools are provided via the registry (SUPERVISOR_TOOLS), not via a
        separate toolset, to avoid name conflicts with MCP servers that may also expose read_todos.

        Note: Tool filtering for local_mode is handled in code_gen_agent.py.
        Here we only filter out show_diff in local mode (VSCode extension handles diff display).
        """
        # Import tools here to avoid circular imports
        from app.modules.intelligence.tools.code_changes_manager import (
            create_code_changes_management_tools,
        )
        from app.modules.intelligence.tools.requirement_verification_tool import (
            create_requirement_verification_tools,
        )

        # Create code changes tools (always create, filter show_diff in local mode)
        code_changes_tools = create_code_changes_management_tools()
        if local_mode:
            # In local mode, filter out show_diff (VSCode extension handles diff display)
            code_changes_tools = [
                t for t in code_changes_tools if t.name != "show_diff"
            ]
        requirement_tools = create_requirement_verification_tools()
        delegation_tools = self.build_delegation_tools()

        if self.tool_resolver:
            if use_tool_search_flow:
                # Phase 3: supervisor gets discovery meta-tools instead of full list
                supervisor_tools = self.tool_resolver.get_discovery_tools_for_agent(
                    "supervisor",
                    local_mode=local_mode,
                    exclude_embedding_tools=False,
                    log_tool_annotations=log_tool_annotations,
                )
                all_tools = (
                    wrap_structured_tools(supervisor_tools)
                    + delegation_tools
                    + wrap_structured_tools(code_changes_tools)
                    + wrap_structured_tools(requirement_tools)
                )
                logger.info(
                    "AgentFactory: supervisor discovery tools from registry, "
                    "discovery_tool_count=3, total_with_coordination=%s",
                    len(all_tools),
                )
            else:
                # Phase 2: supervisor gets curated set from registry (no terminal, no full execution suite)
                supervisor_tools = self.tool_resolver.get_tools_for_agent(
                    "supervisor",
                    local_mode=local_mode,
                    exclude_embedding_tools=False,
                    log_tool_annotations=log_tool_annotations,
                )
                all_tools = (
                    wrap_structured_tools(supervisor_tools)
                    + delegation_tools
                    + wrap_structured_tools(code_changes_tools)
                    + wrap_structured_tools(requirement_tools)
                )
                logger.info(
                    "AgentFactory: supervisor tools from registry (allow_list=supervisor), "
                    "supervisor_tool_count=%s, total_with_coordination=%s",
                    len(supervisor_tools),
                    len(all_tools),
                )
        else:
            logger.debug(
                "build_supervisor_agent_tools: local_mode=%s, initial_tools_count=%s (no tool_resolver)",
                local_mode,
                len(self.tools),
            )
            all_tools = (
                wrap_structured_tools(self.tools)
                + delegation_tools
                + wrap_structured_tools(code_changes_tools)
                + wrap_structured_tools(requirement_tools)
            )

        final_tools = deduplicate_tools_by_name(all_tools)
        logger.debug(
            "build_supervisor_agent_tools: local_mode=%s, final_tools_count=%s",
            local_mode,
            len(final_tools),
        )
        return final_tools

    def create_delegate_agent(self, agent_type: AgentType, ctx: ChatContext) -> Agent:
        """Create a delegate agent - either generic (THINK_EXECUTE) or integration-specific"""
        # Cache key includes curr_agent_id, local_mode, and use_tool_search_flow (Phase 3)
        # local_mode must be in key so we don't reuse a delegate built for the other mode (tools differ)
        use_tool_search_flow = getattr(ctx, "use_tool_search_flow", False)
        local_mode = ctx.local_mode if hasattr(ctx, "local_mode") else False
        cache_key = (agent_type, ctx.curr_agent_id, local_mode, use_tool_search_flow)
        if cache_key in self._agent_instances:
            return self._agent_instances[cache_key]

        # Determine if this is an integration agent
        integration_agents = {
            AgentType.JIRA,
            AgentType.GITHUB,
            AgentType.CONFLUENCE,
            AgentType.LINEAR,
        }

        log_tool_annotations = getattr(ctx, "log_tool_annotations", True)
        if agent_type in integration_agents:
            # Use integration-specific tools and instructions (no discovery flow)
            tools = self.build_integration_agent_tools(
                agent_type,
                local_mode=local_mode,
                log_tool_annotations=log_tool_annotations,
            )
            instructions = get_integration_agent_instructions(agent_type.value)
        else:
            # Use generic tools and instructions (THINK_EXECUTE); Phase 3 discovery flow when enabled
            tools = self.build_delegate_agent_tools(
                local_mode=local_mode,
                use_tool_search_flow=use_tool_search_flow,
                log_tool_annotations=log_tool_annotations,
            )
            instructions = get_delegate_agent_instructions(local_mode=local_mode)
            if use_tool_search_flow:
                instructions = instructions + SEARCH_FLOW_INSTRUCTIONS

        agent = Agent(
            model=self.llm_provider.get_pydantic_model(),
            tools=tools,
            # NOTE: Subagents don't use MCP servers - they're focused workers that don't need them
            # Instructions are specialized for integration agents, generic for THINK_EXECUTE
            instructions=instructions,
            output_retries=3,
            output_type=str,
            defer_model_check=True,
            end_strategy="exhaustive",
            # NOTE: No history_processors for delegate agents - they start fresh with empty
            # history and don't need token management. The history processor's tool pairing
            # logic can also break OpenAI's message format requirements.
            instrument=True,
        )
        self._agent_instances[cache_key] = agent
        return agent

    def create_supervisor_agent(self, ctx: ChatContext, config: AgentConfig) -> Agent:
        """Create the supervisor agent that coordinates other agents"""
        # Cache key includes conversation_id, local_mode, and use_tool_search_flow (Phase 3)
        conversation_id = ctx.curr_agent_id
        local_mode = ctx.local_mode if hasattr(ctx, "local_mode") else False
        use_tool_search_flow = getattr(ctx, "use_tool_search_flow", False)
        cache_key = (conversation_id, local_mode, use_tool_search_flow)
        if cache_key in self._supervisor_agents:
            logger.debug(
                "Returning cached supervisor agent for conversation_id=%s, local_mode=%s, use_tool_search_flow=%s",
                conversation_id,
                local_mode,
                use_tool_search_flow,
            )
            return self._supervisor_agents[cache_key]

        # Prepare multimodal instructions if images are present
        multimodal_instructions = prepare_multimodal_instructions(ctx)

        # Get supervisor task description
        supervisor_task_description = create_supervisor_task_description(ctx)

        # Generate supervisor instructions (task_description is code_gen_task_prompt or code_gen_task_prompt_local when code gen agent)
        task_description = config.tasks[0].description if config.tasks else ""
        if task_description and "local mode" in task_description.lower():
            logger.info(
                "create_supervisor_agent: using LOCAL task prompt (code_gen_task_prompt_local) [local_mode=%s]",
                local_mode,
            )
        instructions = get_supervisor_instructions(
            config_role=config.role,
            config_goal=config.goal,
            task_description=task_description,
            multimodal_instructions=multimodal_instructions,
            supervisor_task_description=supervisor_task_description,
            local_mode=local_mode,
        )
        if use_tool_search_flow:
            instructions = instructions + SEARCH_FLOW_INSTRUCTIONS

        # Build tools (local_mode affects show_diff; use_tool_search_flow selects discovery vs full list)
        log_tool_annotations = getattr(ctx, "log_tool_annotations", True)
        tools = self.build_supervisor_agent_tools(
            local_mode=local_mode,
            use_tool_search_flow=use_tool_search_flow,
            log_tool_annotations=log_tool_annotations,
        )
        logger.info(
            f"Creating supervisor agent: conversation_id={conversation_id}, local_mode={local_mode}, "
            f"tools_count={len(tools)}, tool_names={[t.name for t in tools[:10]]}..."
        )

        supervisor_agent = Agent(
            model=self.llm_provider.get_pydantic_model(),
            tools=tools,
            mcp_servers=self.create_mcp_servers(),
            instrument=True,
            instructions=instructions,
            output_retries=3,
            output_type=str,
            defer_model_check=True,
            end_strategy="exhaustive",
            history_processors=[self.history_processor],
        )
        self._supervisor_agents[cache_key] = supervisor_agent
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
