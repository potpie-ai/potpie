"""Static registry definitions: tool metadata (tier, category) and allow-lists."""

from typing import Dict, FrozenSet, List

from app.modules.intelligence.tools.registry.schema import AllowListDefinition

# Tools that ToolService may omit when optional deps/config are missing
# (e.g. API keys). The registry keeps them for allow-lists; do not warn
# "registry has names not in ToolService" for these.
OPTIONAL_TOOL_NAMES: FrozenSet[str] = frozenset(
    {
        "webpage_extractor",
    }
)


# --- Tool metadata definitions (tier, category). Description filled at population from ToolService. ---

# Optional per-tool: short_description (≤100 chars for discovery), defer_loading (Phase 3)
TOOL_DEFINITIONS: Dict[str, dict] = {
    # Search (Phase 4: read_only, idempotent)
    "search_text": {
        "tier": "medium",
        "category": "search",
        "short_description": "Search for text in the codebase.",
        "read_only": True,
        "idempotent": True,
    },
    "search_files": {"tier": "medium", "category": "search"},
    "search_symbols": {"tier": "medium", "category": "search"},
    "search_workspace_symbols": {"tier": "medium", "category": "search"},
    "search_references": {"tier": "medium", "category": "search"},
    "search_definitions": {"tier": "medium", "category": "search"},
    "search_code_structure": {"tier": "medium", "category": "search"},
    "search_bash": {"tier": "low", "category": "search"},
    "semantic_search": {"tier": "medium", "category": "search"},
    # Terminal (local_mode_only: only sent when local_mode=True, i.e. VS Code extension)
    "execute_terminal_command": {
        "tier": "low",
        "category": "terminal",
        "short_description": "Execute a terminal command.",
        "destructive": True,
        "requires_confirmation": True,
        "local_mode_only": True,
    },
    "terminal_session_output": {
        "tier": "low",
        "category": "terminal",
        "local_mode_only": True,
    },
    "terminal_session_signal": {
        "tier": "low",
        "category": "terminal",
        "local_mode_only": True,
    },
    # --- Legacy CodeChangesManager staging family + repo_manager-gated git
    # tools removed during the sandbox migration. Agents now edit the
    # worktree via sandbox_text_editor and commit via sandbox_git. ---
    # Todo (pydantic-ai-todo)
    "read_todos": {"tier": "high", "category": "todo"},
    "write_todos": {"tier": "high", "category": "todo"},
    "add_todo": {"tier": "high", "category": "todo"},
    "update_todo_status": {"tier": "high", "category": "todo"},
    "remove_todo": {"tier": "high", "category": "todo"},
    "add_subtask": {"tier": "high", "category": "todo"},
    "set_dependency": {"tier": "high", "category": "todo"},
    "get_available_tasks": {"tier": "high", "category": "todo"},
    "add_requirements": {"tier": "high", "category": "requirement"},
    "get_requirements": {"tier": "high", "category": "requirement"},
    "delete_requirements": {"tier": "high", "category": "requirement"},
    # Web
    "webpage_extractor": {"tier": "high", "category": "web"},
    "web_search_tool": {
        "tier": "high",
        "category": "web",
        "short_description": "Search the web for information.",
    },
    # Knowledge graph
    "ask_knowledge_graph_queries": {"tier": "high", "category": "knowledge_graph"},
    "get_nodes_from_tags": {"tier": "high", "category": "knowledge_graph"},
    "get_code_from_probable_node_name": {"tier": "high", "category": "knowledge_graph"},
    "get_code_from_node_id": {"tier": "high", "category": "knowledge_graph"},
    "get_code_from_multiple_node_ids": {"tier": "high", "category": "knowledge_graph"},
    "get_code_graph_from_node_id": {
        "tier": "high",
        "category": "knowledge_graph",
        "defer_loading": True,
    },
    "get_node_neighbours_from_node_id": {"tier": "high", "category": "knowledge_graph"},
    # File fetch / analysis
    "fetch_file": {"tier": "medium", "category": "file_fetch"},
    "fetch_files_batch": {"tier": "medium", "category": "file_fetch"},
    "get_code_file_structure": {"tier": "medium", "category": "file_fetch"},
    "analyze_code_structure": {"tier": "medium", "category": "analysis"},
    "intelligent_code_graph": {"tier": "high", "category": "analysis"},
    "change_detection": {"tier": "medium", "category": "analysis"},
    # Context graph minimal agent port
    "context_resolve": {
        "tier": "medium",
        "category": "search",
        "read_only": True,
        "idempotent": True,
        "short_description": "Resolve a bounded task context wrap.",
    },
    "context_search": {
        "tier": "medium",
        "category": "search",
        "read_only": True,
        "idempotent": True,
        "short_description": "Search project memory for a known follow-up.",
    },
    "context_status": {
        "tier": "medium",
        "category": "search",
        "read_only": True,
        "idempotent": True,
        "short_description": "Check pot readiness and recommended recipe.",
    },
    "context_record": {
        "tier": "medium",
        "category": "knowledge_graph",
        "short_description": "Record durable project memory.",
    },
    # Integrations
    "get_linear_issue": {"tier": "high", "category": "integration_linear"},
    "update_linear_issue": {"tier": "high", "category": "integration_linear"},
    "get_jira_issue": {
        "tier": "high",
        "category": "integration_jira",
        "short_description": "Get a Jira issue by key.",
        "defer_loading": True,
        "read_only": True,
        "idempotent": True,
    },
    "search_jira_issues": {"tier": "high", "category": "integration_jira"},
    "create_jira_issue": {"tier": "high", "category": "integration_jira"},
    "update_jira_issue": {"tier": "high", "category": "integration_jira"},
    "add_jira_comment": {"tier": "high", "category": "integration_jira"},
    "transition_jira_issue": {"tier": "high", "category": "integration_jira"},
    "get_jira_projects": {"tier": "high", "category": "integration_jira"},
    "get_jira_project_details": {"tier": "high", "category": "integration_jira"},
    "get_jira_project_users": {"tier": "high", "category": "integration_jira"},
    "link_jira_issues": {"tier": "high", "category": "integration_jira"},
    "get_confluence_spaces": {"tier": "high", "category": "integration_confluence"},
    "get_confluence_page": {"tier": "high", "category": "integration_confluence"},
    "search_confluence_pages": {"tier": "high", "category": "integration_confluence"},
    "get_confluence_space_pages": {
        "tier": "high",
        "category": "integration_confluence",
    },
    "create_confluence_page": {"tier": "high", "category": "integration_confluence"},
    "update_confluence_page": {"tier": "high", "category": "integration_confluence"},
    "add_confluence_comment": {"tier": "high", "category": "integration_confluence"},
    "code_provider_tool": {
        "tier": "high",
        "category": "integration_github",
        "aliases": ["github_tool"],
    },
    "code_provider_create_branch": {
        "tier": "high",
        "category": "integration_github",
        "aliases": ["github_create_branch"],
    },
    "code_provider_create_pr": {
        "tier": "high",
        "category": "integration_github",
        "aliases": ["github_create_pull_request"],
        "requires_confirmation": True,
    },
    "code_provider_add_pr_comments": {
        "tier": "high",
        "category": "integration_github",
        "aliases": ["github_add_pr_comments"],
    },
    "code_provider_update_file": {
        "tier": "high",
        "category": "integration_github",
        "aliases": ["github_update_branch"],
    },
    # Sandbox group — Anthropic-style consolidated tools over app/src/sandbox.
    "sandbox_text_editor": {
        "tier": "medium",
        "category": "sandbox",
        "short_description": "View / create / str_replace / insert on worktree files.",
    },
    "sandbox_shell": {
        "tier": "low",
        "category": "sandbox",
        "short_description": "Run a single shell command inside the sandbox.",
        "destructive": True,
    },
    "sandbox_search": {
        "tier": "medium",
        "category": "sandbox",
        "short_description": "Ripgrep across the worktree.",
        "read_only": True,
        "idempotent": True,
    },
    "sandbox_git": {
        "tier": "medium",
        "category": "sandbox",
        "short_description": "Git status / diff / log / commit / push on the worktree.",
    },
}

# --- Allow-lists ---

CODE_GEN_BASE_TOOLS: List[str] = [
    # Knowledge / discovery
    "webpage_extractor",
    "web_search_tool",
    "search_text",
    "search_files",
    "search_symbols",
    "search_workspace_symbols",
    "search_references",
    "search_definitions",
    "search_code_structure",
    "search_bash",
    "semantic_search",
    "ask_knowledge_graph_queries",
    # Local-mode terminal (filtered out in non-local via exclude_in_local)
    "execute_terminal_command",
    "terminal_session_output",
    "terminal_session_signal",
    # Run state
    "read_todos",
    "write_todos",
    "add_todo",
    "update_todo_status",
    "remove_todo",
    "add_subtask",
    "set_dependency",
    "get_available_tasks",
    "add_requirements",
    "get_requirements",
    "delete_requirements",
    # Sandbox tools (Anthropic-style consolidated surface).
    "sandbox_text_editor",
    "sandbox_shell",
    "sandbox_search",
    "sandbox_git",
]

CODE_GEN_ADD_WHEN_NON_LOCAL: List[str] = [
    # Context-graph and KG-backed readers — secondary to sandbox_text_editor
    # for files the agent might edit, but useful for graph-shaped queries
    # (impacts, neighbours, semantic / cross-cut search).
    "context_status",
    "context_resolve",
    "context_search",
    "context_record",
    "get_code_from_multiple_node_ids",
    "get_node_neighbours_from_node_id",
    "get_code_from_probable_node_name",
    "get_nodes_from_tags",
    "analyze_code_structure",
    # PR creation lives outside the sandbox (auth chain in code_provider).
    "code_provider_create_branch",
    "code_provider_create_pr",
]
CODE_GEN_EXCLUDE_IN_LOCAL: List[str] = []

GENERAL_PURPOSE_TOOLS: List[str] = [
    "webpage_extractor",
    "web_search_tool",
]

# --- Phase 2: Scoped tool sets for supervisor and subagents ---

# Supervisor: coordination + light discovery + todo/requirement (no terminal, no full code edit suite).
# Todo/requirement tools are included here so we have a single source; do not also pass
# create_todo_management_toolset() as a toolset or we get "read_todos" name conflicts with MCP.
SUPERVISOR_TOOLS: List[str] = [
    "context_status",
    "context_resolve",
    "context_search",
    "context_record",
    "fetch_file",
    "get_code_file_structure",
    "web_search_tool",
    "webpage_extractor",
    "read_todos",
    "write_todos",
    "add_todo",
    "update_todo_status",
    "remove_todo",
    "add_subtask",
    "set_dependency",
    "get_available_tasks",
    "add_requirements",
    "get_requirements",
    "delete_requirements",
]

# Execute: same shape as code_gen, minus the agent-specific narrative pieces.
# Sandbox tools replace the code_changes_manager staging family.
EXECUTE_TOOLS: List[str] = [
    "webpage_extractor",
    "web_search_tool",
    "context_status",
    "context_resolve",
    "context_search",
    "context_record",
    "ask_knowledge_graph_queries",
    "execute_terminal_command",
    "terminal_session_output",
    "terminal_session_signal",
    # Sandbox: read / edit / shell / git on the agent worktree.
    "sandbox_text_editor",
    "sandbox_shell",
    "sandbox_search",
    "sandbox_git",
    # PR creation — outside the sandbox, uses code_provider's auth chain.
    "code_provider_create_branch",
    "code_provider_create_pr",
]
EXECUTE_ADD_WHEN_NON_LOCAL: List[str] = [
    "get_code_from_multiple_node_ids",
    "get_node_neighbours_from_node_id",
    "get_code_from_probable_node_name",
    "get_nodes_from_tags",
    "analyze_code_structure",
]
EXECUTE_EXCLUDE_IN_LOCAL: List[str] = []

# Supervisor delegates writes to subagents — give it sandbox_search + sandbox_git
# (status / diff / log) for inspection only. No editor / shell at the supervisor level.
SUPERVISOR_ADD_WHEN_NON_LOCAL: List[str] = [
    "sandbox_search",
    "sandbox_git",
]

# Explore: minimal read-only set — sandbox_text_editor (view) + sandbox_search.
EXPLORE_TOOLS: List[str] = [
    "sandbox_text_editor",
    "sandbox_search",
]

# Integration: primary names only (registry resolves aliases)
INTEGRATION_JIRA_TOOLS: List[str] = [
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
]
INTEGRATION_GITHUB_TOOLS: List[str] = [
    "code_provider_tool",
    "code_provider_create_branch",
    "code_provider_create_pr",
    "code_provider_add_pr_comments",
    "code_provider_update_file",
    # Sandbox git surface (replaces apply_changes + git_commit + git_push +
    # create_pr_workflow — push from sandbox_git, then code_provider_create_pr).
    "sandbox_git",
]
INTEGRATION_CONFLUENCE_TOOLS: List[str] = [
    "get_confluence_spaces",
    "get_confluence_page",
    "search_confluence_pages",
    "get_confluence_space_pages",
    "create_confluence_page",
    "update_confluence_page",
    "add_confluence_comment",
]
INTEGRATION_LINEAR_TOOLS: List[str] = [
    "get_linear_issue",
    "update_linear_issue",
]

# The four consolidated sandbox tools. Agents that want full sandbox access
# include the "sandbox" allow-list group, or reference SANDBOX_TOOLS directly.
SANDBOX_TOOLS: List[str] = [
    "sandbox_text_editor",
    "sandbox_shell",
    "sandbox_search",
    "sandbox_git",
]

ALLOW_LIST_DEFINITIONS: List[AllowListDefinition] = [
    AllowListDefinition(
        name="code_gen",
        tool_names=CODE_GEN_BASE_TOOLS,
        add_when_non_local=CODE_GEN_ADD_WHEN_NON_LOCAL,
        exclude_in_local=CODE_GEN_EXCLUDE_IN_LOCAL,
        tier_filter=None,
    ),
    AllowListDefinition(
        name="general_purpose",
        tool_names=GENERAL_PURPOSE_TOOLS,
        tier_filter=None,
    ),
    # Phase 2: scoped sets for multi-agent
    AllowListDefinition(
        name="supervisor",
        tool_names=SUPERVISOR_TOOLS,
        add_when_non_local=SUPERVISOR_ADD_WHEN_NON_LOCAL,
        tier_filter=None,
    ),
    AllowListDefinition(
        name="execute",
        tool_names=EXECUTE_TOOLS,
        add_when_non_local=EXECUTE_ADD_WHEN_NON_LOCAL,
        exclude_in_local=EXECUTE_EXCLUDE_IN_LOCAL,
        tier_filter=None,
    ),
    AllowListDefinition(name="explore", tool_names=EXPLORE_TOOLS, tier_filter=None),
    AllowListDefinition(
        name="integration_jira",
        tool_names=INTEGRATION_JIRA_TOOLS,
        tier_filter=None,
    ),
    AllowListDefinition(
        name="integration_github",
        tool_names=INTEGRATION_GITHUB_TOOLS,
        tier_filter=None,
    ),
    AllowListDefinition(
        name="integration_confluence",
        tool_names=INTEGRATION_CONFLUENCE_TOOLS,
        tier_filter=None,
    ),
    AllowListDefinition(
        name="integration_linear",
        tool_names=INTEGRATION_LINEAR_TOOLS,
        tier_filter=None,
    ),
    AllowListDefinition(
        name="sandbox",
        tool_names=SANDBOX_TOOLS,
        tier_filter=None,
    ),
]
