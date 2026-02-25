"""Static registry definitions: tool metadata (tier, category) and allow-lists."""

from typing import Dict, FrozenSet, List

from app.modules.intelligence.tools.registry.schema import AllowListDefinition

# Tools that ToolService may omit when optional deps/config are missing (e.g. API keys).
# Registry keeps them for allow-lists; do not warn "registry has names not in ToolService".
# Also includes tools gated on REPO_MANAGER_ENABLED (bash_command, apply_changes,
# git_commit, git_push) which are absent from ToolService when the flag is off.
OPTIONAL_TOOL_NAMES: FrozenSet[str] = frozenset(
    {
        "webpage_extractor",
        "bash_command",
        "apply_changes",
        "git_commit",
        "git_push",
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
    "bash_command": {"tier": "low", "category": "terminal"},
    # Code changes
    "add_file_to_changes": {
        "tier": "medium",
        "category": "code_changes",
        "short_description": "Add a file to the tracked code changes.",
    },
    "update_file_in_changes": {
        "tier": "medium",
        "category": "code_changes",
        "destructive": True,
    },
    "update_file_lines": {"tier": "medium", "category": "code_changes"},
    "replace_in_file": {
        "tier": "medium",
        "category": "code_changes",
        "destructive": True,
    },
    "insert_lines": {"tier": "medium", "category": "code_changes", "destructive": True},
    "delete_lines": {
        "tier": "medium",
        "category": "code_changes",
        "destructive": True,
    },
    "delete_file_in_changes": {
        "tier": "medium",
        "category": "code_changes",
        "destructive": True,
    },
    "get_file_from_changes": {"tier": "medium", "category": "code_changes"},
    "list_files_in_changes": {"tier": "medium", "category": "code_changes"},
    "search_content_in_changes": {"tier": "medium", "category": "code_changes"},
    "clear_file_from_changes": {
        "tier": "medium",
        "category": "code_changes",
        "destructive": True,
    },
    "clear_all_changes": {
        "tier": "medium",
        "category": "code_changes",
        "destructive": True,
    },
    "get_changes_summary": {"tier": "medium", "category": "code_changes"},
    "get_changes_for_pr": {"tier": "medium", "category": "code_changes"},
    "export_changes": {"tier": "medium", "category": "code_changes"},
    "show_updated_file": {"tier": "medium", "category": "code_changes"},
    "get_file_diff": {"tier": "medium", "category": "code_changes"},
    "show_diff": {"tier": "medium", "category": "code_changes"},
    "revert_file": {
        "tier": "medium",
        "category": "code_changes",
        "destructive": True,
    },
    "get_session_metadata": {"tier": "medium", "category": "code_changes"},
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
    # Git workflow tools (apply changes from Redis to worktree)
    "apply_changes": {
        "tier": "medium",
        "category": "code_changes",
        "short_description": "Apply changes from CodeChangesManager to the worktree filesystem.",
        "destructive": True,
    },
    "git_commit": {
        "tier": "medium",
        "category": "code_changes",
        "short_description": "Stage and commit changes in the repository worktree.",
        "destructive": True,
    },
    "git_push": {
        "tier": "medium",
        "category": "code_changes",
        "short_description": "Push the current branch to the remote repository.",
        "destructive": True,
    },
    # Composite PR workflow tool (combines apply_changes + git_commit + git_push + create_pr)
    "create_pr_workflow": {
        "tier": "medium",
        "category": "code_changes",
        "short_description": "Composite tool: apply changes, commit, push, and create PR in one operation.",
        "destructive": True,
    },
}

# --- Allow-lists ---

CODE_GEN_BASE_TOOLS: List[str] = [
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
    "execute_terminal_command",
    "terminal_session_output",
    "terminal_session_signal",
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
    "add_file_to_changes",
    "update_file_in_changes",
    "update_file_lines",
    "replace_in_file",
    "insert_lines",
    "delete_lines",
    "delete_file_in_changes",
    "get_file_from_changes",
    "list_files_in_changes",
    "search_content_in_changes",
    "clear_file_from_changes",
    "clear_all_changes",
    "get_changes_summary",
    "export_changes",
    "show_updated_file",
    "get_file_diff",
    "revert_file",
    "get_session_metadata",
    # Git workflow tools for PR creation
    "apply_changes",
    "git_commit",
    "git_push",
    # Composite PR workflow tool (single tool for apply + commit + push + create_pr)
    "create_pr_workflow",
]

CODE_GEN_ADD_WHEN_NON_LOCAL: List[str] = [
    "get_code_from_multiple_node_ids",
    "get_node_neighbours_from_node_id",
    "get_code_from_probable_node_name",
    "get_nodes_from_tags",
    "get_code_file_structure",
    "fetch_file",
    "fetch_files_batch",
    "analyze_code_structure",
    "show_diff",
    "bash_command",
]
CODE_GEN_EXCLUDE_IN_LOCAL: List[str] = ["show_diff"]

GENERAL_PURPOSE_TOOLS: List[str] = [
    "webpage_extractor",
    "web_search_tool",
]

# --- Phase 2: Scoped tool sets for supervisor and subagents ---

# Supervisor: coordination + light discovery + todo/requirement (no terminal, no full code edit suite).
# Todo/requirement tools are included here so we have a single source; do not also pass
# create_todo_management_toolset() as a toolset or we get "read_todos" name conflicts with MCP.
SUPERVISOR_TOOLS: List[str] = [
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

# Execute: CODE_GEN_BASE_TOOLS minus todo/requirement (same as current delegate)
EXECUTE_TOOLS: List[str] = [
    "webpage_extractor",
    "web_search_tool",
    "ask_knowledge_graph_queries",
    "execute_terminal_command",
    "terminal_session_output",
    "terminal_session_signal",
    "add_file_to_changes",
    "update_file_in_changes",
    "update_file_lines",
    "replace_in_file",
    "insert_lines",
    "delete_lines",
    "delete_file_in_changes",
    "get_file_from_changes",
    "list_files_in_changes",
    "search_content_in_changes",
    "clear_file_from_changes",
    "clear_all_changes",
    "get_changes_summary",
    "get_changes_for_pr",
    "export_changes",
    "show_updated_file",
    "get_file_diff",
    "revert_file",
    "get_session_metadata",
    # Git workflow tools for PR creation
    "apply_changes",
    "git_commit",
    "git_push",
    "code_provider_create_branch",
    "code_provider_create_pr",
    # Composite PR workflow (apply + commit + push + create_pr in one call)
    "create_pr_workflow",
]
EXECUTE_ADD_WHEN_NON_LOCAL: List[str] = [
    "get_code_from_multiple_node_ids",
    "get_node_neighbours_from_node_id",
    "get_code_from_probable_node_name",
    "get_nodes_from_tags",
    "get_code_file_structure",
    "fetch_file",
    "fetch_files_batch",
    "analyze_code_structure",
    "show_diff",
    "bash_command",
]
EXECUTE_EXCLUDE_IN_LOCAL: List[str] = ["show_diff"]

# Supervisor non-local additions: repo-manager-backed tools not needed in local/VSCode mode
SUPERVISOR_ADD_WHEN_NON_LOCAL: List[str] = [
    "bash_command",
]

# Explore: minimal read-only set (for future use)
EXPLORE_TOOLS: List[str] = [
    "get_code_file_structure",
    "fetch_file",
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
    # Get changes summary for PR (verify before create_pr_workflow)
    "get_changes_for_pr",
    # Git workflow tools for PR creation from worktree
    "apply_changes",
    "git_commit",
    "git_push",
    # Composite PR workflow (apply + commit + push + create_pr in one call)
    "create_pr_workflow",
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
]
