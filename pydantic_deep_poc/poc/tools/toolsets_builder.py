"""Build FunctionToolsets scoped per Potpie role (supervisor, execute, integrations)."""

from __future__ import annotations

from pydantic_ai.toolsets import FunctionToolset

from poc.managers.deps import PoCDeepDeps
from poc.tools import (
    bash,
    ccm_tools,
    file_ops,
    git_tools,
    integrations,
    requirements_tools,
    todo_tools,
    web,
)


def _fs(id_: str) -> FunctionToolset[PoCDeepDeps]:
    return FunctionToolset(id=id_)


def supervisor_toolset() -> FunctionToolset[PoCDeepDeps]:
    """Supervisor toolset - orchestration only, no shell, no direct edits."""
    fs = _fs("poc-supervisor-tools")
    for fn in (
        # Read-only file operations
        file_ops.fetch_file,
        file_ops.fetch_files_batch,
        file_ops.get_code_file_structure,
        file_ops.analyze_code_structure,
        # Web/docs for context
        web.web_search_tool,
        web.webpage_extractor,
        # Todo/requirements management
        todo_tools.read_todos,
        todo_tools.write_todos,
        todo_tools.add_todo,
        todo_tools.update_todo_status,
        todo_tools.remove_todo,
        todo_tools.add_subtask,
        todo_tools.set_dependency,
        todo_tools.get_available_tasks,
        requirements_tools.add_requirements,
        requirements_tools.get_requirements,
        requirements_tools.delete_requirements,
        # NO bash_command - supervisor should not run shell commands
        # NO checkout_worktree_branch - only harness creates worktrees
        # Git operations (read-only and apply)
        git_tools.show_diff,
        git_tools.get_verification_status,
        git_tools.apply_changes,
        # CCM read-only operations
        ccm_tools.get_changes_summary,
        ccm_tools.list_files_in_changes,
        ccm_tools.get_changes_for_pr,
    ):
        fs.add_function(fn)
    return fs


def discovery_toolset() -> FunctionToolset[PoCDeepDeps]:
    """Discovery toolset - read-only exploration with policy-scoped shell."""
    fs = _fs("poc-discovery-tools")
    for fn in (
        web.webpage_extractor,
        web.web_search_tool,
        file_ops.fetch_file,
        file_ops.fetch_files_batch,
        file_ops.get_code_file_structure,
        file_ops.analyze_code_structure,
        # Policy-scoped read-only bash (replaces unrestricted bash_command)
        bash.read_only_bash,
    ):
        fs.add_function(fn)
    return fs


def implementation_toolset() -> FunctionToolset[PoCDeepDeps]:
    """Implementation toolset - CCM writes with validation-only shell."""
    fs = _fs("poc-implementation-tools")
    for fn in (
        file_ops.fetch_file,
        file_ops.fetch_files_batch,
        file_ops.analyze_code_structure,
        # Policy-scoped validate-only bash (replaces unrestricted bash_command)
        bash.validate_only_bash,
        # CCM write operations
        ccm_tools.add_file_to_changes,
        ccm_tools.update_file_in_changes,
        ccm_tools.update_file_lines,
        ccm_tools.replace_in_file,
        ccm_tools.insert_lines,
        ccm_tools.delete_lines,
        ccm_tools.delete_file_in_changes,
        # CCM read operations
        ccm_tools.get_file_from_changes,
        ccm_tools.list_files_in_changes,
        ccm_tools.get_changes_summary,
        ccm_tools.get_changes_for_pr,
        ccm_tools.export_changes,
        ccm_tools.show_updated_file,
        ccm_tools.get_file_diff,
        ccm_tools.get_session_metadata,
        # Git diff for context
        git_tools.show_diff,
    ):
        fs.add_function(fn)
    return fs


def verification_toolset() -> FunctionToolset[PoCDeepDeps]:
    """Verification toolset - read-only validation with policy-scoped shell."""
    fs = _fs("poc-verification-tools")
    for fn in (
        file_ops.fetch_file,
        file_ops.fetch_files_batch,
        file_ops.analyze_code_structure,
        # Policy-scoped validate-only bash
        bash.validate_only_bash,
        # CCM read-only operations
        ccm_tools.get_changes_summary,
        ccm_tools.get_changes_for_pr,
        ccm_tools.list_files_in_changes,
        git_tools.show_diff,
        git_tools.record_verification_result,
    ):
        fs.add_function(fn)
    return fs


def jira_toolset() -> FunctionToolset[PoCDeepDeps]:
    fs = _fs("poc-jira-tools")
    for fn in (
        integrations.get_jira_issue,
        integrations.search_jira_issues,
        integrations.create_jira_issue,
        integrations.update_jira_issue,
        integrations.add_jira_comment,
        integrations.transition_jira_issue,
        integrations.get_jira_projects,
        integrations.get_jira_project_details,
        integrations.get_jira_project_users,
        integrations.link_jira_issues,
    ):
        fs.add_function(fn)
    return fs


def github_toolset() -> FunctionToolset[PoCDeepDeps]:
    fs = _fs("poc-github-tools")
    for fn in (
        integrations.code_provider_tool,
        git_tools.code_provider_create_branch,
        git_tools.code_provider_create_pr,
        git_tools.code_provider_add_pr_comments,
        git_tools.code_provider_update_file,
        integrations.get_changes_for_pr,
        git_tools.apply_changes,
        git_tools.git_commit,
        git_tools.git_push,
        git_tools.create_pr_workflow,
    ):
        fs.add_function(fn)
    return fs


def confluence_toolset() -> FunctionToolset[PoCDeepDeps]:
    fs = _fs("poc-confluence-tools")
    for fn in (
        integrations.get_confluence_spaces,
        integrations.get_confluence_page,
        integrations.search_confluence_pages,
        integrations.get_confluence_space_pages,
        integrations.create_confluence_page,
        integrations.update_confluence_page,
        integrations.add_confluence_comment,
    ):
        fs.add_function(fn)
    return fs


def linear_toolset() -> FunctionToolset[PoCDeepDeps]:
    fs = _fs("poc-linear-tools")
    for fn in (integrations.get_linear_issue, integrations.update_linear_issue):
        fs.add_function(fn)
    return fs
