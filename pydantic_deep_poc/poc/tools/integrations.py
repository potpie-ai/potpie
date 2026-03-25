"""Stub integrations: Jira, GitHub, Confluence, Linear."""

from __future__ import annotations

from pydantic_ai import RunContext

from poc.managers.deps import PoCDeepDeps

STUB = "[PoC stub]"


async def _ok(name: str, **kwargs: object) -> str:
    return f"{STUB} {name} args={kwargs!r}"


async def get_jira_issue(ctx: RunContext[PoCDeepDeps], key: str) -> str:
    return await _ok("get_jira_issue", key=key)


async def search_jira_issues(ctx: RunContext[PoCDeepDeps], jql: str) -> str:
    return await _ok("search_jira_issues", jql=jql)


async def create_jira_issue(ctx: RunContext[PoCDeepDeps], project: str, summary: str) -> str:
    return await _ok("create_jira_issue", project=project, summary=summary)


async def update_jira_issue(ctx: RunContext[PoCDeepDeps], key: str, fields: str) -> str:
    return await _ok("update_jira_issue", key=key)


async def add_jira_comment(ctx: RunContext[PoCDeepDeps], key: str, body: str) -> str:
    return await _ok("add_jira_comment", key=key)


async def transition_jira_issue(ctx: RunContext[PoCDeepDeps], key: str, name: str) -> str:
    return await _ok("transition_jira_issue", key=key, name=name)


async def get_jira_projects(ctx: RunContext[PoCDeepDeps]) -> str:
    return await _ok("get_jira_projects")


async def get_jira_project_details(ctx: RunContext[PoCDeepDeps], key: str) -> str:
    return await _ok("get_jira_project_details", key=key)


async def get_jira_project_users(ctx: RunContext[PoCDeepDeps], key: str) -> str:
    return await _ok("get_jira_project_users", key=key)


async def link_jira_issues(ctx: RunContext[PoCDeepDeps], inward: str, outward: str) -> str:
    return await _ok("link_jira_issues", inward=inward, outward=outward)


async def code_provider_tool(
    ctx: RunContext[PoCDeepDeps],
    repo_name: str = "",
    issue_number: int | None = None,
    is_pull_request: bool = False,
) -> str:
    return await _ok(
        "code_provider_tool",
        repo=repo_name,
        issue_number=issue_number,
        is_pull_request=is_pull_request,
    )


async def get_changes_for_pr(ctx: RunContext[PoCDeepDeps], conversation_id: str = "") -> str:
    from poc.tools import ccm_core

    return ccm_core.get_changes_for_pr(ctx.deps.poc_run)


async def get_confluence_spaces(ctx: RunContext[PoCDeepDeps]) -> str:
    return await _ok("get_confluence_spaces")


async def get_confluence_page(ctx: RunContext[PoCDeepDeps], page_id: str) -> str:
    return await _ok("get_confluence_page", page_id=page_id)


async def search_confluence_pages(ctx: RunContext[PoCDeepDeps], query: str) -> str:
    return await _ok("search_confluence_pages", query=query)


async def get_confluence_space_pages(ctx: RunContext[PoCDeepDeps], space: str) -> str:
    return await _ok("get_confluence_space_pages", space=space)


async def create_confluence_page(ctx: RunContext[PoCDeepDeps], space: str, title: str, body: str) -> str:
    return await _ok("create_confluence_page", space=space, title=title)


async def update_confluence_page(ctx: RunContext[PoCDeepDeps], page_id: str, body: str) -> str:
    return await _ok("update_confluence_page", page_id=page_id)


async def add_confluence_comment(ctx: RunContext[PoCDeepDeps], page_id: str, body: str) -> str:
    return await _ok("add_confluence_comment", page_id=page_id)


async def get_linear_issue(ctx: RunContext[PoCDeepDeps], issue_id: str) -> str:
    return await _ok("get_linear_issue", issue_id=issue_id)


async def update_linear_issue(ctx: RunContext[PoCDeepDeps], issue_id: str, fields: str) -> str:
    return await _ok("update_linear_issue", issue_id=issue_id)
