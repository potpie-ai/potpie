"""CCM tools wrapping ccm_core for pydantic-ai FunctionToolset."""

from __future__ import annotations

from pydantic_ai import RunContext

from poc.managers.deps import PoCDeepDeps
from poc.tools import ccm_core


async def add_file_to_changes(
    ctx: RunContext[PoCDeepDeps],
    path: str,
    content: str,
    description: str = "",
) -> dict:
    return ccm_core.add_file_to_changes(ctx.deps.poc_run, path, content, description)


async def update_file_in_changes(
    ctx: RunContext[PoCDeepDeps], path: str, content: str
) -> dict:
    """Update file content in changes with size guard to prevent JSON truncation."""
    # Size guard: prevent passing huge file contents as JSON argument
    MAX_CONTENT_SIZE = 10_000  # characters
    if len(content) > MAX_CONTENT_SIZE:
        return {
            "status": "error",
            "message": (
                f"Content size ({len(content)} chars) exceeds maximum ({MAX_CONTENT_SIZE} chars). "
                "This tool is for targeted edits only. "
                "Use replace_in_file for string replacements or update_file_lines for line-range edits. "
                "If you need to rewrite a large file, consider breaking it into smaller incremental changes."
            ),
        }
    return ccm_core.update_file_in_changes(ctx.deps.poc_run, path, content)


async def get_file_from_changes(
    ctx: RunContext[PoCDeepDeps], path: str, with_line_numbers: bool = False
) -> str:
    return ccm_core.get_file_from_changes(ctx.deps.poc_run, path, with_line_numbers)


async def update_file_lines(
    ctx: RunContext[PoCDeepDeps],
    path: str,
    start_line: int,
    end_line: int,
    new_content: str,
) -> dict:
    return ccm_core.update_file_lines(
        ctx.deps.poc_run, path, start_line, end_line, new_content
    )


async def replace_in_file(
    ctx: RunContext[PoCDeepDeps], path: str, old_str: str, new_str: str
) -> dict:
    return ccm_core.replace_in_file(ctx.deps.poc_run, path, old_str, new_str)


async def insert_lines(
    ctx: RunContext[PoCDeepDeps],
    path: str,
    line: int,
    content: str,
    insert_after: bool = True,
) -> dict:
    return ccm_core.insert_lines(ctx.deps.poc_run, path, line, content, insert_after)


async def delete_lines(
    ctx: RunContext[PoCDeepDeps],
    path: str,
    start_line: int,
    end_line: int | None = None,
) -> dict:
    return ccm_core.delete_lines(ctx.deps.poc_run, path, start_line, end_line)


async def delete_file_in_changes(ctx: RunContext[PoCDeepDeps], path: str) -> dict:
    return ccm_core.delete_file_in_changes(ctx.deps.poc_run, path)


async def clear_file_from_changes(ctx: RunContext[PoCDeepDeps], path: str) -> dict:
    return ccm_core.clear_file_from_changes(ctx.deps.poc_run, path)


async def clear_all_changes(ctx: RunContext[PoCDeepDeps]) -> dict:
    return ccm_core.clear_all_changes(ctx.deps.poc_run)


async def list_files_in_changes(ctx: RunContext[PoCDeepDeps]) -> list[str]:
    return ccm_core.list_files_in_changes(ctx.deps.poc_run)


async def get_changes_summary(ctx: RunContext[PoCDeepDeps]) -> dict:
    return ccm_core.get_changes_summary(ctx.deps.poc_run)


async def get_changes_for_pr(ctx: RunContext[PoCDeepDeps]) -> str:
    return ccm_core.get_changes_for_pr(ctx.deps.poc_run)


async def export_changes(ctx: RunContext[PoCDeepDeps], target_dir: str) -> dict:
    return ccm_core.export_changes(ctx.deps.poc_run, target_dir)


async def show_updated_file(ctx: RunContext[PoCDeepDeps], path: str) -> str:
    return ccm_core.show_updated_file(ctx.deps.poc_run, path)


async def get_file_diff(ctx: RunContext[PoCDeepDeps], path: str) -> str:
    return ccm_core.get_file_diff(ctx.deps.poc_run, path)


async def revert_file(ctx: RunContext[PoCDeepDeps], path: str) -> dict:
    return ccm_core.revert_file(ctx.deps.poc_run, path)


async def get_session_metadata(ctx: RunContext[PoCDeepDeps]) -> str:
    r = ctx.deps.poc_run
    return f"session_id={r.session_id} project_root={r.project_root} worktree={r.worktree_path}"
