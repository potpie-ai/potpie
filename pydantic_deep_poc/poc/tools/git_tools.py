"""Git / PR tools using worktree + CCM."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic_ai import RunContext

from poc.managers.deps import PoCDeepDeps
from poc.repo_setup import create_worktree
from poc.tools import ccm_core


async def checkout_worktree_branch(
    ctx: RunContext[PoCDeepDeps], branch: str, base_hint: str = "main"
) -> str:
    path = create_worktree(branch)
    ctx.deps.poc_run.worktree_path = path
    ctx.deps.poc_run.project_root = path
    ctx.deps.poc_run.branch_name = branch
    return path


async def apply_changes(ctx: RunContext[PoCDeepDeps]) -> dict:
    return ccm_core.apply_to_worktree(ctx.deps.poc_run)


async def git_commit(
    ctx: RunContext[PoCDeepDeps], message: str, conversation_id: str = ""
) -> str:
    root = Path(ctx.deps.poc_run.worktree_path or ctx.deps.poc_run.project_root)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=False)
    p = subprocess.run(
        ["git", "-C", str(root), "commit", "-m", message],
        capture_output=True,
        text=True,
    )
    return p.stdout + p.stderr or f"exit={p.returncode}"


async def git_push(ctx: RunContext[PoCDeepDeps], remote: str = "origin") -> str:
    root = Path(ctx.deps.poc_run.worktree_path or ctx.deps.poc_run.project_root)
    p = subprocess.run(
        ["git", "-C", str(root), "push", remote],
        capture_output=True,
        text=True,
    )
    return f"(PoC stub push) exit={p.returncode}\n{p.stdout}{p.stderr}"


async def show_diff(ctx: RunContext[PoCDeepDeps]) -> str:
    return ccm_core.show_diff_all(ctx.deps.poc_run)


async def record_verification_result(
    ctx: RunContext[PoCDeepDeps], status: str, report: str = ""
) -> dict:
    normalized = status.strip().lower()
    if normalized not in {"pass", "fail"}:
        return {"ok": False, "error": "status must be 'pass' or 'fail'"}
    ctx.deps.poc_run.verification_passed = normalized == "pass"
    ctx.deps.poc_run.verification_report = report.strip()
    return {
        "ok": True,
        "verification_passed": ctx.deps.poc_run.verification_passed,
        "report": ctx.deps.poc_run.verification_report,
    }


async def get_verification_status(ctx: RunContext[PoCDeepDeps]) -> dict:
    return {
        "verification_passed": ctx.deps.poc_run.verification_passed,
        "report": ctx.deps.poc_run.verification_report,
    }


async def create_pr_workflow(
    ctx: RunContext[PoCDeepDeps],
    branch_name: str = "",
    commit_message: str = "",
    pr_title: str = "",
    pr_body: str = "",
    base_branch: str = "main",
) -> str:
    return (
        f"[PoC stub] create_pr_workflow branch={branch_name} title={pr_title!r} "
        f"base={base_branch} — would open PR in production."
    )


async def code_provider_create_branch(
    ctx: RunContext[PoCDeepDeps], branch_name: str
) -> str:
    return f"[stub] would create branch {branch_name}"


async def code_provider_create_pr(ctx: RunContext[PoCDeepDeps], title: str, body: str) -> str:
    return f"[stub] PR {title!r}"


async def code_provider_add_pr_comments(ctx: RunContext[PoCDeepDeps], pr_number: int, body: str) -> str:
    return f"[stub] comment on PR {pr_number}"


async def code_provider_update_file(
    ctx: RunContext[PoCDeepDeps], path: str, content: str
) -> str:
    return f"[stub] update {path}"


async def code_provider_add_pr_comments(
    ctx: RunContext[PoCDeepDeps], pr_number: int, body: str
) -> str:
    return f"[stub] PR comment on #{pr_number}"
