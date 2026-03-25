"""Requirement verification (in-memory list)."""

from __future__ import annotations

from pydantic_ai import RunContext

from poc.managers.deps import PoCDeepDeps


async def add_requirements(ctx: RunContext[PoCDeepDeps], markdown: str) -> str:
    lines = [ln.strip() for ln in markdown.splitlines() if ln.strip()]
    ctx.deps.poc_run.requirements = lines
    return f"stored {len(lines)} requirements"


async def get_requirements(ctx: RunContext[PoCDeepDeps]) -> str:
    return "\n".join(ctx.deps.poc_run.requirements) or "(none)"


async def delete_requirements(ctx: RunContext[PoCDeepDeps]) -> str:
    ctx.deps.poc_run.requirements.clear()
    return "ok"
