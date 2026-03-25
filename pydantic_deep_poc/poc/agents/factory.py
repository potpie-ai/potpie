"""Assemble create_deep_agent for PoC with scoped subagents and OpenRouter model."""

from __future__ import annotations

from typing import Any

from pydantic_deep import create_deep_agent
from subagents_pydantic_ai import create_subagent_toolset

from poc.agents import (
    confluence,
    discover,
    github,
    implement,
    jira,
    linear,
    prompts_loader,
    supervisor,
    verify,
)
from poc.config.provider import get_model, get_model_settings
from poc.config.settings import MODEL_MAX_CONCURRENCY
from poc.managers.deps import PoCDeepDeps
from poc.tools.toolsets_builder import implementation_toolset, supervisor_toolset
from poc.tracing.logfire_tracer import (
    is_logfire_enabled,
    should_instrument_pydantic_ai_runtime,
)


def _subagent_configs() -> list[Any]:
    return [
        discover.subagent_config(),
        implement.subagent_config(),
        verify.subagent_config(),
        jira.subagent_config(),
        github.subagent_config(),
        confluence.subagent_config(),
        linear.subagent_config(),
    ]


def _instrument_flag() -> bool | None:
    if is_logfire_enabled() and should_instrument_pydantic_ai_runtime():
        return True
    return None


def assemble_agent(
    context_block: str = "",
    *,
    include_subagents: bool = True,
    subagents_only: list[str] | None = None,
) -> Any:
    task_prompt = prompts_loader.load_code_gen_task_prompt()
    instructions = prompts_loader.build_supervisor_instructions(
        supervisor.ROLE,
        supervisor.GOAL,
        supervisor.BACKSTORY,
        task_prompt,
        context_block,
    )
    subs = _subagent_configs()
    if subagents_only:
        names = set(subagents_only)
        subs = [s for s in subs if s["name"] in names]

    model = get_model()
    model_settings = get_model_settings()
    inst = _instrument_flag()
    subagent_toolset = create_subagent_toolset(
        id="poc-orchestrator-subagents",
        subagents=subs if include_subagents else None,
        default_model=model,
        include_general_purpose=False,
        max_nesting_depth=0,
        descriptions=prompts_loader.build_task_tool_descriptions(),
    )
    skill_dir = str(prompts_loader.skills_directory().resolve())

    kwargs: dict[str, Any] = dict(
        model=model,
        instructions=instructions,
        toolsets=[supervisor_toolset(), subagent_toolset],
        subagents=None,
        include_todo=False,
        include_filesystem=False,
        include_subagents=False,
        include_skills=True,
        skill_directories=[skill_dir],
        include_plan=False,
        include_general_purpose_subagent=False,
        cost_tracking=False,
        context_manager=False,
        deps_type=PoCDeepDeps,
        model_settings=model_settings,
        max_concurrency=MODEL_MAX_CONCURRENCY,
    )
    if inst is not None:
        kwargs["instrument"] = inst

    return create_deep_agent(**kwargs)


def default_deps(run: Any) -> PoCDeepDeps:
    from pydantic_ai_backends import StateBackend

    async def _sync_subagent_question_fallback(question: str, _choices: list[Any]) -> str:
        return (
            "No live parent-response channel is available in sync mode. "
            "Do not continue guessing. Return a BLOCKED result that includes this exact "
            f"question for the orchestrator to handle asynchronously: {question}"
        )

    return PoCDeepDeps(
        backend=StateBackend(),
        poc_run=run,
        ask_user=_sync_subagent_question_fallback,
    )


def assemble_single_execute_only(context_block: str = "") -> Any:
    """Scenario 1: no subagents; main agent has the implementation tool suite only."""
    task_prompt = prompts_loader.load_code_gen_task_prompt()
    instructions = prompts_loader.build_supervisor_instructions(
        supervisor.ROLE,
        supervisor.GOAL,
        supervisor.BACKSTORY,
        task_prompt,
        context_block,
    )
    model = get_model()
    model_settings = get_model_settings()
    inst = _instrument_flag()
    kwargs: dict[str, Any] = dict(
        model=model,
        instructions=instructions,
        toolsets=[implementation_toolset()],
        subagents=None,
        include_todo=False,
        include_filesystem=False,
        include_subagents=False,
        include_skills=True,
        skill_directories=[str(prompts_loader.skills_directory().resolve())],
        include_plan=False,
        include_general_purpose_subagent=False,
        cost_tracking=False,
        context_manager=False,
        deps_type=PoCDeepDeps,
        model_settings=model_settings,
        max_concurrency=MODEL_MAX_CONCURRENCY,
    )
    if inst is not None:
        kwargs["instrument"] = inst
    return create_deep_agent(**kwargs)
