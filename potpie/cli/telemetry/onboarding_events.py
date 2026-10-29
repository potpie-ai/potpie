from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final, Iterator, Literal

from potpie_context_engine.core.lifecycle import FAILED, SetupPlan, StepResult

from potpie.skills.errors import InvalidSkillsInstallPathError, UnknownAgentTargetError

from .product_analytics import AnalyticsValue, capture_event

_CURRENT_SETUP_RUN_ID: ContextVar[str | None] = ContextVar(
    "potpie_cli_setup_run_id",
    default=None,
)
_CURRENT_ENTRYPOINT: ContextVar[str | None] = ContextVar(
    "potpie_cli_onboarding_entrypoint",
    default=None,
)
_CANONICAL_AGENT_SKILLS_AGENTS = frozenset({"claude", "cursor", "opencode", "codex"})
_CANONICAL_AGENT_SKILLS_SCOPES = frozenset({"global", "project"})
_CANONICAL_AGENT_SKILLS_OUTCOMES = frozenset(
    {"installed", "already_installed", "failed", "cancelled"}
)
_CANONICAL_AGENT_SKILLS_SELECTION_OUTCOMES = frozenset(
    {"selected", "skipped", "cancelled"}
)
_CANONICAL_AGENT_SKILLS_FAILURE_KINDS = frozenset(
    {"invalid_agent", "permission_denied", "filesystem", "unexpected"}
)
_ACTIVATION_FAILURE_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "authentication",
        "authorization",
        "cancellation",
        "dependency",
        "not_implemented",
        "other_expected",
        "selection",
        "unavailable",
        "validation",
    }
)


def begin_setup_run() -> str:
    setup_run_id = f"setup_{uuid.uuid4().hex}"
    _CURRENT_SETUP_RUN_ID.set(setup_run_id)
    return setup_run_id


def current_setup_run_id() -> str | None:
    return _CURRENT_SETUP_RUN_ID.get()


@contextmanager
def onboarding_entrypoint(entrypoint: str) -> Iterator[None]:
    token = _CURRENT_ENTRYPOINT.set(entrypoint)
    try:
        yield
    finally:
        _CURRENT_ENTRYPOINT.reset(token)


def current_entrypoint(default: str) -> str:
    return _CURRENT_ENTRYPOINT.get() or default


def now_ms() -> int:
    return int(time.perf_counter() * 1000)


def elapsed_ms(start_ms: int) -> int:
    return max(now_ms() - start_ms, 0)


def repo_location_kind(repo: str | None) -> str:
    if repo is None or not repo.strip():
        return "none"
    if repo.strip() == ".":
        return "current_directory"
    return "explicit_path"


def capture_setup_started(
    plan: SetupPlan,
    *,
    interactive: bool,
    json_output: bool,
    dry_run: bool = False,
) -> None:
    _capture(
        "cli_onboarding_setup_started",
        "base_setup",
        "setup",
        {
            **_setup_plan_properties(plan),
            "interactive": interactive,
            "json_output": json_output,
            "dry_run": dry_run,
        },
    )


def capture_setup_dry_run_completed(
    *, plan: SetupPlan, planned_step_count: int, hard_step_count: int
) -> None:
    _capture(
        "cli_onboarding_setup_dry_run_completed",
        "base_setup",
        "setup",
        {
            **_setup_plan_properties(plan),
            "dry_run": True,
            "planned_step_count": planned_step_count,
            "hard_step_count": hard_step_count,
        },
    )


def capture_setup_completed(
    *,
    plan: SetupPlan,
    ok: bool,
    duration_ms: int,
    hard_failed_step: str | None,
    soft_warning_count: int,
) -> None:
    name = "cli_onboarding_setup_completed" if ok else "cli_onboarding_setup_incomplete"
    props: dict[str, AnalyticsValue] = {
        **_setup_plan_properties(plan),
        "dry_run": False,
        "duration_ms": duration_ms,
        "soft_warning_count": soft_warning_count,
    }
    if not ok:
        props["incomplete_kind"] = "hard_failure"
    if hard_failed_step is not None:
        props["failure_stage"] = hard_failed_step
    _capture(name, "base_setup", "setup", props)


def capture_setup_incomplete(
    *,
    plan: SetupPlan,
    incomplete_kind: Literal["cancelled"],
    duration_ms: int,
    failure_stage: str,
    dry_run: bool = False,
) -> None:
    _capture(
        "cli_onboarding_setup_incomplete",
        "base_setup",
        "setup",
        {
            **_setup_plan_properties(plan),
            "dry_run": dry_run,
            "incomplete_kind": incomplete_kind,
            "failure_stage": failure_stage,
            "duration_ms": duration_ms,
        },
    )


def capture_wizard_event(
    name: str, *, duration_ms: int | None = None, failed_step: str | None = None
) -> None:
    props: dict[str, AnalyticsValue] = {}
    if duration_ms is not None:
        props["duration_ms"] = duration_ms
    if failed_step is not None:
        props["failure_stage"] = failed_step
    _capture(name, "base_setup", "setup", props)


def capture_project_binding_event(
    name: str,
    *,
    entrypoint: str,
    properties: dict[str, AnalyticsValue] | None = None,
) -> None:
    _capture(name, "project_binding", entrypoint, properties or {})


def capture_onboarding_event(
    name: str,
    *,
    phase: str,
    entrypoint: str,
    properties: dict[str, AnalyticsValue] | None = None,
) -> None:
    _capture(name, phase, entrypoint, properties or {})


def canonical_agent_skills_agent(agent: str) -> str | None:
    """Return a supported agent value for canonical skills-install events."""
    normalized = agent.strip().lower()
    if normalized in _CANONICAL_AGENT_SKILLS_AGENTS:
        return normalized
    return None


def canonical_agent_skills_scope(scope: str) -> str | None:
    """Return a supported scope value for canonical skills-install events."""
    normalized = scope.strip().lower()
    if normalized in _CANONICAL_AGENT_SKILLS_SCOPES:
        return normalized
    return None


def capture_agent_skills_install_outcome(
    *,
    agent: str,
    entrypoint: str,
    scope: str,
    outcome: str,
    duration_ms: int,
    failure_kind: str | None = None,
) -> None:
    """Capture a supported harness installation's terminal outcome."""
    canonical_agent = canonical_agent_skills_agent(agent)
    canonical_scope = canonical_agent_skills_scope(scope)
    canonical_outcome = _canonical_agent_skills_value(
        outcome, _CANONICAL_AGENT_SKILLS_OUTCOMES
    )
    if canonical_agent is None or canonical_scope is None or canonical_outcome is None:
        return
    props: dict[str, AnalyticsValue] = {
        "agent": canonical_agent,
        "scope": canonical_scope,
        "outcome": canonical_outcome,
        "duration_ms": duration_ms,
    }
    if failure_kind is not None:
        canonical_failure_kind = _canonical_agent_skills_value(
            failure_kind, _CANONICAL_AGENT_SKILLS_FAILURE_KINDS
        )
        if canonical_failure_kind is None:
            return
        props["failure_kind"] = canonical_failure_kind
    _capture(
        "cli_onboarding_agent_skills_install_outcome",
        "agent_skills",
        entrypoint,
        props,
    )


def capture_agent_skills_selection_outcome(
    *,
    selection_outcome: str,
    selected_agents: tuple[str, ...],
    duration_ms: int,
    entrypoint: str = "post_setup_agent_skills",
) -> None:
    """Capture the wizard's agent-selection prompt outcome."""
    canonical_selection = _canonical_agent_skills_value(
        selection_outcome, _CANONICAL_AGENT_SKILLS_SELECTION_OUTCOMES
    )
    if canonical_selection is None:
        return
    canonical_selected = (
        _canonical_agent_skills_agents(selected_agents)
        if canonical_selection == "selected"
        else ()
    )
    _capture(
        "cli_onboarding_agent_skills_selection_outcome",
        "agent_skills",
        entrypoint,
        {
            "selection_outcome": canonical_selection,
            "selected_agent_count": len(canonical_selected),
            "selected_agents": canonical_selected,
            "duration_ms": duration_ms,
        },
    )


def _canonical_agent_skills_agents(agents: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        canonical
        for agent in agents
        if (canonical := canonical_agent_skills_agent(agent)) is not None
    )


def _canonical_agent_skills_value(value: str, allowed: frozenset[str]) -> str | None:
    normalized = value.strip().lower()
    if normalized in allowed:
        return normalized
    return None


def capture_github_prompt_shown(*, default_answer: bool) -> None:
    _capture(
        "cli_onboarding_github_prompt_shown",
        "integration_auth",
        "post_setup_github_prompt",
        {"default_answer": default_answer},
    )


def capture_github_prompt_outcome(outcome: str, *, duration_ms: int) -> None:
    name = {
        "accepted": "cli_onboarding_github_prompt_accepted",
        "declined": "cli_onboarding_github_prompt_declined",
        "aborted": "cli_onboarding_github_prompt_aborted",
        "skipped": "cli_onboarding_github_prompt_skipped",
    }.get(outcome, "cli_onboarding_github_prompt_aborted")
    _capture(
        name,
        "integration_auth",
        "post_setup_github_prompt",
        {"duration_ms": duration_ms},
    )


def capture_integration_auth_event(
    name: str,
    *,
    provider: str,
    entrypoint: str,
    duration_ms: int | None = None,
    failure_kind: str | None = None,
) -> None:
    props: dict[str, AnalyticsValue] = {"provider": provider}
    if duration_ms is not None:
        props["duration_ms"] = duration_ms
    if failure_kind is not None:
        props["failure_kind"] = failure_kind
    _capture(name, "integration_auth", entrypoint, props)


def capture_github_auth_event(
    name: str,
    *,
    entrypoint: str,
    duration_ms: int | None = None,
    failure_stage: str | None = None,
    failure_kind: str | None = None,
    browser_opened: bool | None = None,
) -> None:
    props: dict[str, AnalyticsValue] = {}
    if duration_ms is not None:
        props["duration_ms"] = duration_ms
    if failure_stage is not None:
        props["failure_stage"] = failure_stage
    if failure_kind is not None:
        props["failure_kind"] = failure_kind
    if browser_opened is not None:
        props["browser_opened"] = browser_opened
    _capture(name, "integration_auth", entrypoint, props)


def capture_activation_command_outcome(
    *,
    command: Literal["resolve", "search", "status"],
    outcome: Literal["succeeded", "expected_failed", "cancelled"],
    result_kind: Literal["context_result", "status_result"],
    duration_ms: int,
    failure_category: str | None = None,
) -> None:
    props: dict[str, AnalyticsValue] = {
        "command": command,
        "outcome": outcome,
        "result_kind": result_kind,
        "duration_ms": max(duration_ms, 0),
    }
    if failure_category is not None:
        props["failure_category"] = (
            failure_category
            if failure_category in _ACTIVATION_FAILURE_CATEGORIES
            else "other_expected"
        )
    _capture(
        "cli_onboarding_activation_command_outcome",
        "activation",
        "direct_command",
        props,
    )


def capture_context_result_returned(
    *,
    command: Literal["resolve", "search"],
    item_count: int,
    confidence: str,
) -> None:
    bounded_confidence = (
        confidence if confidence in {"high", "medium", "low", "unknown"} else "unknown"
    )
    _capture(
        "cli_onboarding_context_result_returned",
        "activation",
        "direct_command",
        {
            "command": command,
            "result_kind": "non_empty" if item_count > 0 else "empty",
            "item_count": max(item_count, 0),
            "confidence": bounded_confidence,
        },
    )


def sanitized_failure_kind(exc: BaseException) -> str:
    return type(exc).__name__


def agent_skills_failure_kind(exc: BaseException) -> str:
    """Return a bounded, privacy-safe category for skills-install failures."""
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, OSError):
        return "filesystem"
    if isinstance(exc, InvalidSkillsInstallPathError):
        return "filesystem"
    if isinstance(exc, UnknownAgentTargetError):
        return "invalid_agent"
    return "unexpected"


class CliSetupAnalyticsObserver:
    def __init__(self) -> None:
        self.current_or_last_step = "setup_execution"

    def step_started(self, *, step: str, hard: bool) -> None:
        self.current_or_last_step = step
        _capture(
            "cli_onboarding_setup_step_started",
            "base_setup",
            "setup",
            {"step": step, "step_hard": hard},
        )

    def step_completed(self, *, result: StepResult, duration_ms: int) -> None:
        props: dict[str, AnalyticsValue] = {
            "step": result.step,
            "step_state": result.state,
            "step_hard": result.hard,
            "duration_ms": duration_ms,
        }
        _capture("cli_onboarding_setup_step_completed", "base_setup", "setup", props)
        if result.state == FAILED:
            _capture(
                "cli_onboarding_setup_step_failed",
                "base_setup",
                "setup",
                {**props, "failure_stage": result.step},
            )


def _setup_plan_properties(plan: SetupPlan) -> dict[str, AnalyticsValue]:
    repo_kind = repo_location_kind(plan.repo)
    return {
        "mode": plan.mode,
        "host_mode": plan.host_mode,
        "backend": plan.backend,
        "agent": plan.agent,
        "agent_explicit": plan.agent != "claude",
        "scan_requested": plan.scan,
        "assume_yes": plan.assume_yes,
        "repo_provided": repo_kind != "none",
        "repo_explicit": repo_kind == "explicit_path",
        "repo_location_kind": repo_kind,
    }


def _capture(
    name: str,
    phase: str,
    entrypoint: str,
    properties: dict[str, AnalyticsValue],
) -> None:
    props: dict[str, AnalyticsValue] = {
        "onboarding_phase": phase,
        "entrypoint": entrypoint,
        **properties,
    }
    setup_run_id = current_setup_run_id()
    if setup_run_id is not None:
        props["setup_run_id"] = setup_run_id
    capture_event(name, props)
