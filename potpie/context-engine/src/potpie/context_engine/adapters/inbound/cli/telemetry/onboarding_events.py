from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from potpie.context_engine.domain.lifecycle import FAILED, SetupPlan, StepResult

from .product_analytics import AnalyticsValue, capture_event

_CURRENT_SETUP_RUN_ID: ContextVar[str | None] = ContextVar(
    "potpie_cli_setup_run_id",
    default=None,
)
_CURRENT_ENTRYPOINT: ContextVar[str | None] = ContextVar(
    "potpie_cli_onboarding_entrypoint",
    default=None,
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
) -> None:
    _capture(
        "cli_onboarding_setup_started",
        "base_setup",
        "setup",
        {
            **_setup_plan_properties(plan),
            "interactive": interactive,
            "json_output": json_output,
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
        "duration_ms": duration_ms,
        "soft_warning_count": soft_warning_count,
    }
    if hard_failed_step is not None:
        props["failure_stage"] = hard_failed_step
    _capture(name, "base_setup", "setup", props)


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


def capture_activation_succeeded(
    *, command: str, result_kind: str, item_count: int | None = None
) -> None:
    props: dict[str, AnalyticsValue] = {"command": command, "result_kind": result_kind}
    if item_count is not None:
        props["item_count"] = item_count
    _capture(
        "cli_onboarding_first_use_command_succeeded",
        "activation",
        "direct_command",
        props,
    )
    if result_kind == "context_result":
        _capture(
            "cli_onboarding_first_context_result_returned",
            "activation",
            "direct_command",
            props,
        )


def sanitized_failure_kind(exc: BaseException) -> str:
    return type(exc).__name__


class CliSetupAnalyticsObserver:
    def step_started(self, *, step: str, hard: bool) -> None:
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
