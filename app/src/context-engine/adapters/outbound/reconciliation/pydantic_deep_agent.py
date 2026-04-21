"""Reconciliation agent backed by pydantic-deep (PyPI: ``pydantic-deep``).

Upstream: https://github.com/vstorm-co/pydantic-deepagents
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Protocol

from domain.context_events import EventRef
from domain.reconciliation import ReconciliationPlan, ReconciliationRequest

from adapters.outbound.reconciliation.llm_plan_convert import (
    llm_plan_to_reconciliation_plan,
)
from adapters.outbound.reconciliation.llm_plan_schema import LlmReconciliationPlan

logger = logging.getLogger(__name__)

_RECONCILIATION_INSTRUCTIONS = """You are a context-graph reconciliation planner for a software project.

You MUST return only the structured plan schema you are given. Do not execute tools that mutate external systems.

Rules:
- All structural mutations must belong to the given pot_id partition. Never reference another pot.
- Prefer concise episode titles and bodies that capture decisions and rationale.
- Use stable entity_key strings (e.g. github:pr:owner/repo:123) when upserting entities.
- Only include structural mutations that are justified by the event payload and safe for the repository.
- If unsure, add a warning and keep the plan minimal rather than inventing facts.
"""


def _event_ref(request: ReconciliationRequest) -> EventRef:
    return EventRef(
        event_id=request.event.event_id,
        source_system=request.event.source_system,
        pot_id=request.pot_id,
    )


def _user_prompt(request: ReconciliationRequest) -> str:
    ev = request.event
    payload = {
        "pot_id": request.pot_id,
        "repo_name": request.repo_name,
        "prior_attempts": request.prior_attempts,
        "event": {
            "event_id": ev.event_id,
            "source_system": ev.source_system,
            "event_type": ev.event_type,
            "action": ev.action,
            "pot_id": ev.pot_id,
            "provider": ev.provider,
            "provider_host": ev.provider_host,
            "repo_name": ev.repo_name,
            "source_id": ev.source_id,
            "source_event_id": ev.source_event_id,
            "artifact_refs": ev.artifact_refs,
            "occurred_at": ev.occurred_at.isoformat() if ev.occurred_at else None,
            "received_at": ev.received_at.isoformat() if ev.received_at else None,
            "payload": ev.payload,
        },
    }
    return json.dumps(payload, indent=2, default=str)


def _pydantic_deep_version() -> str:
    try:
        from importlib.metadata import version

        return version("pydantic-deep")
    except Exception:
        return "unknown"


class AgentWorkRecorder(Protocol):
    def record(
        self,
        event_kind: str,
        *,
        title: str | None = None,
        body: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None: ...


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except TypeError:
        return str(value)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return {"value": _jsonable(value)}


class PydanticDeepReconciliationAgent:
    """`ReconciliationAgentPort` using pydantic-deep structured output."""

    def __init__(
        self,
        *,
        model: str | None = None,
        instructions: str | None = None,
    ) -> None:
        import os

        self._model = model or os.getenv(
            "CONTEXT_ENGINE_RECONCILIATION_MODEL", "openai:gpt-5.4-mini"
        )
        self._instructions = instructions or _RECONCILIATION_INSTRUCTIONS
        self._work_recorder: AgentWorkRecorder | None = None

    def set_work_event_recorder(self, recorder: AgentWorkRecorder | None) -> None:
        self._work_recorder = recorder

    def _record_work_event(
        self,
        event_kind: str,
        *,
        title: str | None = None,
        body: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._work_recorder is None:
            return
        try:
            self._work_recorder.record(
                event_kind,
                title=title,
                body=body,
                payload=payload,
            )
        except Exception:
            logger.exception("failed to record reconciliation agent work event")

    def capability_metadata(self) -> dict[str, Any]:
        return {
            "agent": "pydantic-deep",
            "version": _pydantic_deep_version(),
            "toolset_version": "read-only-plan",
            "model": self._model,
        }

    def run_reconciliation(self, request: ReconciliationRequest) -> ReconciliationPlan:
        try:
            from pydantic_deep import (
                Hook,
                HookEvent,
                HookResult,
                create_deep_agent,
                create_default_deps,
            )
        except ImportError as exc:
            raise ImportError(
                "pydantic-deep is required for PydanticDeepReconciliationAgent. "
                "Install: pip install 'context-engine[reconciliation-agent]'"
            ) from exc

        async def _record_pre_tool_use(hook_input: Any) -> Any:
            self._record_work_event(
                "tool_call",
                title=getattr(hook_input, "tool_name", None),
                payload={
                    "event": getattr(hook_input, "event", None),
                    "tool_name": getattr(hook_input, "tool_name", None),
                    "tool_input": _json_dict(getattr(hook_input, "tool_input", {})),
                },
            )
            return HookResult(allow=True)

        async def _record_post_tool_use(hook_input: Any) -> Any:
            result = getattr(hook_input, "tool_result", None)
            self._record_work_event(
                "tool_result",
                title=getattr(hook_input, "tool_name", None),
                body=str(result)[:20000] if result is not None else None,
                payload={
                    "event": getattr(hook_input, "event", None),
                    "tool_name": getattr(hook_input, "tool_name", None),
                    "tool_input": _json_dict(getattr(hook_input, "tool_input", {})),
                    "tool_error": getattr(hook_input, "tool_error", None),
                },
            )
            return HookResult(allow=True)

        agent = create_deep_agent(
            model=self._model,
            instructions=self._instructions,
            output_type=LlmReconciliationPlan,
            include_todo=False,
            include_filesystem=False,
            include_subagents=False,
            include_skills=False,
            include_plan=False,
            include_web=False,
            include_memory=False,
            include_teams=False,
            include_checkpoints=False,
            include_general_purpose_subagent=False,
            context_manager=False,
            cost_tracking=False,
            include_history_archive=False,
            hooks=[
                Hook(event=HookEvent.PRE_TOOL_USE, handler=_record_pre_tool_use),
                Hook(event=HookEvent.POST_TOOL_USE, handler=_record_post_tool_use),
                Hook(
                    event=HookEvent.POST_TOOL_USE_FAILURE, handler=_record_post_tool_use
                ),
            ],
        )
        deps = create_default_deps()
        prompt = _user_prompt(request)
        self._record_work_event(
            "prompt",
            title="reconciliation request",
            body=prompt,
            payload={
                "model": self._model,
                "pot_id": request.pot_id,
                "repo_name": request.repo_name,
                "event_id": request.event.event_id,
                "source_system": request.event.source_system,
            },
        )

        async def _run() -> LlmReconciliationPlan:
            result = await agent.run(prompt, deps=deps)
            try:
                messages = json.loads(result.new_messages_json().decode("utf-8"))
            except Exception:
                logger.exception(
                    "failed to serialize pydantic-deep reconciliation messages"
                )
                messages = []
            self._record_work_event(
                "model_messages",
                title="agent message history",
                payload={
                    "messages": messages,
                    "usage": _json_dict(result.usage()),
                    "pydantic_ai_run_id": getattr(result, "run_id", None),
                },
            )
            return result.output

        try:
            llm_plan = asyncio.run(_run())
        except Exception:
            self._record_work_event("error", title="agent run failed")
            logger.exception("pydantic-deep reconciliation run failed")
            raise

        ref = _event_ref(request)
        plan = llm_plan_to_reconciliation_plan(llm_plan, event_ref=ref)
        self._record_work_event(
            "plan_output",
            title="structured reconciliation plan",
            body=plan.summary,
            payload={
                "episode_count": len(plan.episodes),
                "entity_mutation_count": len(plan.entity_upserts),
                "edge_mutation_count": len(plan.edge_upserts) + len(plan.edge_deletes),
                "warning_count": len(plan.warnings),
            },
        )
        return plan
