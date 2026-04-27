"""Reconciliation agent backed by pydantic-deep (PyPI: ``pydantic-deep``).

Upstream: https://github.com/vstorm-co/pydantic-deepagents
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Protocol

from domain.context_events import EventRef
from domain.ports.reconciliation_tools import ReconciliationToolsPort
from domain.reconciliation import ReconciliationPlan, ReconciliationRequest

from adapters.outbound.reconciliation.context_graph_tools import (
    build_initial_context_snapshot,
)
from adapters.outbound.reconciliation.llm_plan_convert import (
    llm_plan_to_reconciliation_plan,
)
from adapters.outbound.reconciliation.llm_plan_schema import LlmReconciliationPlan

logger = logging.getLogger(__name__)

_RECONCILIATION_INSTRUCTIONS = """You are a context-graph reconciliation planner for a software project.

You MUST return only the structured plan schema you are given. Do not execute tools that mutate external systems.

You have access to bounded read-only context tools (when provided) to inspect
existing graph memory before planning:
- context_search(query, limit?, node_labels?): semantic search across the pot's facts
- context_recent_changes(file_path?/function_name?/pr_number?, limit?): recent change history near a target
- context_file_owners(file_path, limit?): owners/reviewers for a file
- context_graph_overview(limit?): pot readiness + activity signal

Use tools to (1) confirm whether this event duplicates, refines, or conflicts
with existing facts, (2) find entities to upsert against rather than creating
duplicates, and (3) discover invalidations when the event supersedes prior
facts. Every important mutation should be justified by either the event
payload or tool-observed facts. Attach `evidence_refs` and a numeric
`confidence` (0.0-1.0) to each mutation when possible.

Rules:
- All structural mutations must belong to the given pot_id partition. Never reference another pot.
- Prefer concise episode titles and bodies that capture decisions and rationale.
- Use stable entity_key strings (e.g. github:pr:owner/repo:123) when upserting entities.
- Only include structural mutations that are justified by the event payload and safe for the repository.
- If unsure, add a warning and keep the plan minimal rather than inventing facts.
- If context tools surface an existing entity, upsert against its key rather than creating a new one.
- When this event supersedes a prior fact, emit an invalidation referencing the prior entity/edge.

Ontology guidance for entity upserts:
- Always add at least one canonical label from the vocabulary below. Do NOT use only generic "Entity".
- Use multiple labels when an entity spans categories (e.g., ["Decision", "Feature"]).
- Include required properties for each label:
  - Fix: fix_type (e.g., "code", "config", "infrastructure", "process"), summary
  - DiagnosticSignal: signal_type (e.g., "metric", "log", "error_rate"), summary
  - Incident: title, severity (e.g., "critical", "high", "medium", "low"), status
  - Alert: title, severity, status
  - Runbook: title
  - Service: name, criticality (e.g., "critical", "standard", "experimental"), lifecycle_state
  - Deployment: version, deployed_at
  - For other labels, infer sensible values from the payload; never leave required properties empty.

Canonical label vocabulary:
- Decision: architectural choices, strategy decisions, ADRs, design choices
- Feature: product features, capabilities, functionality, user-facing work
- Fix: bug fixes, patches, resolutions to specific problems
- BugPattern: recurring bugs, anti-patterns, known failure modes
- DiagnosticSignal: metrics, monitoring signals, observability data, health checks
- Incident: outages, failures, post-mortems, on-call events, production issues
- Alert: alert rules, thresholds, paging rules, monitoring alerts
- Runbook: operational procedures, playbooks, operational guides
- Service: microservices, components, backends, systems, APIs
- Deployment: deployment configs, release strategies, environments
- Document: documentation, guides, READMEs, API docs, wikis
- Environment: staging, production, dev, QA environments
- Person / Team: owners, maintainers, reviewers, stakeholders
- Activity: a timeline happening — verb performed by an actor at a time,
  touching zero or more subjects. Emit one Activity per significant event
  you see in the payload (merge, deploy, work-declaration, decision-made,
  incident-opened, etc.). Required properties: verb, occurred_at, summary.
  Verb is a short snake_case string from this preferred set:
  merged_pr, opened_pr, reviewed_pr, authored_commit, opened_issue,
  state_changed, commented, assigned, deployed, decided,
  declared_progress, declared_completed, performed (fallback).
- Period: daily-bucket rollup for Activities. You do not normally need to
  create Periods by hand — the timeline helper handles this — but if the
  event spans a multi-day effort you may upsert an explicit Period with
  period_kind="declared" and label=<short_slug>.

Edge type guidance:
- Use only these common edge types: DECIDES_FOR, AFFECTS, RESOLVED, MITIGATES, IMPACTS, INDICATES, FIRED_IN, MATCHES_PATTERN, DEPENDS_ON, USES, CALLS, CONTAINS, IMPLEMENTS, EXPOSES, DEPLOYED_TO, HOSTS, OWNS, MEMBER_OF, ONCALL_FOR, REVIEWS, HAS_SIGNAL, HAS_ROOT_CAUSE, RELATED_TO, PERFORMED, TOUCHED, IN_PERIOD
- If unsure which edge type fits, use RELATED_TO as the safe fallback.

Timeline-subgraph rules (apply to EVERY plan you produce):
- Every event worth tracking produces at least one Activity upsert + the
  edges PERFORMED (actor -> Activity), TOUCHED (Activity -> affected
  subject), and IN_PERIOD (Activity -> daily Period bucket).
- Identify the actor (Person or Agent) from the payload when possible —
  commit author, PR merger, assignee, commenter, deploying user, the
  coding-agent session running this reconciliation, etc. If no actor is
  identifiable, emit the Activity without PERFORMED rather than guessing.
- Use the deterministic Activity entity_key shape
  ``timeline:activity:<verb>:<short_hex>`` and the Period entity_key shape
  ``timeline:period:daily:<pot_id>:<YYYY-MM-DD>`` so re-ingestion of the
  same event idempotently upserts the same rows.
- The Activity's ``summary`` should be one short, human-readable line
  naming who did what to what ("alice merged PR #42 in potpie/api"). This
  is what the timeline query tool surfaces to downstream agents — write it
  as if answering "what happened here?".
"""


def _event_ref(request: ReconciliationRequest) -> EventRef:
    return EventRef(
        event_id=request.event.event_id,
        source_system=request.event.source_system,
        pot_id=request.pot_id,
    )


def _user_prompt(
    request: ReconciliationRequest,
    *,
    context_snapshot: dict[str, Any] | None = None,
) -> str:
    ev = request.event
    payload: dict[str, Any] = {
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
    if context_snapshot:
        payload["current_context_snapshot"] = context_snapshot
    return json.dumps(payload, indent=2, default=str)


def _semantic_seed_from_event(request: ReconciliationRequest) -> str | None:
    """Derive a best-effort semantic search seed from the event payload."""
    ev = request.event
    candidates: list[str] = []
    payload = ev.payload or {}
    for key in ("title", "summary", "name", "description"):
        val = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(val, str) and val.strip():
            candidates.append(val.strip())
            break
    if ev.action:
        candidates.append(ev.action)
    if ev.event_type:
        candidates.append(ev.event_type)
    seed = " ".join(candidates).strip()
    return seed[:200] or None


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
        tools: ReconciliationToolsPort | None = None,
    ) -> None:
        import os

        self._model = model or os.getenv(
            "CONTEXT_ENGINE_RECONCILIATION_MODEL", "openai:gpt-5.4-mini"
        )
        self._instructions = instructions or _RECONCILIATION_INSTRUCTIONS
        self._work_recorder: AgentWorkRecorder | None = None
        self._tools_port: ReconciliationToolsPort | None = tools

    def set_work_event_recorder(self, recorder: AgentWorkRecorder | None) -> None:
        self._work_recorder = recorder

    def set_context_tools(self, tools: ReconciliationToolsPort | None) -> None:
        """Attach (or replace) the bounded context tools after construction."""
        self._tools_port = tools

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
            "toolset_version": (
                "context-aware-v1" if self._tools_port is not None else "read-only-plan"
            ),
            "model": self._model,
            "has_context_tools": self._tools_port is not None,
        }

    def _build_agent_tools(self, request: ReconciliationRequest) -> list[Any]:
        """Build pydantic-ai tool callables from the bounded tools port.

        Each callable logs PRE/POST via the existing work-recorder hooks; we
        additionally record ``tool_result`` summaries at return time so
        downstream ledger inspection can reproduce the agent's reasoning path.
        """
        if self._tools_port is None:
            return []
        try:
            from pydantic_ai import Tool  # type: ignore[import-not-found]
        except Exception:
            try:
                from pydantic_deep import Tool  # type: ignore[import-not-found, no-redef]
            except Exception:
                logger.warning(
                    "pydantic-ai/pydantic-deep Tool not importable; skipping agent tool wiring"
                )
                return []

        tools_port = self._tools_port
        descriptors = tools_port.list_tools(request)
        built: list[Any] = []

        def _make_handler(name: str):
            def _handler(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
                return tools_port.execute_read_tool(request, name, arguments or {})

            _handler.__name__ = name
            return _handler

        for desc in descriptors:
            try:
                fn = _make_handler(desc.name)
                built.append(Tool(fn, name=desc.name, description=desc.description))
            except Exception:
                logger.exception("failed to build agent tool %s", desc.name)
        return built

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

        context_snapshot: dict[str, Any] | None = None
        if self._tools_port is not None:
            try:
                context_snapshot = build_initial_context_snapshot(
                    self._tools_port,
                    request,
                    semantic_seed=_semantic_seed_from_event(request),
                )
            except Exception:
                logger.exception("failed to build initial context snapshot")
                context_snapshot = None
            if context_snapshot is not None:
                self._record_work_event(
                    "context_snapshot",
                    title="initial context snapshot",
                    payload={"snapshot_keys": sorted(context_snapshot.keys())},
                )

        tool_callables = self._build_agent_tools(request)

        agent_kwargs: dict[str, Any] = {}
        if tool_callables:
            agent_kwargs["tools"] = tool_callables

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
            **agent_kwargs,
        )
        deps = create_default_deps()
        prompt = _user_prompt(request, context_snapshot=context_snapshot)
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
