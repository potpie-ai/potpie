"""LLM-backed read-side query agent using ``pydantic_ai.Agent`` with a tool loop.

Unlike the answer synthesizer (single-shot, no tools), this agent *drives*
retrieval: it iteratively calls the pot's context-graph read tools until it can
answer, then returns a structured answer plus the tool trace and the evidence
it gathered.

Graceful degradation: any failure (import, network, validation, timeout,
usage-limit exhaustion) logs and returns ``None`` so the caller falls back to
the deterministic resolve path.

When a :class:`TelemetryPort` is provided, every run emits a
``kind="query_agent"`` :class:`CostEvent` with model id, token usage (when the
SDK exposes it), and wall-time latency; the most recent run's usage is also
exposed via :attr:`last_usage`.
"""

from __future__ import annotations

import asyncio
import json
import logging

from observability import get_logger
import os
import time
from typing import Any

from adapters.outbound.query_agent.prompt import (
    QUERY_AGENT_INSTRUCTIONS,
    build_query_agent_prompt,
)
from domain.graph_query import ContextGraphQuery
from domain.ports.query_agent import (
    QueryAgentResult,
    QueryAgentStep,
    ToolRunner,
)
from domain.ports.reconciliation_tools import ToolDescriptor
from domain.ports.telemetry import CostEvent, NoOpTelemetry, TelemetryPort

logger = get_logger(__name__)

_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_REQUEST_LIMIT = 8
_MAX_ITEMS_TO_LLM = 6
_MAX_FIELD_LEN = 400
_MAX_EVIDENCE = 60


class _RunState:
    """Per-investigation accumulator for the trace + full evidence.

    Tool results are truncated before they reach the LLM (token budget) but
    kept whole here so the response envelope can surface real evidence.
    """

    def __init__(self) -> None:
        self.steps: list[QueryAgentStep] = []
        self.evidence: list[dict[str, Any]] = []
        self.source_refs: list[Any] = []
        self._seen_refs: set[str] = set()

    def record(self, tool: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        rows = result.get("result")
        kind = str(result.get("kind") or result.get("error") or "result")
        if isinstance(rows, list):
            count = len(rows)
            for row in rows:
                if isinstance(row, dict):
                    if len(self.evidence) < _MAX_EVIDENCE:
                        self.evidence.append(row)
                    self._collect_refs(row.get("source_refs"))
        elif isinstance(rows, dict):
            count = 1
            if len(self.evidence) < _MAX_EVIDENCE:
                self.evidence.append(rows)
        else:
            count = 0
        self.steps.append(
            QueryAgentStep(
                tool=tool,
                arguments=dict(args),
                result_kind=kind,
                result_count=count,
            )
        )

    def _collect_refs(self, refs: Any) -> None:
        if not isinstance(refs, list):
            return
        for ref in refs:
            key = ref if isinstance(ref, str) else json.dumps(ref, sort_keys=True, default=str)
            if key in self._seen_refs:
                continue
            self._seen_refs.add(key)
            self.source_refs.append(ref)


def _trim(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()[:_MAX_FIELD_LEN]
    if isinstance(value, list):
        return [_trim(v) for v in value[:_MAX_ITEMS_TO_LLM]]
    if isinstance(value, dict):
        return {k: _trim(v) for k, v in value.items()}
    return value


def _compact_for_llm(result: dict[str, Any]) -> dict[str, Any]:
    """Bound a raw tool result before it re-enters the model context."""
    if result.get("error"):
        return {"error": str(result.get("error"))}
    rows = result.get("result")
    if isinstance(rows, list):
        return {
            "kind": result.get("kind"),
            "count": len(rows),
            "items": [_trim(r) for r in rows[:_MAX_ITEMS_TO_LLM]],
        }
    if isinstance(rows, dict):
        return {"kind": result.get("kind"), "result": _trim(rows)}
    return {"kind": result.get("kind"), "result": rows}


def _describe(desc: ToolDescriptor) -> str:
    schema = desc.json_schema or {}
    if not schema:
        return desc.description
    return f"{desc.description}\nArguments JSON schema: {json.dumps(schema)}"


class PydanticQueryAgent:
    """`QueryAgentPort` backed by a tool-using ``pydantic_ai.Agent``."""

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout_s: float | None = None,
        request_limit: int | None = None,
        telemetry: TelemetryPort | None = None,
    ) -> None:
        # Reuse the synthesis model default so a single env var configures the
        # whole read-side LLM surface; CONTEXT_ENGINE_QUERY_AGENT_MODEL wins
        # when set so the agent can run a stronger model than synthesis.
        self._model = (
            model
            or os.getenv("CONTEXT_ENGINE_QUERY_AGENT_MODEL")
            or os.getenv("CONTEXT_ENGINE_ANSWER_SYNTHESIS_MODEL")
            or "openai-responses:gpt-5.4-mini"
        )
        self._timeout_s = (
            timeout_s
            if timeout_s is not None
            else float(os.getenv("CONTEXT_ENGINE_QUERY_AGENT_TIMEOUT_SECS", _DEFAULT_TIMEOUT_S))
        )
        self._request_limit = (
            request_limit
            if request_limit is not None
            else int(os.getenv("CONTEXT_ENGINE_QUERY_AGENT_REQUEST_LIMIT", _DEFAULT_REQUEST_LIMIT))
        )
        self._telemetry: TelemetryPort = telemetry or NoOpTelemetry()
        self.last_usage: dict[str, int | str | None] | None = None

    async def investigate(
        self,
        request: ContextGraphQuery,
        *,
        tools: list[ToolDescriptor],
        run_tool: ToolRunner,
    ) -> QueryAgentResult | None:
        self.last_usage = None
        try:
            from pydantic import BaseModel, Field
            from pydantic_ai import Agent, Tool
        except ImportError:
            logger.warning("pydantic_ai not installed; skipping agentic query")
            return None

        class AgentAnswer(BaseModel):
            answer: str = Field(description="2-5 sentence grounded answer.")
            citations: list[str] = Field(
                default_factory=list,
                description="source_refs (kind:ref) actually used.",
            )
            confidence: float = Field(
                default=0.5, ge=0.0, le=1.0, description="0-1 confidence."
            )

        state = _RunState()

        def _make_tool(name: str) -> Any:
            async def _tool(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
                args = arguments or {}
                try:
                    result = await run_tool(name, args)
                except Exception as exc:  # noqa: BLE001 - surfaced to the model
                    logger.exception("query agent tool %s failed", name, name=name)
                    return {"error": f"tool_failed:{exc}"}
                state.record(name, args, result)
                return _compact_for_llm(result)

            _tool.__name__ = name
            return _tool

        tool_objs = [
            Tool(_make_tool(d.name), name=d.name, description=_describe(d))
            for d in tools
        ]

        prompt = build_query_agent_prompt(request)
        t0 = time.perf_counter()
        try:
            agent = Agent(
                self._model,
                output_type=AgentAnswer,
                system_prompt=QUERY_AGENT_INSTRUCTIONS,
                tools=tool_objs,
            )
            run_kwargs: dict[str, Any] = {}
            try:
                from pydantic_ai.usage import UsageLimits

                run_kwargs["usage_limits"] = UsageLimits(
                    request_limit=self._request_limit
                )
            except Exception:  # noqa: BLE001 - usage limits are best-effort
                logger.debug("pydantic_ai UsageLimits unavailable", exc_info=True)
            result = await asyncio.wait_for(
                agent.run(prompt, **run_kwargs), timeout=self._timeout_s
            )
        except Exception:
            logger.exception("agentic query failed; falling back to resolve path")
            return None
        latency_ms = int((time.perf_counter() - t0) * 1000)

        usage = _extract_usage(result)
        self.last_usage = {
            "model": self._model,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "latency_ms": latency_ms,
            "iterations": len(state.steps),
        }
        try:
            self._telemetry.record_cost(
                CostEvent(
                    pot_id=request.pot_id,
                    kind="query_agent",
                    model=self._model,
                    input_tokens=_int_or_none(usage.get("input_tokens")),
                    output_tokens=_int_or_none(usage.get("output_tokens")),
                    total_tokens=_int_or_none(usage.get("total_tokens")),
                    latency_ms=latency_ms,
                )
            )
        except Exception:
            logger.debug("telemetry: query_agent cost emission failed", exc_info=True)

        output = getattr(result, "output", None) or getattr(result, "data", None)
        answer_text = getattr(output, "answer", None)
        if not isinstance(answer_text, str) or not answer_text.strip():
            return None
        citations = list(getattr(output, "citations", []) or [])
        confidence = getattr(output, "confidence", None)
        return QueryAgentResult(
            answer=answer_text.strip(),
            steps=state.steps,
            evidence=state.evidence,
            source_refs=state.source_refs or citations,
            confidence=float(confidence) if confidence is not None else None,
            iterations=len(state.steps),
            usage=self.last_usage,
        )


def _extract_usage(result: object) -> dict[str, int | None]:
    """Best-effort pydantic-ai usage extraction across SDK versions."""
    usage_callable = getattr(result, "usage", None)
    try:
        u = usage_callable() if callable(usage_callable) else usage_callable
    except Exception:
        u = None
    if u is None:
        return {}
    inp = getattr(u, "input_tokens", None)
    if inp is None:
        inp = getattr(u, "request_tokens", None)
    out = getattr(u, "output_tokens", None)
    if out is None:
        out = getattr(u, "response_tokens", None)
    total = getattr(u, "total_tokens", None)
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": total}


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
