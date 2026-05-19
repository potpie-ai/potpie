"""Async port for the read-side agentic query loop (``goal=investigate``).

Unlike :class:`AnswerSynthesizerPort` — a single-shot summary over an
already-resolved bundle — a :class:`QueryAgentPort` *drives* retrieval: it
runs an LLM tool loop over the pot's context-graph read tools, deciding which
tools to call (and with what arguments) until it can answer the query.

Implementations return ``None`` on any failure (import, network, validation,
timeout) so the caller can fall back to the deterministic resolve path and the
client always gets a non-empty answer.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from domain.graph_query import ContextGraphQuery
from domain.ports.reconciliation_tools import ToolDescriptor

# Async callable the adapter hands to the agent: ``(tool_name, arguments) ->
# raw tool result dict``. The adapter owns pot-scoping and graph access; the
# agent only decides *what* to call.
ToolRunner = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class QueryAgentStep:
    """One tool invocation the agent made, for the trace surfaced to clients."""

    tool: str
    arguments: dict[str, Any]
    result_kind: str
    result_count: int


@dataclass(slots=True)
class QueryAgentResult:
    """Outcome of an agentic investigation over the context graph."""

    answer: str
    steps: list[QueryAgentStep] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    source_refs: list[Any] = field(default_factory=list)
    confidence: float | None = None
    iterations: int = 0
    usage: dict[str, int | str | None] | None = None


class QueryAgentPort(Protocol):
    async def investigate(
        self,
        request: ContextGraphQuery,
        *,
        tools: list[ToolDescriptor],
        run_tool: ToolRunner,
    ) -> QueryAgentResult | None:
        """Run the agentic loop, or return ``None`` to trigger the resolve fallback."""
        ...
