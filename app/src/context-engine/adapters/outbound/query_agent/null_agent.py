"""No-op query agent — returns ``None`` so the caller uses the resolve path."""

from __future__ import annotations

from domain.graph_query import ContextGraphQuery
from domain.ports.query_agent import QueryAgentResult, ToolRunner
from domain.ports.reconciliation_tools import ToolDescriptor


class NullQueryAgent:
    async def investigate(
        self,
        request: ContextGraphQuery,
        *,
        tools: list[ToolDescriptor],
        run_tool: ToolRunner,
    ) -> QueryAgentResult | None:
        _ = (request, tools, run_tool)
        return None
