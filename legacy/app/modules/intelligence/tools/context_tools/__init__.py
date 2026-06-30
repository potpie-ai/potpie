"""Agent-facing context graph tools."""

import os


if os.getenv("LEGACY_SKIP_CONTEXT_GRAPH", "").strip().lower() in (
    "1",
    "true",
    "yes",
):

    def create_agent_context_tools(*args, **kwargs):
        return []

else:
    from .agent_context_tools import create_agent_context_tools

__all__ = [
    "create_agent_context_tools",
]
