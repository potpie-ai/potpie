"""System instructions + opening prompt for the read-side query agent.

Kept deterministic and bounded so loop latency stays predictable and unit
tests can pin the structure.
"""

from __future__ import annotations

import json
from typing import Any

from domain.graph_query import ContextGraphQuery

QUERY_AGENT_INSTRUCTIONS = """You answer questions about a software project by \
investigating its context graph (a per-project memory of decisions, changes, \
owners, incidents, deployments, and discussions).

You have read-only tools that query the graph. Investigate before answering:

- Start broad (context_search), then narrow with the targeted tools
  (context_coding_preferences, context_infra_topology, context_timeline,
  context_prior_bugs) using concrete file paths, function names, or PR numbers
  surfaced by earlier calls.
- Make follow-up calls when a result hints at something you haven't checked.
  Stop as soon as you can answer — do not call tools you don't need.
- Ground every claim in tool results. Do not invent facts. When a fact has a
  source reference, cite it inline as (kind:ref) and list it under citations.
- Be concise: 2-5 sentences, no markdown headers or bullet lists.
- If the graph has no relevant memory, say exactly:
  "No project context found for this query." and set confidence low.
- Set confidence in [0,1]: high only when multiple tool results corroborate.
"""


def build_query_agent_prompt(request: ContextGraphQuery) -> str:
    """Compact JSON brief handed to the agent as the opening user message."""
    scope = request.scope
    payload: dict[str, Any] = {
        "query": (request.query or "").strip(),
        "intent": request.intent,
        "scope": {
            k: v
            for k, v in {
                "repo_name": scope.repo_name,
                "branch": scope.branch,
                "file_path": scope.file_path,
                "function_name": scope.function_name,
                "pr_number": scope.pr_number,
                "services": list(scope.services),
                "features": list(scope.features),
                "environment": scope.environment,
            }.items()
            if v not in (None, "", [])
        },
        "max_results_per_tool": request.budget.max_items,
    }
    return json.dumps(payload, indent=2, default=str)
