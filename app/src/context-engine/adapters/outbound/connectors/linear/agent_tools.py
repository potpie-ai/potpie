"""Linear agent tools: fetch issue detail via the host's ``LinearIssueFetcher``.

Mirrors :mod:`adapters.outbound.connectors.github.agent_tools`. The fetcher
resolves the Linear access token *per pot* (walking
``pot_id → context_graph_pot_sources → integrations`` and decrypting the
stored OAuth token), so reads run against whatever Linear account the pot's
owner connected — never a shared service key. The host wires this in via
``PydanticDeepReconciliationAgent.add_extra_tools([build_linear_tools(...)])``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from adapters.outbound.connectors.linear.fetcher import LinearIssueFetcher

logger = logging.getLogger(__name__)


def build_linear_tools(
    fetcher: LinearIssueFetcher,
) -> Callable[[Any], list[Any]]:
    """Return a per-batch tool builder that exposes Linear read endpoints.

    Args:
        fetcher: Resolves a Linear issue by identifier for a given pot.
            Typically ``ContextEngineLinearFetcher`` (per-pot OAuth token).

    Returns:
        A callable matching the agent's ``add_extra_tools`` contract. The
        builder closes over ``state.pot_id`` so token resolution stays scoped
        to the pot being reconciled.
    """

    def _builder(state: Any) -> list[Any]:
        try:
            from pydantic_ai import Tool  # type: ignore[import-not-found]
        except Exception:
            try:
                from pydantic_deep import Tool  # type: ignore[import-not-found, no-redef]
            except Exception:
                logger.warning(
                    "pydantic-ai/pydantic-deep Tool not importable; skipping linear tools"
                )
                return []

        pot_id = getattr(state, "pot_id", None)

        def linear_get_issue(issue_id: str) -> dict[str, Any]:
            """Fetch one Linear issue's detail by identifier (``ENG-123`` or UUID)."""
            try:
                issue = fetcher.get_issue(issue_id, pot_id=pot_id)
            except PermissionError as exc:
                # Auth/token problem for this pot's Linear connection — surface
                # it so the agent adds a warning instead of inventing facts.
                return {"error": "linear_auth_failed", "message": str(exc)}
            except Exception as exc:
                logger.exception("linear_get_issue %s failed", issue_id)
                return {"error": str(exc)}
            if issue is None:
                return {"found": False, "issue_id": issue_id}
            return {"found": True, "issue": issue}

        return [
            Tool(
                linear_get_issue,
                name="linear_get_issue",
                description=(
                    "Fetch one Linear issue by its identifier (team-prefixed "
                    "key like 'ENG-123' or the opaque Linear UUID). Returns the "
                    "issue detail (title, description, state, assignee, team, "
                    "labels, timestamps) for the Linear account connected to "
                    "this pot. Use it to ground issue references found in "
                    "commits, PR bodies, or webhook payloads."
                ),
            ),
        ]

    return _builder


__all__ = ["build_linear_tools"]
