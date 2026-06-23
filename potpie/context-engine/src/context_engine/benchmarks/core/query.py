"""Wrappers around context_resolve / context_status / context_search."""

from __future__ import annotations

from typing import Any

from context_engine.adapters.outbound.http.potpie_context_api_client import PotpieContextApiClient

from context_engine.benchmarks.core.scenario import QuerySpec


def resolve_context(
    client: PotpieContextApiClient, pot_id: str, query: QuerySpec
) -> dict[str, Any]:
    """Issue a context_resolve call. Returns the agent envelope."""
    # One mode-based read contract: request the canonical evidence envelope
    # (goal=retrieve → _orchestrate → envelope_to_dict). There is no
    # server-side answer synthesis; the invariant judge synthesises from the
    # returned facts/evidence itself.
    body: dict[str, Any] = {
        "pot_id": pot_id,
        "goal": "retrieve",
        "intent": query.intent,
        "include": list(query.include),
        "mode": query.mode,
        "source_policy": query.source_policy,
    }
    if query.scope:
        body["scope"] = dict(query.scope)
    return client.context_graph_query(body)
