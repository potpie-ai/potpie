"""Context-engine container for standalone HTTP (env maps; no Potpie DB projects)."""

from __future__ import annotations

import os

from adapters.outbound.github.unavailable_source_control import UnavailableSourceControl
from adapters.outbound.reconciliation.context_graph_tools import (
    ContextGraphReconciliationTools,
)
from adapters.outbound.reconciliation.factory import try_pydantic_deep_reconciliation_agent
from bootstrap.container import ContextEngineContainer, build_container, build_container_with_github_token
from bootstrap.env_pots import merged_pot_repo_map
from bootstrap.http_projects import ExplicitPotResolution
from bootstrap.queue_factory import get_context_graph_job_queue


def build_standalone_context_engine_container() -> ContextEngineContainer:
    """
    Same dependency wiring as production queue selection; pot list from merged env maps.

    GitHub token is optional for raw episodic ingest (Graphiti-only); PR/backfill flows need a token.
    """
    mapping = merged_pot_repo_map()
    if not mapping:
        raise RuntimeError(
            'CONTEXT_ENGINE_POTS env JSON is required, e.g. {"pot-id":"owner/repo"}, '
            "and/or CONTEXT_ENGINE_REPO_TO_POT"
        )
    pots = ExplicitPotResolution(mapping)
    jobs = get_context_graph_job_queue()
    token = (os.getenv("CONTEXT_ENGINE_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN") or "").strip()
    reco = try_pydantic_deep_reconciliation_agent()
    if token:
        container = build_container_with_github_token(
            token=token,
            pots=pots,
            reconciliation_agent=reco,
            jobs=jobs,
        )
    else:
        container = build_container(
            pots=pots,
            source_for_repo=lambda _repo: UnavailableSourceControl(),
            reconciliation_agent=reco,
            jobs=jobs,
        )
    _attach_context_tools(container)
    return container


def _attach_context_tools(container: ContextEngineContainer) -> None:
    """Wire bounded read-only context tools into the reconciliation agent, if supported."""
    agent = container.reconciliation_agent
    graph = container.context_graph
    if agent is None or graph is None:
        return
    setter = getattr(agent, "set_context_tools", None)
    if setter is None:
        return
    setter(ContextGraphReconciliationTools(graph))
