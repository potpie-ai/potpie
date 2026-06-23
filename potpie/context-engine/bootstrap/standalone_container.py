"""Context-engine container for standalone HTTP (env maps; no Potpie DB projects)."""

from __future__ import annotations

import os

from adapters.outbound.connectors._bench_stubs import register_bench_stubs
from adapters.outbound.connectors.notion import NotionConnector
from adapters.outbound.reconciliation.factory import (
    try_pydantic_deep_reconciliation_agent,
)
from application.services.source_connector_registry import SourceConnectorRegistry
from bootstrap.ingestion_server import (
    IngestionServerContainer,
    build_ingestion_server,
    build_ingestion_server_with_github_token,
)
from bootstrap.env_pots import merged_pot_repo_map
from bootstrap.http_projects import ExplicitPotResolution
from bootstrap.queue_factory import get_context_graph_job_queue


def build_standalone_context_engine_container() -> IngestionServerContainer:
    """
    Same dependency wiring as production queue selection; pot list from merged env maps.

    GitHub token is optional for narrative ingest; PR/backfill flows need a token.
    """
    mapping = merged_pot_repo_map()
    if not mapping:
        raise RuntimeError(
            'CONTEXT_ENGINE_POTS env JSON is required, e.g. {"pot-id":"owner/repo"}, '
            "and/or CONTEXT_ENGINE_REPO_TO_POT"
        )
    pots = ExplicitPotResolution(mapping)
    jobs = get_context_graph_job_queue()
    token = (
        os.getenv("CONTEXT_ENGINE_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN") or ""
    ).strip()
    reco = try_pydantic_deep_reconciliation_agent()
    if token:
        return build_ingestion_server_with_github_token(
            token=token,
            pots=pots,
            reconciliation_agent=reco,
            jobs=jobs,
        )
    # Without a GitHub token the registry still ships with Notion so
    # ``context_status`` returns a non-empty connector manifest.
    registry = SourceConnectorRegistry()
    registry.register(NotionConnector())
    # Bench-time stubs — same rationale as in build_ingestion_server_with_github_token.
    register_bench_stubs(registry)
    return build_ingestion_server(
        pots=pots,
        connectors=registry,
        reconciliation_agent=reco,
        jobs=jobs,
    )
