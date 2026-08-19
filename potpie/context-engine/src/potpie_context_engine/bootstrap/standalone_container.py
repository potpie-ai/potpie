"""Context-engine container for standalone HTTP (env maps; no Potpie DB projects)."""

from __future__ import annotations

import os

from potpie_context_engine.adapters.outbound.reconciliation.factory import (
    try_pydantic_deep_reconciliation_agent,
)
from potpie_context_engine.bootstrap.ingestion_server import (
    IngestionServerContainer,
    build_ingestion_server_with_source_tokens,
)
from potpie_context_engine.bootstrap.env_pots import merged_pot_repo_map
from potpie_context_engine.bootstrap.http_projects import ExplicitPotResolution
from potpie_context_engine.bootstrap.queue_factory import get_context_graph_job_queue
from potpie_context_core.reconciliation_flags import reconciliation_config_from_env


def build_standalone_context_engine_container() -> IngestionServerContainer:
    """
    Same dependency wiring as production queue selection; pot list from merged env maps.

    Code-host tokens are optional for narrative ingest; PR/MR and backfill
    flows need one. GitHub and GitLab are independent — configure either,
    both, or neither. Without any token the registry still ships Notion and
    the bench stubs so ``context_status`` returns a non-empty manifest.
    """
    mapping = merged_pot_repo_map()
    if not mapping:
        raise RuntimeError(
            'CONTEXT_ENGINE_POTS env JSON is required, e.g. {"pot-id":"owner/repo"}, '
            "and/or CONTEXT_ENGINE_REPO_TO_POT"
        )
    pots = ExplicitPotResolution(mapping)
    jobs = get_context_graph_job_queue()
    reconciliation = reconciliation_config_from_env()
    reco = try_pydantic_deep_reconciliation_agent(reconciliation_config=reconciliation)
    return build_ingestion_server_with_source_tokens(
        pots=pots,
        github_token=(os.getenv("CONTEXT_ENGINE_GITHUB_TOKEN") or "").strip(),
        gitlab_token=(os.getenv("CONTEXT_ENGINE_GITLAB_TOKEN") or "").strip(),
        gitlab_url=(os.getenv("CONTEXT_ENGINE_GITLAB_URL") or "").strip() or None,
        reconciliation_agent=reco,
        jobs=jobs,
        reconciliation_config=reconciliation,
    )
