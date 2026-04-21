"""Build :class:`ContextEngineContainer` for CLI (env + optional git-scoped pot resolution)."""

from __future__ import annotations

import os

from adapters.inbound.cli.cli_pot_resolution import CliPotResolution
from adapters.outbound.github.unavailable_source_control import UnavailableSourceControl
from adapters.outbound.reconciliation.factory import (
    try_pydantic_deep_reconciliation_agent,
)
from bootstrap.container import (
    ContextEngineContainer,
    build_container,
    build_container_with_github_token,
)
from bootstrap.env_pots import merged_pot_repo_map
from domain.ports.jobs import NoOpJobEnqueue


def build_cli_container(
    *,
    cwd: str | None = None,
) -> ContextEngineContainer:
    """
    Container for ``potpie`` CLI and local MCP ingest: merged pot maps, GitHub when token set.

    Always uses :class:`NoOpJobEnqueue` for the job queue so the CLI never imports Potpie Celery, Redis,
    or worker-only modules. Persisted ingest still writes to Postgres when configured; agent planning
    and episode apply run in-process. For true broker-backed async, use the HTTP
    API (or a worker entrypoint), not this container.

    Raises:
        RuntimeError: if no pot mapping can be built (set CONTEXT_ENGINE_POTS / CONTEXT_ENGINE_REPO_TO_POT
            or ``potpie pot use`` with a git checkout for origin fallback).
    """
    mapping = merged_pot_repo_map()
    pots = CliPotResolution(mapping, cwd=cwd)
    jobs = NoOpJobEnqueue()
    reco = try_pydantic_deep_reconciliation_agent()
    token = (
        os.getenv("CONTEXT_ENGINE_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN") or ""
    ).strip()
    if token:
        return build_container_with_github_token(
            token=token,
            pots=pots,
            reconciliation_agent=reco,
            jobs=jobs,
        )
    return build_container(
        pots=pots,
        source_for_repo=lambda _repo: UnavailableSourceControl(),
        reconciliation_agent=reco,
        jobs=jobs,
    )
