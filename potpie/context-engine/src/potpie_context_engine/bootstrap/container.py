"""Minimal composition root for legacy Potpie host wiring.

The standalone context-engine server uses ``ingestion_server`` / ``host_wiring``.
The legacy FastAPI app still imports ``build_container`` when mounting agent tools;
this shim preserves that contract without the removed legacy episodic/Neo4j DI graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from potpie_context_engine.domain.ports.pot_resolution import PotResolutionPort
from potpie_context_engine.domain.ports.pot_source_listing import PotSourceListingPort
from potpie_context_engine.domain.ports.reconciliation_agent import (
    ReconciliationAgentPort,
)
from potpie_context_engine.domain.ports.settings import ContextEngineSettingsPort


@dataclass
class ContextEngineContainer:
    settings: ContextEngineSettingsPort
    pots: PotResolutionPort
    source_for_repo: Callable[[str], Any]
    reconciliation_agent: ReconciliationAgentPort | None = None
    jobs: Any = None
    pot_source_listing: PotSourceListingPort | None = None
    source_resolver: Any = None


def build_container(
    *,
    settings: ContextEngineSettingsPort,
    pots: PotResolutionPort,
    source_for_repo: Callable[[str], Any],
    reconciliation_agent: ReconciliationAgentPort | None = None,
    jobs: Any = None,
) -> ContextEngineContainer:
    return ContextEngineContainer(
        settings=settings,
        pots=pots,
        source_for_repo=source_for_repo,
        reconciliation_agent=reconciliation_agent,
        jobs=jobs,
    )
