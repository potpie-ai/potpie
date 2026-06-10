"""Minimal composition root for legacy Potpie host wiring.

The standalone context-engine server uses ``ingestion_server`` / ``host_wiring``.
The legacy FastAPI app still imports ``build_container`` when mounting agent tools;
this shim preserves that contract without the removed legacy episodic/Neo4j DI graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from domain.ports.pot_resolution import PotResolutionPort
from domain.ports.pot_source_listing import PotSourceListingPort
from domain.ports.reconciliation_agent import ReconciliationAgentPort
from domain.ports.settings import ContextEngineSettingsPort


class _ResolutionServiceShim:
    def __init__(self) -> None:
        self._source_resolver: Any = None

    def set_source_resolver(self, resolver: Any) -> None:
        self._source_resolver = resolver


@dataclass
class ContextEngineContainer:
    settings: ContextEngineSettingsPort
    pots: PotResolutionPort
    source_for_repo: Callable[[str], Any]
    reconciliation_agent: ReconciliationAgentPort | None = None
    jobs: Any = None
    pot_source_listing: PotSourceListingPort | None = None
    source_resolver: Any = None
    resolution_service: _ResolutionServiceShim = field(
        default_factory=_ResolutionServiceShim
    )


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
