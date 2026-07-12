"""Engine-only component graph used by :class:`ContextEngine`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from potpie_context_engine.application.services.graph_workbench import (
    GraphWorkbenchService,
)
from potpie_context_engine.application.services.nudge_service import NudgeService
from potpie_context_engine.domain.ports.agent_context import AgentContextPort
from potpie_context_engine.domain.ports.graph.backend import GraphBackend
from potpie_context_engine.domain.ports.context_graph_job_queue import (
    ContextGraphJobQueuePort,
)
from potpie_context_engine.domain.ports.ledger.client import (
    EventLedgerClientPort,
    LedgerHealth,
    LedgerPage,
    LedgerSource,
)
from potpie_context_engine.domain.ports.ledger.cursor import LedgerCursorStorePort
from potpie_context_engine.domain.ports.services.graph_service import GraphService
from potpie_context_engine.domain.ports.services.pot_management import (
    PotManagementService,
)


@dataclass(slots=True)
class LedgerFacade:
    client: EventLedgerClientPort
    cursors: LedgerCursorStorePort

    def status(self) -> LedgerHealth:
        return self.client.health()

    def sources(self, *, pot_id: str) -> list[LedgerSource]:
        return self.client.sources(pot_id=pot_id)

    def query(
        self,
        *,
        pot_id: str,
        source_id: str | None = None,
        kind: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> LedgerPage:
        return self.client.query(
            pot_id=pot_id,
            source_id=source_id,
            kind=kind,
            since=since,
            until=until,
            limit=limit,
        )

    def pull(self, *, pot_id: str, source_id: str, limit: int = 100) -> LedgerPage:
        cursor = self.cursors.get(pot_id=pot_id, source_id=source_id)
        page = self.client.fetch(
            pot_id=pot_id, source_id=source_id, cursor=cursor, limit=limit
        )
        if page.next_cursor is not None:
            self.cursors.set(pot_id=pot_id, cursor=page.next_cursor)
        return page


@dataclass(slots=True)
class EngineComponents:
    agent_context: AgentContextPort
    graph: GraphService
    graph_workbench: GraphWorkbenchService
    pots: PotManagementService
    backend: GraphBackend
    ledger: LedgerFacade
    nudge: NudgeService
    job_queue: ContextGraphJobQueuePort
    profile: str = "local"


__all__ = ["EngineComponents", "LedgerFacade"]
